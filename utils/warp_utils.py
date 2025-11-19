import torch
from typing import Tuple, Optional
import numpy as np
import os
import cv2
from PIL import Image
from utils.general_utils import PILtoTorch
from tqdm import tqdm
from torch import Tensor
import torch.nn.functional as F


def warp_to_pseudo_cams(args, train_cameras, virtul_cameras):
    use_mask = args.use_mask
    bg = torch.tensor([0, 0, 0], dtype=torch.float32, device=args.data_device)
    for virtul_cam in virtul_cameras:
        n_frames = len(virtul_cam.nearest_names_warp)
        ref_cams = []
        src_intr = virtul_cam.get_intrinsics()
        src_extr = virtul_cam.get_extrinsics()
        for i in range(n_frames):
            tmp = [cam for cam in train_cameras if cam.image_name == virtul_cam.nearest_names_warp[i]]
            ref_cams.append(tmp[0])
        for train_cam in ref_cams:
            ref_img, ref_img_gray = train_cam.get_image(bg, with_mask = use_mask)    # [3, H, W]  [1, H, W]
            # ref_mask = train_cam.get_gtMask(with_mask = use_mask)                    # [1, H, W]
            ref_mask = train_cam.get_pmaps(use_mask)              # 用多视图过滤后的mask作为ref_mask
            ref_mono = train_cam.get_mono(with_mask = use_mask)                      # [4, H, W]
            ref_normal = ref_mono[:3]                                                # [3, H, W]
            ref_depth = train_cam.get_alignedD(with_mask = use_mask)                 # [1, H, W]
            ref_intr = train_cam.get_intrinsics()
            ref_extr = train_cam.get_extrinsics()
            _, H, W = ref_img.shape

            trans_points, transformation = compute_transformed_points(ref_depth, ref_extr, src_extr, ref_intr, \
                                            src_intr, ref_mask, device=args.data_device)       # [H, W, 3, 1] 训练相机投影到虚拟相机的像素坐标, [H, W, 1, 1] 训练相机mask
            trans_coordinates = trans_points[:, :, :2, 0] / trans_points[:, :, 2:3, 0]         # [H, W, 2] 训练相机投影到虚拟相机的像素坐标
            trans_depth = trans_points[:, :, 2, 0].unsqueeze(0)              # [1, H, W] 训练相机深度图的每个像素深度投影到虚拟相机后的深度值 != 虚拟相机深度图

            ref_normal_flat = ref_normal.reshape([3, -1])
            trans_normal_flat = transformation[:3, :3] @ ref_normal_flat
            trans_normal = trans_normal_flat.reshape([3, H, W])

            grid = create_grid(H, W).to(args.data_device)                    # [2, H, W]
            flow12 = trans_coordinates.permute(2, 0, 1) - grid               # [2, H, W] 训练相机投影到虚拟相机的像素坐标 - grid

            warped_frame, warped_mask = bilinear_splatting(args, ref_img, ref_mask, trans_depth, flow12, None, "image")   # [3, H, W]  [1, H, W]
            warped_depth = bilinear_splatting(args, trans_depth, ref_mask, trans_depth, flow12, None, "depth")[0]           # [1, H, W]
            warped_normal = bilinear_splatting(args, trans_normal, ref_mask, trans_depth, flow12, None, "normal")[0]        # [3, H, W]

            warp_path = os.path.join(args.model_path, "warp")
            os.makedirs(warp_path, exist_ok=True)
            ref_img_show = ((ref_img).permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
            ref_normal_show = (((ref_normal+1.0)*0.5).permute(1,2,0).clamp(0,1)*255).detach().cpu().numpy().astype(np.uint8)
            warp_img_show = ((warped_frame).permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
            warped_mask_show = (warped_mask.float()*255).cpu().numpy().astype(np.uint8).reshape(H,W)
            warped_mask_show = cv2.applyColorMap(warped_mask_show, cv2.COLORMAP_JET)
            warped_normal_show = (((warped_normal+1.0)*0.5).permute(1,2,0).clamp(0,1)*255).detach().cpu().numpy().astype(np.uint8)
            warped_depth_show = (warped_depth * warped_mask).squeeze().detach().cpu().numpy()
            warped_depth_show = (warped_depth_show - warped_depth_show.min()) / (warped_depth_show.max() - warped_depth_show.min() + 1e-20)
            warped_depth_show = (warped_depth_show*255).clip(0, 255).astype(np.uint8)
            warped_depth_show = cv2.applyColorMap(warped_depth_show, cv2.COLORMAP_JET)
            row0 = np.concatenate([ref_img_show, warp_img_show, warped_mask_show], axis=1)
            row1 = np.concatenate([ref_normal_show, warped_normal_show, warped_depth_show], axis=1)
            image_to_show = np.concatenate([row0, row1], axis=0)
            filename = f"{virtul_cam.image_name}_{train_cam.image_name}"
            cv2.imwrite(os.path.join(warp_path, filename + ".jpg"), image_to_show)

            warped_depth, filter = depth_filtering(warped_depth, src_intr, save_filter_path=None, device=args.data_device, save_filter=False)
            warped_frame *= filter
            warped_mask *= filter
            warped_normal *= filter

            virtul_cam.original_image = warped_frame.detach().cpu()
            tmp_rgb = Image.fromarray((warped_frame.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8))
            tmp_gray = tmp_rgb.convert("L")
            virtul_cam.image_gray = PILtoTorch(tmp_gray)
            virtul_cam.mask = warped_mask.detach().cpu().float()
            virtul_cam.mono = torch.cat([warped_normal.detach().cpu(), torch.ones_like(warped_depth, device="cpu")], dim=0)
            virtul_cam.alignedD = warped_depth.detach().cpu()
            virtul_cam.consist_mask = warped_mask.detach().cpu().float()
            
            # warped_depth_to_ply(args, warped_frame, warped_depth, warped_normal, src_extr, src_intr, filename)





def compute_transformed_points(ref_depth, ref_extr, src_extr, ref_intr, src_intr, ref_mask, device):
    _, H, W = ref_depth.shape
    transformation = torch.matmul(src_extr, torch.linalg.inv(ref_extr))   # 训练相机到虚拟相机转换矩阵
    x1d = torch.arange(0, W)[None]
    y1d = torch.arange(0, H)[:, None]
    x2d = x1d.repeat([H, 1]).to(device)                    # [H, W]
    y2d = y1d.repeat([1, W]).to(device)                    # [H, W]
    ones_2d = torch.ones(size=(H, W)).to(device)           # [H, W]
    ones_4d = ones_2d[:, :, None, None]                    # [H, W, 1, 1]
    pos_vectors_homo = torch.stack([x2d, y2d, ones_2d], dim=2)[:, :, :, None]      # [H, W, 3, 1]

    ref_intr_inv = torch.linalg.inv(ref_intr)
    ref_intr_inv_4d = ref_intr_inv[None, None]      # [1, 1, 3, 3]
    src_intr_4d = src_intr[None, None]              # [1, 1, 3, 3]
    depth_4d = ref_depth[0][:, :, None, None]       # [H, W, 1, 1]
    trans_4d = transformation[None, None]           # [1, 1, 4, 4]

    unnormalized_pos = torch.matmul(ref_intr_inv_4d, pos_vectors_homo)   # [H, W, 3, 1]
    world_points = depth_4d * unnormalized_pos                           # [H, W, 3, 1]   训练相机深度图投影到自身相机系
    world_points_homo = torch.cat([world_points, ones_4d], dim=2)        # [H, W, 4, 1]
    trans_world_homo = torch.matmul(trans_4d, world_points_homo)         # [H, W, 4, 1]   转到虚拟相机系
    trans_world = trans_world_homo[:, :, :3]                             # [H, W, 3, 1]
    trans_norm_points = torch.matmul(src_intr_4d, trans_world)           # [H, W, 3, 1]   在虚拟相机中像素坐标

    return trans_norm_points, transformation

def create_grid(h, w):
    x_1d = torch.arange(0, w)[None]
    y_1d = torch.arange(0, h)[:, None]
    x_2d = x_1d.repeat([h, 1])
    y_2d = y_1d.repeat([1, w])
    grid = torch.stack([x_2d, y_2d], dim=0)   # [2, H, W]

    return grid

def bilinear_splatting(args, frame1: torch.Tensor, mask1: Optional[torch.Tensor], depth1: torch.Tensor,
                        flow12: torch.Tensor, flow12_mask: Optional[torch.Tensor], type: str = "image") -> \
        Tuple[torch.Tensor, torch.Tensor]:
    """
    Bilinear splatting
    :param frame1: (b,c,h,w)
    :param mask1: (b,1,h,w): 1 for known, 0 for unknown. Optional
    :param depth1: (b,1,h,w)
    :param flow12: (b,2,h,w)
    :param flow12_mask: (b,1,h,w): 1 for valid flow, 0 for invalid flow. Optional
    :param is_image: if true, output will be clipped to (-1,1) range
    :return: warped_frame2: (b,c,h,w)
            mask2: (b,1,h,w): 1 for known and 0 for unknown
    """

    c, h, w = frame1.shape
    if mask1 is None:
        mask1 = torch.ones(size=(1, h, w)).to(args.data_device)           # [1, H, W]
    if flow12_mask is None:
        flow12_mask = torch.ones(size=(1, h, w)).to(args.data_device)     # [1, H, W]
    grid = create_grid(h, w).to(args.data_device)                         # [2, H, W]
    trans_pos = flow12 + grid                                             # [2, H, W]  训练相机投影到虚拟相机的像素坐标

    trans_pos_offset = trans_pos + 1                                      # [2, H, W]  训练相机投影到虚拟相机的像素坐标 + 1
    trans_pos_floor = torch.floor(trans_pos_offset).long()                # [2, H, W]  训练相机投影到虚拟相机的像素坐标 + 1 向下取整
    trans_pos_ceil = torch.ceil(trans_pos_offset).long()                  # [2, H, W]  训练相机投影到虚拟相机的像素坐标 + 1 向上取整
    trans_pos_offset = torch.stack([
        torch.clamp(trans_pos_offset[0, ...], min=0, max=w + 1),
        torch.clamp(trans_pos_offset[1, ...], min=0, max=h + 1)], dim=0)  # [2, H, W]  训练相机投影到虚拟相机的像素坐标 + 1
    trans_pos_floor = torch.stack([
        torch.clamp(trans_pos_floor[0, ...], min=0, max=w + 1),
        torch.clamp(trans_pos_floor[1, ...], min=0, max=h + 1)], dim=0)   # [2, H, W]  训练相机投影到虚拟相机的像素坐标 + 1 向下取整
    trans_pos_ceil = torch.stack([
        torch.clamp(trans_pos_ceil[0, ...], min=0, max=w + 1),
        torch.clamp(trans_pos_ceil[1, ...], min=0, max=h + 1)], dim=0)    # [2, H, W]  训练相机投影到虚拟相机的像素坐标 + 1 向上取整

    prox_weight_nw = (1 - (trans_pos_offset[1, ...] - trans_pos_floor[1, ...])) * \
                     (1 - (trans_pos_offset[0, ...] - trans_pos_floor[0, ...]))        # [H, W]
    prox_weight_sw = (1 - (trans_pos_ceil[1, ...] - trans_pos_offset[1, ...])) * \
                     (1 - (trans_pos_offset[0, ...] - trans_pos_floor[0, ...]))        # [H, W]
    prox_weight_ne = (1 - (trans_pos_offset[1, ...] - trans_pos_floor[1, ...])) * \
                     (1 - (trans_pos_ceil[0, ...] - trans_pos_offset[0, ...]))         # [H, W]
    prox_weight_se = (1 - (trans_pos_ceil[1, ...] - trans_pos_offset[1, ...])) * \
                     (1 - (trans_pos_ceil[0, ...] - trans_pos_offset[0, ...]))

    sat_depth1 = torch.clamp(depth1, min=0, max=1000)                  # [1, H, W]
    log_depth1 = torch.log(1 + sat_depth1)                             # [1, H, W]
    depth_weights = torch.exp(log_depth1 / log_depth1.max() * 50)      # [1, H, W]

    weight_nw = torch.moveaxis((prox_weight_nw[None] * mask1 * flow12_mask / depth_weights)[None], [0, 1, 2, 3], [0, 3, 1, 2])  # [1, W, 1, H]
    weight_sw = torch.moveaxis((prox_weight_sw[None] * mask1 * flow12_mask / depth_weights)[None], [0, 1, 2, 3], [0, 3, 1, 2])
    weight_ne = torch.moveaxis((prox_weight_ne[None] * mask1 * flow12_mask / depth_weights)[None], [0, 1, 2, 3], [0, 3, 1, 2])
    weight_se = torch.moveaxis((prox_weight_se[None] * mask1 * flow12_mask / depth_weights)[None], [0, 1, 2, 3], [0, 3, 1, 2])

    warped_frame = torch.zeros(size=(1, h + 2, w + 2, c), dtype=torch.float32).to(args.data_device)
    warped_weights = torch.zeros(size=(1, h + 2, w + 2, 1), dtype=torch.float32).to(args.data_device)

    frame1_cl = torch.moveaxis(frame1[None], [0, 1, 2, 3], [0, 3, 1, 2])             # [1, W, 3, H]
    batch_indices = torch.arange(1)[:, None, None].to(args.data_device)              # [1, 1, 1]
    warped_frame.index_put_((batch_indices, trans_pos_floor[1][None], trans_pos_floor[0][None]),
                            torch.einsum('bhwc,bhwk->bhwc', frame1_cl, weight_nw), accumulate=True)
    warped_frame.index_put_((batch_indices, trans_pos_ceil[1][None], trans_pos_floor[0][None]),
                            torch.einsum('bhwc,bhwk->bhwc', frame1_cl, weight_sw), accumulate=True)
    warped_frame.index_put_((batch_indices, trans_pos_floor[1][None], trans_pos_ceil[0][None]),
                            torch.einsum('bhwc,bhwk->bhwc', frame1_cl, weight_ne), accumulate=True)
    warped_frame.index_put_((batch_indices, trans_pos_ceil[1][None], trans_pos_ceil[0][None]),
                            torch.einsum('bhwc,bhwk->bhwc', frame1_cl, weight_se), accumulate=True)

    warped_weights.index_put_((batch_indices, trans_pos_floor[1][None], trans_pos_floor[0][None]),
                               weight_nw, accumulate=True)
    warped_weights.index_put_((batch_indices, trans_pos_ceil[1][None], trans_pos_floor[0][None]),
                               weight_sw, accumulate=True)
    warped_weights.index_put_((batch_indices, trans_pos_floor[1][None], trans_pos_ceil[0][None]),
                               weight_ne, accumulate=True)
    warped_weights.index_put_((batch_indices, trans_pos_ceil[1][None], trans_pos_ceil[0][None]),
                               weight_se, accumulate=True)

    warped_frame_cf = torch.moveaxis(warped_frame, [0, 1, 2, 3], [0, 2, 3, 1])
    warped_weights_cf = torch.moveaxis(warped_weights, [0, 1, 2, 3], [0, 2, 3, 1])
    cropped_warped_frame = warped_frame_cf[:, :, 1:-1, 1:-1]
    cropped_weights = warped_weights_cf[:, :, 1:-1, 1:-1]

    mask = cropped_weights > 0
    zero_value = -1 if type == "image" else 0
    zero_tensor = torch.tensor(zero_value, dtype=frame1.dtype, device=args.data_device)
    warped_frame2 = torch.where(mask, cropped_warped_frame / cropped_weights, zero_tensor)
    mask2 = mask.to(args.data_device)                                            # [1, 1, H, W]

    if type == "image":
        assert warped_frame2.min() >= -1.1  # Allow for rounding errors
        assert warped_frame2.max() <= 1.1
        warped_frame2 = torch.clamp(warped_frame2, min=0, max=1)                                    # [1, 3, H, W]
    if type == "normal":
        norm_frame2 = torch.norm(warped_frame2, p=2, dim=1, keepdim=True).repeat(1, 3, 1, 1)        # [1, 3, H, W]
        zero_normal = (norm_frame2 == 0)                                                            # [1, 3, H, W]
        warped_frame2[~zero_normal] = warped_frame2[~zero_normal] / norm_frame2[~zero_normal]
        warped_frame2[zero_normal] = 0

    return warped_frame2.squeeze(0), mask2.squeeze(0)

# def virtul_cameras_filter(self, args, resolution_scale):
#             pmaps_dir = os.path.join(args.model_path, "aligned_pmaps")
#             os.makedirs(pmaps_dir, exist_ok=True)
#             print("computing depth probability map")
#             for id, cur_cam in enumerate(self.train_cameras[resolution_scale]):
#                 nearest_cam = []
#                 nearest_names = cur_cam.nearest_names_filter
#                 for nearest_name in nearest_names:
#                     src_cam = [cam for cam in self.getTrainCameras(resolution_scale) if cam.image_name == nearest_name]
#                     nearest_cam.append(src_cam[0])
#                 nearest_cam = None if len(nearest_cam) == 0 else nearest_cam
                    
#                 ref_intrinsics = cur_cam.get_intrinsics()
#                 ref_extrinsics = cur_cam.get_extrinsics()
#                 ref_depth_est = cur_cam.get_alignedD(args.use_mask)
#                 ref_depth_est = ref_depth_est.squeeze()

#                 if nearest_cam != None:
#                     w_ref, h_ref = ref_depth_est.shape[1], ref_depth_est.shape[0]
#                     depth_mask = (ref_depth_est > 0).to(args.data_device)
#                     dynamic_consistency = torch.zeros((h_ref, w_ref), device=args.data_device)
#                     num_consistent = torch.zeros((h_ref, w_ref), device=args.data_device)
#                     for src_cam in nearest_cam:
#                         src_intrinsics = src_cam.get_intrinsics()
#                         src_extrinsics = src_cam.get_extrinsics()
#                         src_depth_est = src_cam.get_alignedD(args.use_mask)
#                         src_depth_est = src_depth_est.squeeze()
#                         y_ref, x_ref = torch.meshgrid(torch.arange(0, h_ref, device=args.data_device), torch.arange(0, w_ref, device=args.data_device))        # p
#                         depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src, mask_reprojected = self.reproject_with_depth(args, ref_depth_est, ref_intrinsics, ref_extrinsics,    
#                                                                                                                                 src_depth_est, src_intrinsics, src_extrinsics)    # d', p', p', q, q
#                         dist = torch.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)
#                         depth_diff = torch.abs(depth_reprojected - ref_depth_est)            # [H, W]
#                         relative_depth_diff = depth_diff / (ref_depth_est + 1e-8)            # [H, W]
#                         err_mask = (dist < 5) & (relative_depth_diff < 0.02)

#                         if not torch.any(err_mask):
#                             continue
#                         tmp_index = dist + 200 * relative_depth_diff
#                         dynamic_consistency[err_mask] += torch.exp(-tmp_index[err_mask])
#                         num_consistent[err_mask] += 1

#                     if torch.any(num_consistent):
#                         multi_view_mask = (num_consistent >= 1) #& (dynamic_consistency > 0.1 * num_consistent)
#                         remove_mask = depth_mask & (~multi_view_mask)
#                         cur_cam.consist_mask = multi_view_mask[None].clone().detach().cpu()
                            
#                         if args.save_color_pmaps and id % 1 == 0:
#                             image_array = np.zeros((h_ref, w_ref, 3), dtype=np.uint8)
#                             image_array[multi_view_mask.cpu().numpy()] = [255, 255, 255]
#                             image_array[remove_mask.cpu().numpy()] = [0, 0, 255]  # OpenCV 使用 BGR 顺序
#                             save_path = os.path.join(pmaps_dir, f'{os.path.splitext(os.path.basename(cur_cam.image_path))[0]}.png')
#                             cv2.imwrite(save_path, image_array)

def depth_filtering(
    depth: Tensor, K: Tensor, save_filter_path: None, device: "cuda", save_filter: bool=False, offset: int=3
) -> Tensor:
    _, h, w = depth.shape
    y_ref, x_ref = torch.meshgrid(torch.arange(0, h, device=device), torch.arange(0, w, device=device))  
    x_ref, y_ref = x_ref.reshape([-1]), y_ref.reshape([-1])
    xyz_ref = torch.matmul(torch.linalg.inv(K),
                                torch.vstack((x_ref, y_ref, torch.ones_like(x_ref))) * depth.reshape([-1])).to(device)
    xyz_ref = xyz_ref.reshape(3, h, w)                      # [3, H, W]   
    xyz_ref = F.pad(xyz_ref, (offset, offset, offset, offset), mode='replicate')        # [3, H+2*offeset, W+2*offset]                         

    bottom_point = xyz_ref[:, 2*offset: h+2*offset, offset: w+offset    ]
    top_point    = xyz_ref[:, 0: h,                 offset: w+offset    ]
    right_point  = xyz_ref[:, offset: h+offset,     2*offset: w+2*offset]
    left_point   = xyz_ref[:, offset: h+offset,     0: w                ]
    midle_point  = xyz_ref[:, offset: h+offset,     offset: w+offset    ]

    dx_ = left_point - midle_point
    dx = right_point - midle_point
    dy_ = bottom_point - midle_point
    dy = top_point - midle_point

    dx_ = dx_ / (torch.linalg.norm(dx_, axis=0, keepdims=True) + 1e-8)
    dx = dx / (torch.linalg.norm(dx, axis=0, keepdims=True) + 1e-8)
    dy_ = dy_ / (torch.linalg.norm(dy_, axis=0, keepdims=True) + 1e-8)
    dy = dy / (torch.linalg.norm(dy, axis=0, keepdims=True) + 1e-8)

    mask_dx_ = dx_[-1, :, :] < 0
    mask_dx_ = mask_dx_.unsqueeze(0).expand_as(dx_)
    mask_dx = dx[-1, :, :] < 0
    mask_dx = mask_dx.unsqueeze(0).expand_as(dx)
    mask_dy_ = dy_[-1, :, :] < 0
    mask_dy_ = mask_dy_.unsqueeze(0).expand_as(dy_)
    mask_dy = dy[-1, :, :] < 0
    mask_dy = mask_dy.unsqueeze(0).expand_as(dy)
    dx_[mask_dx_] *= -1         # [3, H, W]
    dx[mask_dx] *= -1
    dy_[mask_dy_] *= -1
    dy[mask_dy] *= -1

    vectors = torch.stack([dx_, dx, dy_, dy], dim=0)  # [4, 3, H, W]
    z_values = vectors[:, -1, :, :]                   # [4, H, W]
    max_z_indices = torch.argmax(z_values, dim=0)     # [H, W]
    max_z_indices = max_z_indices.unsqueeze(0).unsqueeze(1).expand(1, 3, h, w)  # [1, 3, H, W]
    max_grad_vec = torch.gather(vectors, 0, max_z_indices)  # [1, 3, H, W]
    max_grad_vec = max_grad_vec.squeeze(0)                  # [3, H, W]
    depth_scale_mask = torch.where((depth > 0.001) & (depth < 20), torch.tensor(1), torch.tensor(0)).to(torch.bool)       # 太远的点全局校正不准，需要去掉    [1, H, W]
                                                                                                                                       # 这里深度上限要比计算系数时的上限取大一点
    y_c, x_c = torch.meshgrid(torch.arange(0, h, device=device), torch.arange(0, w, device=device))
    nx = (x_c - K[0, -1]) / K[0, 0]
    ny = (y_c - K[1, -1]) / K[1, 1]
    nz = torch.ones_like(nx)  
    nc = torch.stack([nx, ny, nz], axis=0)
    nc = nc / torch.linalg.norm(nc, axis=0, keepdims=True)       # [3, H, W]

    cos_seta = torch.sum(max_grad_vec * nc, axis=0)     # [H, W]
    angle = torch.arccos(cos_seta)                      # [H, W]
    angle = angle.unsqueeze(0)
    mask = ((angle * 180 / 3.1415) > 8) & depth_scale_mask     # [1, H, W]
    depth = depth * mask                                       # [1, H, W]

    if save_filter:
        save_mask = mask.cpu().numpy().squeeze().astype(np.uint8)
        cv2.imwrite(save_filter_path, save_mask * 255)

    return depth, mask

def warped_depth_to_ply(args, img, depth, normal, extr, intr, filename):

    intr = intr.cpu().numpy()
    fx, fy, cx, cy = intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]
    Tw2c = extr.cpu().numpy()
    img = img.permute(1, 2, 0).detach().cpu().numpy()
    normal = normal.permute(1, 2, 0).detach().cpu().numpy()
    depth = depth.squeeze().detach().cpu().numpy()
    mask = (depth > 0)

    height, width = depth.shape
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    x = (x - cx) / fx
    y = (y - cy) / fy
    z = depth  
    x = x * z
    y = y * z
    x = x[mask]
    y = y[mask]
    z = z[mask]
    
    normal = normal.reshape(-1 ,3)[mask.flatten()]
    normal = normal @ Tw2c[:3, :3]                  # normal转到世界系                 

    colors = 255 * img.reshape(-1, 3)[mask.flatten()]

    points_camera = np.vstack((x.flatten(), y.flatten(), z.flatten(), np.ones_like(z.flatten())))
    points_world = np.linalg.inv(Tw2c) @ points_camera
    points_world = points_world[:3, :].T
    points_with_color_normal = np.hstack((points_world, colors, normal))

    dir_path = os.path.join(args.model_path, "warped_pcd/")
    os.makedirs(dir_path, exist_ok=True)

    def save_ply(points, plyname):
        with open(plyname, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("property float nx\n")
            f.write("property float ny\n")
            f.write("property float nz\n")
            f.write("end_header\n")
            np.savetxt(f, points, fmt="%f %f %f %d %d %d %f %f %f")

    save_ply(points_with_color_normal, dir_path + f'{filename}.ply')
