"""Some useful util functions"""

import os
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from natsort import natsorted
from torch import Tensor
from utils import colormaps
from cuml.neighbors import NearestNeighbors
import imageio
import skimage
from time import time

# Depth Scale Factor m to mm
SCALE_FACTOR = 0.001


def get_filename_list(image_dir: Path, ends_with: Optional[str] = None) -> List:
    """List directory and save filenames

    Returns:
        image_filenames
    """
    image_filenames = os.listdir(image_dir)
    if ends_with is not None:
        image_filenames = [
            image_dir / name
            for name in image_filenames
            if name.lower().endswith(ends_with)
        ]
    else:
        image_filenames = [image_dir / name for name in image_filenames]
    image_filenames = natsorted(image_filenames)
    return image_filenames



def depth_path_to_tensor(
    depth_path: Path, scale_factor: float = SCALE_FACTOR, return_color=False
) -> Tensor:
    """Load depth image in either .npy or .png format and return tensor

    Args:
        depth_path: Path
        scale_factor: float
        return_color: bool
    Returns:
        depth tensor and optionally colored depth tensor
    """
    if depth_path.suffix == ".png":
        depth = cv2.imread(str(depth_path.absolute()), cv2.IMREAD_ANYDEPTH)
    elif depth_path.suffix == ".npy":
        depth = np.load(depth_path, allow_pickle=True)
        if len(depth.shape) == 3:
            depth = depth[..., 0]
    elif depth_path.suffix == ".npz":
        npz_data = np.load(depth_path)
        depth = npz_data[npz_data.files[0]]
        if len(depth.shape) == 3:
            depth = depth[..., 0]
    else:
        raise Exception(f"Format is not supported {depth_path.suffix}")
    depth = depth * scale_factor
    depth = depth.astype(np.float32)
    depth = torch.from_numpy(depth).unsqueeze(-1)
    if not return_color:
        return depth
    else:
        depth_color = colormaps.apply_depth_colormap(depth)
        return depth, depth_color  # type: ignore

def load_mask(path, type):
    if type == "sky":
        alpha = imageio.imread(str(path.absolute()), pilmode='F')
        sky = (alpha > 150)
        alpha[sky] = 0
        alpha[~sky] = 255
        alpha = skimage.img_as_float32(alpha) / 255
    elif type == "object":
        alpha = imageio.imread(str(path.absolute()), pilmode='F')
        alpha = skimage.img_as_float32(alpha) / 255
        white = alpha > 0
        alpha[white] = 1
    return alpha[..., None]
    

def mask_path_to_tensor(
    mask_path: Path
) -> Tensor:
    """Load depth image in either .npy or .png format and return tensor

    Args:
        depth_path: Path
        scale_factor: float
        return_color: bool
    Returns:
        depth tensor and optionally colored depth tensor
    """
    if mask_path.suffix == ".png":
        mask = cv2.imread(str(mask_path.absolute()), cv2.IMREAD_GRAYSCALE)

    mask = torch.from_numpy(mask).unsqueeze(-1)

    return mask

def depth_filtering(
    depth: Tensor, K: Tensor, save_filter_path: None, device: str="cuda", save_filter: bool=False, scale_mask: Tensor=None, offset: int=3, mode: str="dense"
) -> Tensor:
    depth = depth.to(device)
    K = K.to(device)
    scale_mask = scale_mask.to(device)
    if mode == "dense":
        h, w = depth.shape[0], depth.shape[1]
        y_ref, x_ref = torch.meshgrid(torch.arange(0, h, device=device), torch.arange(0, w, device=device))  
        x_ref, y_ref = x_ref.reshape([-1]), y_ref.reshape([-1])
        xyz_ref = torch.matmul(torch.linalg.inv(K),
                                torch.vstack((x_ref, y_ref, torch.ones_like(x_ref, device=device))) * depth.reshape([-1])).to(device)
        xyz_ref = xyz_ref.reshape(3, h, w)                      # [3, H, W]   
        xyz_ref = F.pad(xyz_ref, (offset, offset, offset, offset), mode='replicate').permute(1, 2, 0)        # [H+2*offeset, W+2*offset, 3]                         

        bottom_point = xyz_ref[2*offset: h+2*offset, offset: w+offset,     ...]
        top_point    = xyz_ref[0: h,                 offset: w+offset,     ...]
        right_point  = xyz_ref[offset: h+offset,     2*offset: w+2*offset, ...]
        left_point   = xyz_ref[offset: h+offset,     0: w,                 ...]
        midle_point  = xyz_ref[offset: h+offset,     offset: w+offset,     ...]

        dx_ = left_point - midle_point
        dx = right_point - midle_point
        dy_ = bottom_point - midle_point
        dy = top_point - midle_point

        dx_ = dx_ / (torch.linalg.norm(dx_, axis=-1, keepdims=True) + 1e-8)
        dx = dx / (torch.linalg.norm(dx, axis=-1, keepdims=True) + 1e-8)
        dy_ = dy_ / (torch.linalg.norm(dy_, axis=-1, keepdims=True) + 1e-8)
        dy = dy / (torch.linalg.norm(dy, axis=-1, keepdims=True) + 1e-8)

        dx_[:, :, :][dx_[..., -1] < 0] *= -1         # [H, W, 3]
        dx[:, :, :][dx[..., -1] < 0] *= -1
        dy_[:, :, :][dy_[..., -1] < 0] *= -1
        dy[:, :, :][dy[..., -1] < 0] *= -1

        vectors = torch.stack([dx_, dx, dy_, dy], dim=0)  # [4, H, W, 3]
        z_values = vectors[..., -1]                       # [4, H, W]
        max_z_indices = torch.argmax(z_values, dim=0)     # [H, W]
        max_z_indices = max_z_indices.unsqueeze(0).unsqueeze(-1).expand(1, h, w, 3)  # [1, H, W, 3]
        max_grad_vec = torch.gather(vectors, 0, max_z_indices)  # [H, W, 3]
        depth_scale_mask = torch.where((depth > 0.001) & (depth < 50) & scale_mask, torch.tensor(1), torch.tensor(0)).to(torch.bool)   # 太远的点全局校正不准，需要去掉    [H, W]
                                                                                                                                       # 这里深度上限要比计算系数时的上限取大一点
        y_c, x_c = torch.meshgrid(torch.arange(0, h, device=device), torch.arange(0, w, device=device))
        nx = (x_c - K[0, -1]) / K[0, 0]
        ny = (y_c - K[1, -1]) / K[1, 1]
        nz = torch.ones_like(nx)  
        nc = torch.stack([nx, ny, nz], axis=-1)
        nc = nc / torch.linalg.norm(nc, axis=-1, keepdims=True)       # [H, W, 3]

        cos_seta = torch.sum(max_grad_vec * nc, axis=-1)
        angle = torch.arccos(cos_seta).permute(1, 2, 0)
        mask = ((angle * 180 / 3.1415) > 8) & depth_scale_mask
        depth = depth * mask                                # [H, W, 1]

        if save_filter:
            save_mask = mask.cpu().numpy().squeeze().astype(np.uint8)
            cv2.imwrite(save_filter_path, save_mask * 255)

        return depth, mask
    
    elif mode == "sparse":
        K_NEIGHBORS = 5
        valid_mask = (depth.squeeze() > 0)
        valid_pixels = torch.nonzero(valid_mask, as_tuple=False)   # [N, 2] N为有效像素个数
        if valid_pixels.shape[0] > K_NEIGHBORS:
            knn = NearestNeighbors(n_neighbors=K_NEIGHBORS)
            knn.fit(valid_pixels.to(device))
            dists, ids = knn.kneighbors(valid_pixels.to(device))
            ids = torch.from_numpy(ids[:, 1:].get())                   # [N, K_NEIGHBORS-1]
            
            h, w = depth.shape[0], depth.shape[1]
            y_ref, x_ref = torch.meshgrid(torch.arange(0, h, device=device), torch.arange(0, w, device=device))  
            x_ref, y_ref = x_ref.reshape([-1]), y_ref.reshape([-1])
            xyz_ref = torch.matmul(torch.linalg.inv(K),
                                    torch.vstack((x_ref, y_ref, torch.ones_like(x_ref))) * depth.reshape([-1])).to(device)
            xyz_ref = xyz_ref.reshape(3, h, w)                      # [3, H, W]
            pixels_id = valid_pixels[ids]                           # [N, K_NEIGHBORS-1, 2] N个有效像素的各自最近邻有效像素在[H, W]中索引
            ori_points = xyz_ref[:, valid_pixels[:, 0], valid_pixels[:, 1]]  # [3, N] 源有效像素的3D坐标
            grad_vecs = []
            grad_depths = []
            for i in range(K_NEIGHBORS-1):
                pixels_id_row = pixels_id[:, i, 0]
                pixels_id_col = pixels_id[:, i, 1]
                neighbor_points = xyz_ref[:, pixels_id_row, pixels_id_col]       # [3, N] 邻近有效像素的3D坐标
                grad_vec = (neighbor_points - ori_points) / torch.linalg.norm((neighbor_points - ori_points), axis=0, keepdims=True)  # [3, N] 源像素指向邻近像素的方向向量
                grad_depth = neighbor_points[-1, :] - ori_points[-1, :]
                grad_vecs.append(grad_vec)
                grad_depths.append(grad_depth.unsqueeze(0))
            grad_vecs = torch.stack(grad_vecs, dim=0)                            # [K_NEIGHBORS-1, 3, N]
            grad_depths = torch.stack(grad_depths, dim=0)                        # [K_NEIGHBORS-1, 1, N]
            grad_vecs[(grad_vecs[:, -1, :] < 0).unsqueeze(1).expand(-1, 3, -1)] *= -1  # [K_NEIGHBORS-1, 3, N]
            grad_depths[(grad_depths[:, 0, :] < 0).unsqueeze(1)] *= -1                 # [K_NEIGHBORS-1, 1, N]
            z_values = grad_vecs[:, -1, :]                                       # [K_NEIGHBORS-1, N]
            max_z_indices = torch.argmax(z_values, dim=0)                        # [N]
            max_z_indices = max_z_indices.unsqueeze(0).unsqueeze(1).expand(1, 3, max_z_indices.shape[-1])  # [1, 3, N]
            max_grad_vec = torch.gather(grad_vecs, 0, max_z_indices).squeeze(0)  # [3, N]
            z_values = grad_depths[:, 0, :]                                      # [K_NEIGHBORS-1, N]
            max_z_indices = torch.argmax(z_values, dim=0)                        # [N]
            max_z_indices = max_z_indices.unsqueeze(0).unsqueeze(1).expand(1, 1, max_z_indices.shape[-1])  # [1, 1, N]
            max_grad_depth = torch.gather(grad_depths, 0, max_z_indices).squeeze()  # [N]

            y_c, x_c = torch.meshgrid(torch.arange(0, h, device=device), torch.arange(0, w, device=device))
            nx = (x_c - K[0, -1]) / K[0, 0]
            ny = (y_c - K[1, -1]) / K[1, 1]
            nz = torch.ones_like(nx)  
            nc = torch.stack([nx, ny, nz], axis=0)
            nc = nc / torch.linalg.norm(nc, axis=0, keepdims=True)              # [3, H, W]
            valid_camera_rays = nc[:, valid_pixels[:, 0], valid_pixels[:, 1]]   # [3, N]

            cos_seta = torch.sum(max_grad_vec * valid_camera_rays, axis=0)      # [N]
            angle = torch.arccos(cos_seta)              # [N]
            relative_grad_depth = max_grad_depth / ori_points[-1, :]
            filter = ((angle * 180 / 3.1415) > 15) #& (relative_grad_depth < 0.001)     # [N]

            filtered_pixels = valid_pixels[filter]
            filtered_depth = torch.zeros_like(depth)
            filtered_depth[filtered_pixels[..., 0], filtered_pixels[..., 1], :] = depth[filtered_pixels[..., 0], filtered_pixels[..., 1], :]
            mask = (filtered_depth > 0)

            remove_pixels = valid_pixels[~filter]
            save_mask = torch.zeros((h, w))
            save_mask[remove_pixels[..., 0], remove_pixels[..., 1]] = 1
        else:
            save_mask = torch.zeros((depth.shape[0], depth.shape[1]))
            mask = (depth > 0)
        if save_filter:
            save_mask = save_mask.cpu().numpy().squeeze().astype(np.uint8)
            cv2.imwrite(save_filter_path, save_mask * 255)

        return depth, mask