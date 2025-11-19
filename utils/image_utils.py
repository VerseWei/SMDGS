#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
import numpy as np
import math
from utils.graphics_utils import fov2focal

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def dilate(bin_img, ksize=5):
    pad = (ksize - 1) // 2
    bin_img = F.pad(bin_img, pad=[pad, pad, pad, pad], mode='reflect')
    out = F.max_pool2d(bin_img, kernel_size=ksize, stride=1, padding=0)
    return out

def erode(bin_img, ksize=5):
    out = 1 - dilate(1 - bin_img, ksize)
    return out

def normal2curv(normal, mask):

    n = normal.permute([1, 2, 0])   # [H, W, 3]
    m = mask.permute([1, 2, 0])
    n = torch.nn.functional.pad(n[None], [0, 0, 1, 1, 1, 1], mode='replicate')  # [None]等效于n.unsqueeze(0)
                                                                                # [0, 0]：表示在第3维（最后一维，也就是法线向量的维度，通常为3）上没有填充
                                                                                # [1, 1]：表示在第2维（宽度 W，也就是 192）上，前面和后面各填充1个像素
                                                                                # [1, 1]：表示在第1维（高度 H，也就是 144）上，前面和后面各填充1个像素
    m = torch.nn.functional.pad(m[None].to(torch.float32), [0, 0, 1, 1, 1, 1], mode='replicate').to(torch.bool)
    n_c = (n[:, 1:-1, 1:-1, :]      ) * m[:, 1:-1, 1:-1, :]
    n_u = (n[:,  :-2, 1:-1, :] - n_c) * m[:,  :-2, 1:-1, :]
    n_l = (n[:, 1:-1,  :-2, :] - n_c) * m[:, 1:-1,  :-2, :]
    n_b = (n[:, 2:  , 1:-1, :] - n_c) * m[:, 2:  , 1:-1, :]
    n_r = (n[:, 1:-1, 2:  , :] - n_c) * m[:, 1:-1, 2:  , :]
    curv = (n_u + n_l + n_b + n_r)[0]
    curv = curv.permute([2, 0, 1]) * mask
    curv = curv.norm(1, 0, True)
    
    return curv

def depth2normal(depth, mask, camera):
    # conver to camera position
    camD = depth.permute([1, 2, 0])
    mask = mask.permute([1, 2, 0])
    shape = camD.shape
    device = camD.device
    h, w, _ = torch.meshgrid(torch.arange(0, shape[0]), torch.arange(0, shape[1]), torch.arange(0, shape[2]), indexing='ij')
    # print(h)
    h = h.to(torch.float32).to(device)
    w = w.to(torch.float32).to(device)
    p = torch.cat([w, h], axis=-1)
    
    p[..., 0:1] -= camera.prcppoint[0] * camera.image_width
    p[..., 1:2] -= camera.prcppoint[1] * camera.image_height
    p *= camD
    K00 = fov2focal(camera.FoVy, camera.image_height)
    K11 = fov2focal(camera.FoVx, camera.image_width)
    K = torch.tensor([K00, 0, 0, K11]).reshape([2,2])
    Kinv = torch.inverse(K).to(device)
    # print(p.shape, Kinv.shape)
    p = p @ Kinv.t()
    camPos = torch.cat([p, camD], -1)

    # padded = mod.contour_padding(camPos.contiguous(), mask.contiguous(), torch.zeros_like(camPos), filter_size // 2)
    # camPos = camPos + padded
    p = torch.nn.functional.pad(camPos[None], [0, 0, 1, 1, 1, 1], mode='replicate')
    mask = torch.nn.functional.pad(mask[None].to(torch.float32), [0, 0, 1, 1, 1, 1], mode='replicate').to(torch.bool)
    
    p_c = (p[:, 1:-1, 1:-1, :]      ) * mask[:, 1:-1, 1:-1, :]
    p_u = (p[:,  :-2, 1:-1, :] - p_c) * mask[:,  :-2, 1:-1, :]
    p_l = (p[:, 1:-1,  :-2, :] - p_c) * mask[:, 1:-1,  :-2, :]
    p_b = (p[:, 2:  , 1:-1, :] - p_c) * mask[:, 2:  , 1:-1, :]
    p_r = (p[:, 1:-1, 2:  , :] - p_c) * mask[:, 1:-1, 2:  , :]

    n_ul = torch.cross(p_u, p_l)
    n_ur = torch.cross(p_r, p_u)
    n_br = torch.cross(p_b, p_r)
    n_bl = torch.cross(p_l, p_b)

    n = n_ul + n_ur + n_br + n_bl
    n = n[0]

    mask = mask[0, 1:-1, 1:-1, :]

    n = torch.nn.functional.normalize(n, dim=-1)

    n = (n * mask).permute([2, 0, 1])
    return n

def resize_image(img, factor, mode='bilinear'):
    # print(type(img))
    if factor == 1:
        return img
    is_np = type(img) == np.ndarray
    if is_np:
        resize = torch.from_numpy(img)
    else:
        resize = img.clone()
    dtype = resize.dtype
    if type(factor) == int:
        resize = torch.nn.functional.interpolate(resize[None].to(torch.float32), scale_factor=1/factor, mode=mode)[0].to(dtype)
    elif len(factor) == 2:
        resize = torch.nn.functional.interpolate(resize[None].to(torch.float32), size=factor, mode=mode)[0].to(dtype)
    # else:

    if is_np:
        resize = resize.numpy()
    # print(type(img))
    return resize

def world2scrn(xyz, cams, pad):
    device = xyz.device
    mask = [i.get_gtMask(True).to(device).to(torch.float32) for i in cams]     # 获取相机的mask，没有则全为1

    if pad >= 0:        # pcd=4
        pool = torch.nn.MaxPool2d(9, stride=1, padding=pad)
    else:
        pool = torch.nn.MinPool2d(9, stride=1, padding=-pad)
    mask = pool(torch.cat(mask, 0))   # 对拼接后的mask进行池化操作

    worldPos = xyz#.detach()
    worldPos = torch.cat([worldPos, torch.ones_like(worldPos[:, :1])], 1)[None, :, None]    # 世界系下点坐标，后面补1
    
    view_mat = torch.cat([i.world_view_transform[None] for i in cams], 0).to(device)[:, None]  # 拼接w2c矩阵
    z_near = torch.cat([torch.tensor([[i.znear]]) for i in cams], 0).to(device)
    z_far = torch.cat([torch.tensor([[i.zfar]]) for i in cams], 0).to(device)

    camPos = (worldPos @ view_mat[..., :3]).squeeze()       # 拼接的点在相机系坐标
    outViewZ = torch.le(camPos[..., 2], z_near) + torch.gt(camPos[..., 2], z_far)

    prj_mat = torch.cat([i.full_proj_transform[None] for i in cams], 0).to(device)[:, None]   # 拼接投影矩阵
    projPos = (worldPos @ prj_mat).squeeze()                    
    projPos = projPos[..., :3] / (projPos[..., 3:] + 1e-7)               # 投影点坐标

    outViewX = torch.le(projPos[..., 0], -1) + torch.gt(projPos[..., 0], 1)   # 投影点坐标与-1和比较判断点是否在屏幕上
    outViewY = torch.le(projPos[..., 1], -1) + torch.gt(projPos[..., 1], 1)
    outView = outViewX + outViewY# + outViewZ
    # outAllView = torch.all(outView, dim=0)

    reso = torch.cat([torch.tensor([[[i.image_width, i.image_height]]]) for i in cams], 0).to(device)
    prcp = torch.cat([i.prcppoint[None] for i in cams], 0).to(device)[:, None]

    scrnPos = ((projPos[..., :2] + 1) * reso - 1) * 0.5 + reso * (prcp - 0.5)
    ndc = (scrnPos / reso) * 2 - 1

    scrnPos = torch.clip(scrnPos, torch.zeros_like(reso), reso - 1).to(torch.long)   # 点的屏幕空间坐标

    mask_idx = torch.arange(0, len(mask))[:, None].to(torch.long)

    if mask.mean() == 1:
        inMask = torch.ones_like(outView).to(torch.bool)
    else:
        inMask = mask.permute([0, 2, 1])[mask_idx, scrnPos[..., 0], scrnPos[..., 1]].to(torch.bool)
    # inMaskOrOutView = torch.all(inMask + outView, dim=0)
    # inMaskOrOutView = torch.all(inMask, dim=0)

    # visible = inMaskOrOutView * ~outAllView

    return camPos, ndc, inMask, outView

def project_points_3d_to_2d(points_3d, camera_intrinsics, camera_extrinsics):
    # 获取旋转矩阵和平移向量
    rotation_matrix = camera_extrinsics[:3, :3]  # 旋转矩阵 3x3
    translation_vector = camera_extrinsics[:3, 3]  # 平移向量 3x1

    # 将3D点从世界坐标系转换到相机坐标系
    points_camera = torch.matmul(points_3d, rotation_matrix.t()) + translation_vector.t()

    # 通过相机内参矩阵将相机坐标系的3D点投影到2D平面
    x_img = (camera_intrinsics[0, 0] * points_camera[:, 0] / points_camera[:, 2]) + camera_intrinsics[0, 2]
    y_img = (camera_intrinsics[1, 1] * points_camera[:, 1] / points_camera[:, 2]) + camera_intrinsics[1, 2]

    # 返回投影的2D坐标
    projected_points = torch.stack((x_img, y_img), dim=-1)
    return projected_points

def is_points_visible_in_mask(projected_points, mask, height, width):

    # 确保坐标在图像范围内
    x_img = projected_points[:, 0]
    y_img = projected_points[:, 1]

    # 判断投影点是否在图像范围内
    in_image = (x_img >= 0) & (x_img < width) & (y_img >= 0) & (y_img < height)

    # 将坐标四舍五入为整数，并确保在图像范围内
    x_img_int = x_img.long()
    y_img_int = y_img.long()

    # 获取mask中对应位置的值
    is_in_img = (x_img_int >= 0) & (x_img_int < width) & (y_img_int >= 0) & (y_img_int < height)        # [N]
    d_mask = torch.zeros((projected_points.shape[0])).to(torch.bool).to(projected_points.device)
    d_mask[is_in_img] = (mask[:, y_img_int[is_in_img], x_img_int[is_in_img]] > 0)

    # 判断该点是否在黑色区域 (mask中值为0表示黑色区域)
    invisible = in_image & ~d_mask

    return invisible

def project_and_check_visibility(points_3d, camera_intrinsics, camera_extrinsics, height, width, mask):
    # 通过相机的内外参矩阵投影3D点到2D图像平面
    projected_points = project_points_3d_to_2d(points_3d, camera_intrinsics, camera_extrinsics)

    # 判断投影后的2D点是否在白色区域内可见
    invisible = is_points_visible_in_mask(projected_points, mask, height, width)

    return invisible
