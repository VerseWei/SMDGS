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
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, fov2focal, depth2normal
import copy
from PIL import Image
from utils.general_utils import PILtoTorch
from utils.image_utils import resize_image
import cv2
import torch.nn.functional as F

def dilate(bin_img, ksize=6):
    pad = (ksize - 1) // 2
    bin_img = F.pad(bin_img, pad=[pad, pad, pad, pad], mode='reflect')
    out = F.max_pool2d(bin_img, kernel_size=ksize, stride=1, padding=0)
    return out

def erode(bin_img, ksize=12):
    out = 1 - dilate(1 - bin_img, ksize)
    return out

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, prcppoint, K,
                 image_width, image_height,
                 image_path, image_name, uid, image, mask, mono, alignedD,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, 
                 preload_img=True, data_device = "cuda",
                 mono_filter=None
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.nearest_names_ncc = []
        self.nearest_names_filter = []
        self.nearest_names_pseudo = []
        self.colmap_id = colmap_id
        self.R = R                 
        self.T = T
        self.prcppoint = torch.tensor(prcppoint).to(torch.float32)
        self.K = K
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.image_path = image_path

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image, self.image_gray, self.mask = None, None, None
        self.preload_img = preload_img
        if self.preload_img:
            image = Image.open(self.image_path)
            resized_image = image.resize((image_width, image_height))
            resized_image_rgb = PILtoTorch(resized_image)
            self.original_image = resized_image_rgb[:3, ...].clamp(0.0, 1.0)
            self.image_gray = (0.299 * self.original_image[0] + 0.587 * self.original_image[1] + 0.114 * self.original_image[2])[None]

            resized_mask = cv2.resize(mask.squeeze(), dsize=(image_width, image_height), interpolation=cv2.INTER_NEAREST)[None]
            resized_mono = None if mono is None else resize_image(mono, [image_height, image_width])
            resized_alignedD = None if alignedD is None else resize_image(alignedD, [image_height, image_width])

            self.mask = None if mask is None else torch.from_numpy(resized_mask)
            self.mono = None if mono is None else torch.from_numpy(resized_mono)
            self.alignedD = None if alignedD is None else torch.from_numpy(resized_alignedD)

            self.consist_mask = None if alignedD is None else torch.zeros_like(self.alignedD)

            if mono_filter is not None:
                resized_mono_filter = cv2.resize(mono_filter.squeeze(), dsize=(image_width, image_height), interpolation=cv2.INTER_NEAREST)[None]
            self.mono_filter = None if mono_filter is None else torch.from_numpy(resized_mono_filter)

        self.image_width = image_width
        self.image_height = image_height
        self.resolution = (image_width, image_height)
        self.Fx = fov2focal(FoVx, self.image_width)
        self.Fy = fov2focal(FoVy, self.image_height)
        self.Cx = 0.5 * self.image_width
        self.Cy = 0.5 * self.image_height

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).to(self.data_device)
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).to(self.data_device)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.plane_mask, self.non_plane_mask = None, None

        if mono is not None:
            monoD = self.mono[3:]
            self.mono_d2n = depth2normal(monoD, self.mask, self.prcppoint, self.image_width, self.image_height, FoVx, FoVy)

    def get_image(self, bg=torch.tensor([0, 0, 0]), with_mask=False, mask_overwrite=None):
        bg = bg.to(self.data_device)
        if self.preload_img:
            if self.mask is None or not with_mask:
                return self.original_image.to(self.data_device), self.image_gray.to(self.data_device)
            mask = self.get_gtMask(with_mask) if mask_overwrite is None else mask_overwrite
            gt_image = self.original_image.to(self.data_device)
            gt_image = gt_image * mask + bg[:, None, None] * (1 - mask)
            gt_image_gray = self.image_gray.to(self.data_device)
            gt_image_gray = gt_image_gray * mask + bg[:1, None, None] * (1 - mask)
            return gt_image, gt_image_gray
        else:
            image = Image.open(self.image_path)
            resized_image = image.resize((self.image_width, self.image_height))
            resized_image_rgb = PILtoTorch(resized_image)
            gt_image = resized_image_rgb[:3, ...].clamp(0.0, 1.0).to(self.data_device)
            resized_image_gray =(0.299 * gt_image[0] + 0.587 * gt_image[1] + 0.114 * gt_image[2])[None]
            gt_image_gray = resized_image_gray.clamp(0.0, 1.0).to(self.data_device)
            mask = self.get_gtMask(with_mask) if mask_overwrite is None else mask_overwrite
            gt_image = gt_image * mask + bg[:, None, None] * (1 - mask)
            gt_image_gray = gt_image_gray * mask + bg[:1, None, None] * (1 - mask)
            return gt_image, gt_image_gray
        
    def get_intrinsics(self, scale=1.0):
        intr = torch.tensor([[self.Fx / scale, 0, self.Cx / scale],
                             [0, self.Fy / scale, self.Cy / scale],
                             [0, 0, 1]], device=self.data_device)
        return intr
    
    def get_extrinsics(self):
        return self.world_view_transform.t()
        
    def get_gtMask(self, with_mask=False):
        if self.mask is None or not with_mask:
            self.mask = torch.ones_like(self.original_image[:1])
        return self.mask.to(self.data_device)

    def get_mono_filter(self, with_mask=False, mask_overwrite=None):
        if self.mono_filter is None:
            return self.mask.to(self.data_device)
        mono_filter = self.mono_filter.to(self.data_device)
        mask = self.get_gtMask(with_mask) if mask_overwrite is None else mask_overwrite
        mono_filter *= mask
        return mono_filter
    
    def get_mono(self, with_mask=False, mask_overwrite=None):
        mono = self.mono.to(self.data_device)
        mask = self.get_gtMask(with_mask) if mask_overwrite is None else mask_overwrite
        mono *= mask
        return mono
    
    def get_alignedD(self, with_mask=False, mask_overwrite=None):
        alignedD = self.alignedD.to(self.data_device)
        mask = self.get_gtMask(with_mask) if mask_overwrite is None else mask_overwrite
        alignedD *= mask
        return alignedD
    
    def get_pmaps(self, with_mask=False, mask_overwrite=None):
        return self.consist_mask.to(self.data_device)

    def get_calib_matrix_nerf(self, scale=1.0):
        intrinsic_matrix = self.get_intrinsics()
        extrinsic_matrix = self.get_extrinsics()
        return intrinsic_matrix, extrinsic_matrix
    
    def get_rays(self, scale=1.0):
        W, H = int(self.image_width/scale), int(self.image_height/scale)
        
        ix, iy = torch.meshgrid(
            torch.arange(W), torch.arange(H), indexing='xy')
        rays_d = torch.stack(
                    [(ix-self.Cx/scale) / self.Fx * scale,
                    (iy-self.Cy/scale) / self.Fy * scale,
                    torch.ones_like(ix)], -1).float().to(self.data_device)
        return rays_d
    
    def get_k(self, scale=1.0):
        K = torch.tensor([[self.Fx / scale, 0, self.Cx / scale],
                        [0, self.Fy / scale, self.Cy / scale],
                        [0, 0, 1]], device=self.data_device)
        return K
    
    def get_inv_k(self, scale=1.0):
        K_T = torch.tensor([[scale/self.Fx, 0, -self.Cx/self.Fx],
                            [0, scale/self.Fy, -self.Cy/self.Fy],
                            [0, 0, 1]], device=self.data_device)
        return K_T
    
