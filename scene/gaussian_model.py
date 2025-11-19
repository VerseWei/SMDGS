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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation, build_scaling
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH, SH2RGB
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.image_utils import world2scrn, project_and_check_visibility
from pytorch3d.transforms import quaternion_to_matrix

def storePly(path, xyz, rgb, normals = None):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    if normals is None:
        normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def dilate(bin_img, ksize=5):
    pad = (ksize - 1) // 2
    bin_img = torch.nn.functional.pad(bin_img, pad=[pad, pad, pad, pad], mode='reflect')
    out = torch.nn.functional.max_pool2d(bin_img, kernel_size=ksize, stride=1, padding=0)
    return out

def erode(bin_img, ksize=5):
    out = 1 - dilate(1 - bin_img, ksize)
    return out

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, args):
        self.active_sh_degree = 0
        self.max_sh_degree = args.sh_degree  
        self._xyz = torch.empty(0)
        self._knn_f = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.max_weight = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.knn_dists = None
        self.knn_idx = None
        self.data_device = args.data_device
        self.skybox = args.skybox
        self.skybox_points = args.skybox_points
        self.scene_opac = args.scene_opac
        self.use_mask = args.use_mask
        self.mask_type = args.mask_type
        self.model_path = args.model_path
        self.contrib_densify = args.contrib_densify
        self.patch_size = args.patch_size
        self.sample_num = args.sample_num
        try:
            self.config = [args.scene_opac, args.depth_converter]
        except AttributeError:
            self.config = [False, False]
        self.setup_functions()
        self.use_app = False

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._knn_f,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.max_weight,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._knn_f,
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        self.max_weight,
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale,
        ) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
        
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_smallest_axis(self, return_idx=False):
        rotation_matrices = self.get_rotation_matrix()
        smallest_axis_idx = self.get_scaling.min(dim=-1)[1][..., None, None].expand(-1, 3, -1)
        smallest_axis = rotation_matrices.gather(2, smallest_axis_idx)
        if return_idx:
            return smallest_axis.squeeze(dim=2), smallest_axis_idx[..., 0, 0]
        return smallest_axis.squeeze(dim=2)
    
    def get_normal(self, view_cam=None):
        normal_global = self.get_smallest_axis()
        if view_cam is not None:
            gaussian_to_cam_global = view_cam.camera_center - self._xyz
            neg_mask = (normal_global * gaussian_to_cam_global).sum(-1) < 0.0
            normal_global[neg_mask] = -normal_global[neg_mask]
        return normal_global
    
    def get_rotation_matrix(self):
        return quaternion_to_matrix(self.get_rotation)

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float, spatial_center: float):
        self.spatial_lr_scale = spatial_lr_scale
        self.spatial_center = torch.from_numpy(spatial_center).to(self.data_device)

        if self.skybox:
            xyz = torch.tensor(np.asarray(pcd.points)).float().to(self.data_device)
            minimum, _ = torch.min(xyz, axis=0)
            maximum, _ = torch.max(xyz, axis=0)
            radius = max(torch.linalg.norm(maximum - self.spatial_center), torch.linalg.norm(minimum - self.spatial_center))
            self.skybox_radius = radius * 10
            theta = (2.0 * torch.pi * torch.rand(self.skybox_points, device=self.data_device)).float()
            phi = (torch.arccos(1.0 - 2 * torch.rand(self.skybox_points, device=self.data_device))).float()

            # Initialize skybox gaussians
            skybox_normals = torch.zeros((self.skybox_points, 3)).float().to(self.data_device)
            skybox_normals[:, 0] = -torch.cos(theta) * torch.sin(phi)
            skybox_normals[:, 1] = -torch.sin(theta) * torch.sin(phi)
            skybox_normals[:, 2] = -torch.cos(phi)

            skybox_xyz = self.skybox_radius * (-skybox_normals)
            skybox_xyz += self.spatial_center

            skybox_colors = torch.ones((self.skybox_points, 3), device=self.data_device)
            skybox_colors[:, 0] *= 0.7
            skybox_colors[:, 1] *= 0.8
            skybox_colors[:, 2] *= 0.95

            skybox_dist2 = torch.sqrt(torch.clamp_min(distCUDA2(skybox_xyz), 0.0000001) * 3)
            skybox_dist2 = skybox_dist2[..., None].repeat(1, 3)
            skybox_dist2[:, 2] *= 0.01

            skybox_opacities = torch.ones((skybox_xyz.shape[0], 1), dtype=torch.float, device=self.data_device) * 0.7

            skybox_rots = self.init_rotations(skybox_normals).to(self.data_device)

            # Initialize scene gaussians
            scene_xyz = torch.tensor(np.asarray(pcd.points)).float().to(self.data_device)
            scene_colors = torch.tensor(np.asarray(pcd.colors)).float().to(self.data_device)
            scene_dist2 = torch.sqrt(torch.clamp_min(distCUDA2(scene_xyz), 0.0000001))
            scene_dist2 = scene_dist2[..., None].repeat(1, 3)
            scene_opacities = inverse_sigmoid(0.1 * torch.ones((scene_xyz.shape[0], 1), dtype=torch.float, device=self.data_device))

            scene_normals = torch.from_numpy(pcd.normals)

            scene_rots = torch.zeros((scene_normals.shape[0], 4), device=self.data_device)
            scene_rots[:, 0] = 1
  
            # Combine skybox and scene gaussians
            fused_point_cloud = torch.cat((skybox_xyz, scene_xyz))
            fused_color = torch.cat((skybox_colors, scene_colors))
            features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().to(self.data_device)
            features[:, :3, 0 ] = RGB2SH(fused_color)
            features[:, 3:, 1:] = 0.0
            scales = torch.log(torch.cat((skybox_dist2, scene_dist2)))
            rots = torch.cat((skybox_rots, scene_rots))
            opacities = torch.cat((skybox_opacities, scene_opacities))
            print("Number of points at initialisation : ", fused_point_cloud.shape[0])
            
        else:
            # Initialize scene gaussians
            scene_xyz = torch.tensor(np.asarray(pcd.points)).float().to(self.data_device)
            scene_colors = torch.tensor(np.asarray(pcd.colors)).float().to(self.data_device)
            scene_dist2 = torch.sqrt(torch.clamp_min(distCUDA2(scene_xyz), 0.0000001))
            scene_dist2 = scene_dist2[..., None].repeat(1, 3)
            scene_opacities = inverse_sigmoid(0.1 * torch.ones((scene_xyz.shape[0], 1), dtype=torch.float, device=self.data_device))

            scene_normals = torch.from_numpy(pcd.normals)

            scene_rots = torch.zeros((scene_normals.shape[0], 4), device=self.data_device)
            scene_rots[:, 0] = 1      

            fused_point_cloud = scene_xyz
            fused_color = scene_colors

            features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().to(self.data_device)
            features[:, :3, 0 ] = fused_color
            features[:, 3:, 1:] = 0.0
            scales = torch.log(scene_dist2)
            rots = scene_rots
            opacities = scene_opacities
            print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        knn_f = torch.randn((fused_point_cloud.shape[0], 6)).float().to(self.data_device)
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._knn_f = nn.Parameter(knn_f.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=self.data_device)
        self.max_weight = torch.zeros((self.get_xyz.shape[0]), device=self.data_device)


    def rotation_matrix_to_quaternion(self, rot_mats):
        """
        Converts rotation matrices to quaternions.

        Args:
            rot_mats: Tensor of shape [N, 3, 3], batch of rotation matrices.

        Returns:
            quaternions: Tensor of shape [N, 4], batch of quaternions (w, x, y, z).
        """
        N = rot_mats.size(0)
        quaternions = torch.zeros((N, 4), device=self.data_device)

        # Diagonal elements of the rotation matrix
        trace = torch.einsum('bii->b', rot_mats).to(self.data_device)  # Trace of each matrix in batch

        # Compute quaternion components
        cond1 = trace > 0
        cond2 = ~cond1 & (rot_mats[:, 0, 0] >= rot_mats[:, 1, 1]) & (rot_mats[:, 0, 0] >= rot_mats[:, 2, 2])
        cond3 = ~cond1 & ~cond2 & (rot_mats[:, 1, 1] >= rot_mats[:, 2, 2])
        cond4 = ~cond1 & ~cond2 & ~cond3

        s = torch.zeros_like(trace, device=self.data_device)

        # Case 1: trace > 0
        s[cond1] = torch.sqrt(trace[cond1] + 1.0) * 2
        quaternions[cond1, 0] = 0.25 * s[cond1]
        quaternions[cond1, 1] = (rot_mats[cond1, 2, 1] - rot_mats[cond1, 1, 2]) / s[cond1]
        quaternions[cond1, 2] = (rot_mats[cond1, 0, 2] - rot_mats[cond1, 2, 0]) / s[cond1]
        quaternions[cond1, 3] = (rot_mats[cond1, 1, 0] - rot_mats[cond1, 0, 1]) / s[cond1]

        # Case 2: rot_mats[:, 0, 0] is the largest diagonal entry
        s[cond2] = torch.sqrt(1.0 + rot_mats[cond2, 0, 0] - rot_mats[cond2, 1, 1] - rot_mats[cond2, 2, 2]) * 2
        quaternions[cond2, 0] = (rot_mats[cond2, 2, 1] - rot_mats[cond2, 1, 2]) / s[cond2]
        quaternions[cond2, 1] = 0.25 * s[cond2]
        quaternions[cond2, 2] = (rot_mats[cond2, 0, 1] + rot_mats[cond2, 1, 0]) / s[cond2]
        quaternions[cond2, 3] = (rot_mats[cond2, 0, 2] + rot_mats[cond2, 2, 0]) / s[cond2]

        # Case 3: rot_mats[:, 1, 1] is the largest diagonal entry
        s[cond3] = torch.sqrt(1.0 + rot_mats[cond3, 1, 1] - rot_mats[cond3, 0, 0] - rot_mats[cond3, 2, 2]) * 2
        quaternions[cond3, 0] = (rot_mats[cond3, 0, 2] - rot_mats[cond3, 2, 0]) / s[cond3]
        quaternions[cond3, 1] = (rot_mats[cond3, 0, 1] + rot_mats[cond3, 1, 0]) / s[cond3]
        quaternions[cond3, 2] = 0.25 * s[cond3]
        quaternions[cond3, 3] = (rot_mats[cond3, 1, 2] + rot_mats[cond3, 2, 1]) / s[cond3]

        # Case 4: rot_mats[:, 2, 2] is the largest diagonal entry
        s[cond4] = torch.sqrt(1.0 + rot_mats[cond4, 2, 2] - rot_mats[cond4, 0, 0] - rot_mats[cond4, 1, 1]) * 2
        quaternions[cond4, 0] = (rot_mats[cond4, 1, 0] - rot_mats[cond4, 0, 1]) / s[cond4]
        quaternions[cond4, 1] = (rot_mats[cond4, 0, 2] + rot_mats[cond4, 2, 0]) / s[cond4]
        quaternions[cond4, 2] = (rot_mats[cond4, 1, 2] + rot_mats[cond4, 2, 1]) / s[cond4]
        quaternions[cond4, 3] = 0.25 * s[cond4]

        return quaternions


    def init_rotations(self, normals):
        zg = normals
        xg = torch.zeros_like(zg)
        nonzero_0 = (zg[:, 0] != 0)
        nonzero_1 = (zg[:, 0] == 0) & (zg[:, 1] != 0)
        nonzero_2 = (zg[:, 0] == 0) & (zg[:, 1] == 0) & (zg[:, 2] != 0)
        if torch.any(nonzero_0):
            xg[:, 0][nonzero_0] = -zg[:, 2][nonzero_0] / zg[:, 0][nonzero_0]
            xg[:, 1][nonzero_0] = 0
            xg[:, 2][nonzero_0] = 1
        elif torch.any(nonzero_1):
            xg[:, 0][nonzero_1] = 0
            xg[:, 1][nonzero_1] = -zg[:, 2][nonzero_1] / zg[:, 1][nonzero_1]
            xg[:, 2][nonzero_1] = 1
        elif torch.any(nonzero_2):
            xg[:, 0][nonzero_2] = 1
            xg[:, 1][nonzero_2] = 0
            xg[:, 2][nonzero_2] = -zg[:, 0][nonzero_2] / zg[:, 2][nonzero_2]
        xg = xg / torch.norm(xg, dim=1, keepdim=True)
        yg = torch.cross(zg, xg, dim=1)
        rotations = torch.stack((xg, yg, zg), dim=-1).to(self.data_device)  # [N, 3, 3]   局部系到世界系旋转矩阵
        rots = self.rotation_matrix_to_quaternion(rotations)

        return rots

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=self.data_device)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=self.data_device)
        self.max_all_points = training_args.max_all_points
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._knn_f], 'lr': 0.01, "name": "knn_f"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
    
    def clip_grad(self, norm=1.0):
        for group in self.optimizer.param_groups:
            torch.nn.utils.clip_grad_norm_(group["params"][0], norm)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        normals = self.get_normal().detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        if self.skybox:
            opacities_new = torch.cat((self._opacity[:self.skybox_points], inverse_sigmoid(torch.min(self.get_opacity[self.skybox_points:], 
                                                    torch.ones_like(self.get_opacity[self.skybox_points:])*0.01))), 0)
        else:
            opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device=self.data_device).requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device=self.data_device).transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device=self.data_device).transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device=self.data_device).requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device=self.data_device).requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device=self.data_device).requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._knn_f = optimizable_tensors["knn_f"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.max_weight = self.max_weight[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_knn_f, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "knn_f": new_knn_f,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._knn_f = optimizable_tensors["knn_f"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=self.data_device)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=self.data_device)
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=self.data_device)
        self.max_weight = torch.zeros((self.get_xyz.shape[0]), device=self.data_device)

    def densify_and_split(self, grads, grad_threshold, scene_extent, max_radii2D, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device=self.data_device)
        padded_grad[:grads.shape[0]] = grads.squeeze()
        padded_max_radii2D = torch.zeros((n_init_points), device=self.data_device)
        padded_max_radii2D[:max_radii2D.shape[0]] = max_radii2D.squeeze()

        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        if selected_pts_mask.sum() + n_init_points > self.max_all_points:
            limited_num = self.max_all_points - n_init_points
            padded_grad[~selected_pts_mask] = 0
            ratio = limited_num / float(n_init_points)
            threshold = torch.quantile(padded_grad, (1.0-ratio))
            selected_pts_mask = torch.where(padded_grad > threshold, True, False)

        if self.skybox:
            selected_pts_mask[:self.skybox_points] = False

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device=self.data_device)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_knn_f = self._knn_f[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_knn_f, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device=self.data_device, dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        if selected_pts_mask.sum() + n_init_points > self.max_all_points:
            limited_num = self.max_all_points - n_init_points
            grads_tmp = grads.squeeze().clone()
            grads_tmp[~selected_pts_mask] = 0
            ratio = limited_num / float(n_init_points)
            threshold = torch.quantile(grads_tmp, (1.0-ratio))
            selected_pts_mask = torch.where(grads_tmp > threshold, True, False)

        if self.skybox:
            selected_pts_mask[:self.skybox_points] = False

        if selected_pts_mask.sum() > 0:
            # print(f"clone {selected_pts_mask.sum()}")
            new_xyz = self._xyz[selected_pts_mask]

            stds = self.get_scaling[selected_pts_mask]
            means =torch.zeros((stds.size(0), 3),device=self.data_device)
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation(self._rotation[selected_pts_mask])
            new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask]
            
            new_features_dc = self._features_dc[selected_pts_mask]
            new_features_rest = self._features_rest[selected_pts_mask]
            new_opacities = self._opacity[selected_pts_mask]
            new_scaling = self._scaling[selected_pts_mask]
            new_rotation = self._rotation[selected_pts_mask]
            new_knn_f = self._knn_f[selected_pts_mask]

            self.densification_postfix(new_xyz, new_knn_f, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        max_radii2D = self.max_radii2D.clone()

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent, max_radii2D)

        prune_mask = (self.get_opacity < min_opacity).squeeze()

        if self.skybox:
            prune_mask[:self.skybox_points] = False

        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
            if self.skybox:
                prune_mask[:self.skybox_points] = False
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, viewspace_point_tensor_abs, update_filter, pixels, depth, depth_threshold):
        
        if self.contrib_densify:
            contrib_per_pixel = torch.log(1 + pixels[::2][update_filter] / (pixels[1::2][update_filter] + 1e-8))
            contrib_per_pixel = ((contrib_per_pixel - contrib_per_pixel.min()) / (contrib_per_pixel.max() - contrib_per_pixel.min())).squeeze()
            _, topk_contrib_idx = torch.topk(contrib_per_pixel, int(0.3 * contrib_per_pixel.numel()))
            contrib_filter = torch.zeros_like(contrib_per_pixel, dtype=torch.bool)[:, None]
            contrib_filter[topk_contrib_idx] = 1

            scaling_factor = torch.minimum(2 * torch.ones_like(depth), (depth / depth_threshold) ** 2)
            scaling_factor = scaling_factor.unsqueeze(-1)
            scaling_factor_expanded = scaling_factor.expand_as(pixels[1::2])

            self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor_abs.grad[update_filter,:2], dim=-1, keepdim=True) \
                                                * pixels[1::2][update_filter] \
                                                * scaling_factor_expanded[update_filter] \
                                                * contrib_filter
                                                
            self.denom[update_filter] += pixels[1::2][update_filter] \
                                    * scaling_factor_expanded[update_filter] \
                                    * contrib_filter
        else:
            self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
            self.denom[update_filter] += 1

    def mask_prune(self, cams, pad=4):
        notinMask_sum = torch.zeros((self._xyz.shape[0])).to(self.data_device)
        for cam in cams:
            xyz = self._xyz.detach()
            camera_intrinsic = cam.get_intrinsics()
            camera_extrinsic = cam.get_extrinsics()
            height = cam.image_height
            width = cam.image_width
            mask = cam.get_gtMask(True)
            notinMask = project_and_check_visibility(xyz, camera_intrinsic, camera_extrinsic, height, width, mask)
            notinMask_sum += notinMask
        invisible = (notinMask_sum > 0)
        invisible[:self.skybox_points] = False
        self.prune_points(invisible)


    def get_points_depth_in_depth_map(self, fov_camera, depth, points_in_camera_space, scale=1):
        st = max(int(scale/2)-1,0)
        depth_view = depth[None,:,st::scale,st::scale]
        W, H = int(fov_camera.image_width/scale), int(fov_camera.image_height/scale)
        depth_view = depth_view[:H, :W]
        pts_projections = torch.stack(
                        [points_in_camera_space[:,0] * fov_camera.Fx / points_in_camera_space[:,2] + fov_camera.Cx,
                         points_in_camera_space[:,1] * fov_camera.Fy / points_in_camera_space[:,2] + fov_camera.Cy], -1).float()/scale
        mask = (pts_projections[:, 0] > 0) & (pts_projections[:, 0] < W) &\
               (pts_projections[:, 1] > 0) & (pts_projections[:, 1] < H) & (points_in_camera_space[:,2] > 0.1)

        pts_projections[..., 0] /= ((W - 1) / 2)
        pts_projections[..., 1] /= ((H - 1) / 2)
        pts_projections -= 1
        pts_projections = pts_projections.view(1, -1, 1, 2)
        map_z = torch.nn.functional.grid_sample(input=depth_view,
                                                grid=pts_projections,
                                                mode='bilinear',
                                                padding_mode='border',
                                                align_corners=True
                                                )[0, :, :, 0]
        return map_z, mask
    
    def get_points_from_depth(self, fov_camera, depth, scale=1):
        st = int(max(int(scale/2)-1,0))
        depth_view = depth.squeeze()[st::scale,st::scale]
        rays_d = fov_camera.get_rays(scale=scale)
        depth_view = depth_view[:rays_d.shape[0], :rays_d.shape[1]]
        pts = (rays_d * depth_view[..., None]).reshape(-1,3)
        R = torch.tensor(fov_camera.R).float().cuda()
        T = torch.tensor(fov_camera.T).float().cuda()
        pts = (pts-T)@R.transpose(-1,-2)
        return pts
    