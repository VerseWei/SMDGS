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
import math
from diff_plane_rasterization import GaussianRasterizationSettings as PlaneGaussianRasterizationSettings
from diff_plane_rasterization import GaussianRasterizer as PlaneGaussianRasterizer
from diff_plane_rasterization import DepthMapConverterSettings
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.graphics_utils import normal_from_depth_image

def render_normal(viewpoint_cam, depth, offset=None, normal=None, scale=1):
    # depth: (H, W), bg_color: (3), alpha: (H, W)
    # normal_ref: (3, H, W)
    intrinsic_matrix, extrinsic_matrix = viewpoint_cam.get_intrinsics(), viewpoint_cam.get_extrinsics()
    st = max(int(scale/2)-1,0)
    if offset is not None:
        offset = offset[st::scale,st::scale]

    normal_ref = normal_from_depth_image(depth[st::scale,st::scale], 
                                            intrinsic_matrix, 
                                            extrinsic_matrix, offset)

    normal_ref = normal_ref.permute(2,0,1)
    return normal_ref

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, contrib_densify = False, depth_threshold = None,
           return_plane = True, return_depth_normal = True, nearest_camera = None, use_mask = False, convert_depth = False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device=viewpoint_camera.data_device) + 0
    screenspace_points_abs = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device=viewpoint_camera.data_device) + 0
    try:
        screenspace_points.retain_grad()
        screenspace_points_abs.retain_grad()
    except:
        pass

    if pc.skybox and pc.scene_opac:
        label = torch.ones((pc.get_xyz.shape[0], 1), dtype=torch.float, requires_grad=False, device=pc.data_device)
        label[:pc.skybox_points] = 0
        sky_pixels = 1 - viewpoint_camera.get_gtMask(with_mask=True)
    else:
        label = torch.ones((pc.get_xyz.shape[0], 1), dtype=torch.float, requires_grad=False, device=pc.data_device)
        sky_pixels = torch.zeros_like(viewpoint_camera.get_gtMask(with_mask=True), dtype=torch.float32, device="cuda")

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    means3D = pc.get_xyz
    means2D = screenspace_points
    means2D_abs = screenspace_points_abs
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:   # False
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None

    if override_color is None:  
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features   # True
    else:
        colors_precomp = override_color

    return_dict = None
    if convert_depth:
        pc.config[1] = 1
        convertersettings = DepthMapConverterSettings(viewpoint_camera.world_view_transform, nearest_camera.world_view_transform, viewpoint_camera.get_mono_filter(use_mask), \
                                                      viewpoint_camera.Fx, viewpoint_camera.Fy, nearest_camera.Fx, nearest_camera.Fy, \
                                                      viewpoint_camera.Cx, viewpoint_camera.Cy, nearest_camera.Cx, nearest_camera.Cy, \
                                                      viewpoint_camera.image_height, viewpoint_camera.image_width, \
                                                      nearest_camera.image_height, nearest_camera.image_width, \
                                                      nearest_camera.get_mono_filter(use_mask), True, use_mask, \
                                                      pc.data_device)
    else:
        pc.config[1] = 0
        convertersettings = None

    raster_settings = PlaneGaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            contrib_densify=contrib_densify,
            depth_threshold=depth_threshold,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            render_geo=return_plane,
            debug=pipe.debug,
            config=pc.config,
            device=torch.device(pc.data_device)
        )

    rasterizer = PlaneGaussianRasterizer(raster_settings=raster_settings, converter_settings=convertersettings)

    global_normal = pc.get_normal(viewpoint_camera)
    local_normal = global_normal @ viewpoint_camera.world_view_transform[:3,:3]
    pts_in_cam = means3D @ viewpoint_camera.world_view_transform[:3,:3] + viewpoint_camera.world_view_transform[3,:3]
    depth_z = pts_in_cam[:, 2]
    local_distance = (local_normal * pts_in_cam).sum(-1).abs()
    input_all_map = torch.zeros((means3D.shape[0], 5)).to(pc.data_device).float()
    input_all_map[:, :3] = local_normal
    input_all_map[:, 3] = 1.0                  
    input_all_map[:, 4] = local_distance

    rendered_image, radii, pixels, out_observe, rendered_opac, out_all_map, plane_depth, rendered_scene_opac, \
        projected_depthmap, projected_normalmap, faceNormal_ref = rasterizer(
                means3D = means3D,
                means2D = means2D,
                means2D_abs = means2D_abs,                                                          
                shs = shs,
                colors_precomp = colors_precomp,
                opacities = opacity,
                scales = scales,
                rotations = rotations,
                all_map = input_all_map,
                sky_pixels = sky_pixels,
                cov3D_precomp = cov3D_precomp,
                label = label)

    rendered_normal = out_all_map[0:3]
    rendered_alpha = out_all_map[3:4, ]
    rendered_distance = out_all_map[4:5, ]

    radii = torch.where(torch.isnan(radii) | torch.isinf(radii), torch.tensor(0.0, device=radii.device), radii)
    return_dict =  {"render": rendered_image,
                    "viewspace_points": screenspace_points,
                    "viewspace_points_abs": screenspace_points_abs,
                    "visibility_filter" : radii > 0,
                    "radii": radii,
                    "pixels": pixels,
                    "out_observe": out_observe,
                    "opac": rendered_opac,
                    "rendered_normal": rendered_normal,
                    "plane_depth": plane_depth,
                    "rendered_distance": rendered_distance,
                    "opac_s": rendered_scene_opac,
                    "projected_depthmap": projected_depthmap,
                    "projected_normalmap": projected_normalmap,
                    "depth_z": depth_z,
                    "face_normal": faceNormal_ref
                    }
    
    if return_depth_normal:
        rn = render_normal(viewpoint_camera, plane_depth.squeeze())
        depth_normal = rn * (rendered_alpha).detach()
        return_dict.update({"depth_normal": depth_normal})
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return return_dict