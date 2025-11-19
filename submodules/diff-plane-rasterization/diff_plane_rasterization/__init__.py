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

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_gaussians(
    means3D,
    means2D,
    means2D_abs,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    all_map,
    sky_pixels,
    raster_settings,
    converter_settings,
    label,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        means2D_abs,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        all_map,
        sky_pixels,
        raster_settings,
        converter_settings,
        label,
    )

class _RasterizeGaussians(torch.autograd.Function): 
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,            
        means2D_abs,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        all_maps,
        sky_pixels,
        raster_settings,
        converter_settings,
        label,
    ):
        # Restructure arguments the way that the C++ lib expects them
        args_raster = (
            raster_settings.bg, 
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            all_maps,
            sky_pixels,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.render_geo,
            raster_settings.debug,
            raster_settings.config,
            label
        )

        convert_depth = (raster_settings.config[1] > 0)

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_rast_args = cpu_deep_copy_tuple(args_raster) # Copy them before they can be corrupted
            try:
                num_rendered, color, radii, pixels, out_observe, opac, out_all_map, out_plane_depth, opac_s, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args_raster)
            except Exception as ex:
                torch.save(cpu_rast_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, color, radii, pixels, out_observe, opac, out_all_map, out_plane_depth, opac_s, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args_raster)

        if convert_depth:
            args_converter = (
                out_plane_depth,
                converter_settings.ref_Mask,
                converter_settings.src_Mask,
                converter_settings.viewmatrix,
                converter_settings.transmatrix,
                converter_settings.debug,
                converter_settings.ref_fx,
                converter_settings.ref_fy,
                converter_settings.src_fx,
                converter_settings.src_fy,
                converter_settings.ref_cx,
                converter_settings.ref_cy,
                converter_settings.src_cx,
                converter_settings.src_cy,
                converter_settings.ref_height,
                converter_settings.ref_width,
                converter_settings.src_height,
                converter_settings.src_width,
                converter_settings.use_mask
            )
            if raster_settings.debug:
                cpu_convert_args = cpu_deep_copy_tuple(args_converter)
                try:
                    num_projected, MaxNumVertices, MaxNumFaces, projected_depthmap, projected_normalmap, vertices, rays, faces, pixel_label,faceNormal_ref, faceNormal, geomBuffer_cv, binningBuffer_cv, imgBuffer_cv = _C.depth_converter(*args_converter)
                except Exception as ex:
                    torch.save(cpu_convert_args, "snapshot_cv.dump")
                    print("\nAn error occured in convert. Please forward snapshot_cv.dump for debugging.")
                    raise ex
            else:
                num_projected, MaxNumVertices, MaxNumFaces, projected_depthmap, projected_normalmap, vertices, rays, faces, pixel_label, faceNormal_ref, faceNormal, geomBuffer_cv, binningBuffer_cv, imgBuffer_cv = _C.depth_converter(*args_converter)
            
        else:
            projected_depthmap = None
            projected_normalmap = None
            faceNormal_ref = None
            faceNormal = None

        # Keep relevant tensors for backward
        if convert_depth:
            ctx.convert_depth = convert_depth
            ctx.raster_settings = raster_settings
            ctx.num_rendered = num_rendered
            ctx.converter_settings = converter_settings
            ctx.num_projected =  num_projected
            ctx.MaxNumVertices = MaxNumVertices
            ctx.MaxNumFaces = MaxNumFaces
            ctx.save_for_backward(out_all_map, colors_precomp, all_maps, sky_pixels, means3D, scales, rotations, cov3Ds_precomp, radii, sh, label, geomBuffer, binningBuffer, imgBuffer, \
                                  vertices, rays, faces, pixel_label, faceNormal, geomBuffer_cv, binningBuffer_cv, imgBuffer_cv)
        else:
            ctx.convert_depth = convert_depth
            ctx.raster_settings = raster_settings
            ctx.num_rendered = num_rendered
            ctx.save_for_backward(out_all_map, colors_precomp, all_maps, sky_pixels, means3D, scales, rotations, cov3Ds_precomp, radii, sh, label, geomBuffer, binningBuffer, imgBuffer)
        return color, radii, pixels, out_observe, opac, out_all_map, out_plane_depth, opac_s, projected_depthmap, projected_normalmap, faceNormal_ref

    @staticmethod
    def backward(ctx, grad_out_color, grad_radii, grad_out_pixels, grad_out_observe, grad_out_opac, grad_out_all_map, grad_out_plane_depth, grad_out_opac_s, \
                 grad_out_proj_depth, grad_out_proj_normal, grad_out_facenormal_ref):  
                                                        
        # Restore necessary values from context
        
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        convert_depth = ctx.convert_depth
        if convert_depth:
            converter_settings = ctx.converter_settings
            num_projected = ctx.num_projected
            MaxNumVertices = ctx.MaxNumVertices
            MaxNumFaces = ctx.MaxNumFaces
            all_map_pixels, colors_precomp, all_maps, sky_pixels, means3D, scales, rotations, cov3Ds_precomp, radii, sh, label, geomBuffer, binningBuffer, imgBuffer, \
                                vertices, rays, faces, pixel_label, faceNormal, geomBuffer_cv, binningBuffer_cv, imgBuffer_cv = ctx.saved_tensors
        else:
            all_map_pixels, colors_precomp, all_maps, sky_pixels, means3D, scales, rotations, cov3Ds_precomp, radii, sh, label, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        # Restructure args as C++ method expects them
        if convert_depth:
            args = (raster_settings.bg,
                    all_map_pixels,
                    means3D, 
                    radii, 
                    colors_precomp, 
                    all_maps,
                    sky_pixels,
                    scales, 
                    rotations, 
                    raster_settings.scale_modifier, 
                    cov3Ds_precomp, 
                    raster_settings.viewmatrix, 
                    raster_settings.projmatrix, 
                    raster_settings.tanfovx, 
                    raster_settings.tanfovy,
                    grad_out_color, 
                    grad_out_opac,
                    grad_out_all_map,
                    grad_out_plane_depth,
                    grad_out_opac_s,
                    sh, 
                    raster_settings.sh_degree, 
                    raster_settings.campos,
                    geomBuffer,
                    num_rendered,
                    binningBuffer,
                    imgBuffer,
                    raster_settings.render_geo,
                    raster_settings.debug,
                    raster_settings.config,
                    label,
                    grad_out_proj_depth,
                    grad_out_proj_normal,
                    converter_settings.viewmatrix,
                    converter_settings.transmatrix,
                    converter_settings.debug,
                    converter_settings.ref_fx,
                    converter_settings.ref_fy,
                    converter_settings.src_fx,
                    converter_settings.src_fy,
                    converter_settings.ref_cx,
                    converter_settings.ref_cy,
                    converter_settings.src_cx,
                    converter_settings.src_cy,
                    converter_settings.ref_height,
                    converter_settings.ref_width,
                    converter_settings.src_height,
                    converter_settings.src_width,
                    num_projected,
                    MaxNumVertices,
                    MaxNumFaces,
                    rays,
                    vertices,
                    faces,
                    pixel_label,
                    faceNormal,
                    geomBuffer_cv,
                    binningBuffer_cv,
                    imgBuffer_cv)
            # Compute gradients for relevant tensors by invoking backward method
            if raster_settings.debug:
                cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
                try:
                    grad_means2D, grad_means2D_abs, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations, \
                                                                                    depth, gard_all_map = _C.rasterize_gaussians_backward_with_convert(*args)
                except Exception as ex:
                    torch.save(cpu_args, "snapshot_bw.dump")
                    print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                    raise ex
            else:
                grad_means2D, grad_means2D_abs, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations, \
                                                                                    depth, gard_all_map = _C.rasterize_gaussians_backward_with_convert(*args)
        else:
            args = (raster_settings.bg,
                    all_map_pixels,
                    means3D, 
                    radii, 
                    colors_precomp, 
                    all_maps,
                    sky_pixels,
                    scales, 
                    rotations, 
                    raster_settings.scale_modifier, 
                    cov3Ds_precomp, 
                    raster_settings.viewmatrix, 
                    raster_settings.projmatrix, 
                    raster_settings.tanfovx, 
                    raster_settings.tanfovy,
                    grad_out_color, 
                    grad_out_opac,
                    grad_out_all_map,
                    grad_out_plane_depth,
                    grad_out_opac_s,
                    sh, 
                    raster_settings.sh_degree, 
                    raster_settings.campos,
                    geomBuffer,
                    num_rendered,
                    binningBuffer,
                    imgBuffer,
                    raster_settings.render_geo,
                    raster_settings.debug,
                    raster_settings.config,
                    label)
            # Compute gradients for relevant tensors by invoking backward method
            if raster_settings.debug:
                cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
                try:
                    grad_means2D, grad_means2D_abs, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations, \
                                                                                    depth, gard_all_map = _C.rasterize_gaussians_backward(*args)
                except Exception as ex:
                    torch.save(cpu_args, "snapshot_bw.dump")
                    print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                    raise ex
            else:
                grad_means2D, grad_means2D_abs, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations, \
                                                                                    depth, gard_all_map = _C.rasterize_gaussians_backward(*args)

        scaled_grad_means2D = grad_means2D
        scaled_grad_means2D_abs = grad_means2D_abs

        grads = (
            grad_means3D,
            scaled_grad_means2D,               
            scaled_grad_means2D_abs,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            gard_all_map,
            None,
            None,
            None,
            None
        )

        return grads

class GaussianRasterizationSettings():
    image_height: int
    image_width: int
    tanfovx: float
    tanfovy: float
    bg: torch.Tensor
    scale_modifier: float
    contrib_densify: bool
    depth_threshold: float
    viewmatrix: torch.Tensor
    projmatrix: torch.Tensor
    sh_degree: int
    campos: torch.Tensor
    prefiltered: bool
    render_geo: bool
    debug: bool
    config: torch.Tensor

    def __init__(self, image_height: int, image_width: int, tanfovx: float, tanfovy: float, bg: torch.Tensor, scale_modifier: float,
                 contrib_densify: bool, depth_threshold: float, viewmatrix: torch.Tensor, projmatrix: torch.Tensor, sh_degree: int,
                 campos: torch.Tensor, prefiltered: bool, render_geo: bool, debug: bool, config: torch.Tensor, device=torch.device('cuda:1')):
        self.device = device
        self.image_height = image_height
        self.image_width = image_width
        self.tanfovx = tanfovx
        self.tanfovy = tanfovy
        self.bg = bg.to(self.device)      
        self.scale_modifier = scale_modifier
        self.contrib_densify = contrib_densify
        self.depth_threshold = depth_threshold
        self.viewmatrix = viewmatrix.to(self.device)  
        self.projmatrix = projmatrix.to(self.device)  
        self.sh_degree = sh_degree
        self.campos = campos.to(self.device)      
        self.prefiltered = prefiltered
        self.render_geo = render_geo
        self.debug = debug
        self.config = config.to(self.device)

class DepthMapConverterSettings():
    viewmatrix: torch.Tensor
    transmatrix: torch.Tensor
    ref_Mask: torch.Tensor
    ref_fx: float
    ref_fy: float
    src_fx: float
    src_fy: float
    ref_cx: float
    ref_cy: float
    src_cx: float
    src_cy: float
    ref_height: int
    ref_width: int
    src_height: int
    src_width: int
    src_Mask: torch.Tensor
    debug: bool
    use_mask: bool

    def __init__(self, viewmatrix: torch.Tensor, transmatrix: torch.Tensor, ref_Mask: torch.Tensor,
                 ref_fx: float, ref_fy: float, src_fx: float, src_fy:float, ref_cx: float, ref_cy: float, src_cx: float, src_cy: float,
                 ref_height: int, ref_width: int, src_height: int, src_width: int, src_Mask: torch.Tensor, debug: bool, use_mask: bool,
                 device=torch.device('cuda:1')):
        self.device = device
        self.viewmatrix = viewmatrix.to(self.device)
        self.transmatrix = transmatrix.to(self.device)
        self.ref_Mask = ref_Mask.to(self.device)
        if self.ref_Mask.dtype not in [torch.int32, torch.int64]:  # 支持 int32 和 int64
            self.ref_Mask = self.ref_Mask.to(torch.int)
        self.ref_fx = ref_fx
        self.ref_fy = ref_fy
        self.src_fx = src_fx
        self.src_fy = src_fy
        self.ref_cx = ref_cx
        self.ref_cy = ref_cy
        self.src_cx = src_cx
        self.src_cy = src_cy
        self.ref_height = ref_height
        self.ref_width = ref_width
        self.src_height = src_height
        self.src_width = src_width
        self.src_Mask = src_Mask.to(self.device)
        if self.src_Mask.dtype not in [torch.int32, torch.int64]:  # 支持 int32 和 int64
            self.src_Mask = self.src_Mask.to(torch.int)
        self.debug = debug
        self.use_mask = use_mask

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings, converter_settings=None):
        super().__init__()
        self.raster_settings = raster_settings
        self.converter_settings = converter_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(self, means3D, means2D, means2D_abs, opacities, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None, all_map=None, sky_pixels=None, label=None):
        
        raster_settings = self.raster_settings
        converter_settings = self.converter_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])
        if all_map is None:
            all_map = torch.Tensor([])
        if sky_pixels is None:
            sky_pixels = torch.Tensor([])
        if label is None:
            label = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            means2D_abs,
            shs,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            all_map,
            sky_pixels,
            raster_settings,
            converter_settings,
            label,
        )

