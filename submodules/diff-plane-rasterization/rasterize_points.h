/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <string>
#include <cuda_runtime.h>        // Include the CUDA runtime header for cudaSetDevice()
	
std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& all_map,
	const torch::Tensor& sky_pixels,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
    const int image_height,
    const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool render_geo,
	const bool debug,
	const torch::Tensor& config,
	const torch::Tensor& label);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansBackwardwithConvertCUDA(
 	const torch::Tensor& background,
	const torch::Tensor& all_map_pixels,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
    const torch::Tensor& colors,
	const torch::Tensor& all_maps,
	const torch::Tensor& sky_pixels,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
    const torch::Tensor& dL_dout_color,
	const torch::Tensor& dL_dout_opac,
	torch::Tensor& dL_dout_all_map,
	torch::Tensor& dL_dout_plane_depth,
	const torch::Tensor& dL_dout_opac_s,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer_rt,
	const int R,
	const torch::Tensor& binningBuffer_rt,
	const torch::Tensor& imageBuffer_rt,
	const bool render_geo,
	const bool debug,
	const torch::Tensor& config,
	const torch::Tensor& label,
	const torch::Tensor& dL_dout_proj_depth,
	const torch::Tensor& dL_dout_proj_normal,
	const torch::Tensor& viewmatrix_cv,
	const torch::Tensor& transmatrix_cv,
	const bool debug_cv,
	const float ref_fx,
	const float ref_fy,
	const float src_fx,
	const float src_fy,
	const float ref_cx,
	const float ref_cy,
	const float src_cx,
	const float src_cy,
	const int ref_height,
	const int ref_width,
	const int src_height,
	const int src_width,
	const int num_projected,
	const int MaxNumVertices,
	const int MaxNumFaces,
	const torch::Tensor& vertices,
	const torch::Tensor& rays,
	const torch::Tensor& faces,
	const torch::Tensor& pixel_label,
	const torch::Tensor& faceNormal,
	const torch::Tensor& geomBuffer_cv,
	const torch::Tensor& binningBuffer_cv,
	const torch::Tensor& imgBuffer_cv);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansBackwardCUDA(
 	const torch::Tensor& background,
	const torch::Tensor& all_map_pixels,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
    const torch::Tensor& colors,
	const torch::Tensor& all_maps,
	const torch::Tensor& sky_pixels,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
    const torch::Tensor& dL_dout_color,
	const torch::Tensor& dL_dout_opac,
	const torch::Tensor& dL_dout_all_map,
	const torch::Tensor& dL_dout_plane_depth,
	const torch::Tensor& dL_dout_opac_s,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer_cv,
	const int R,
	const torch::Tensor& binningBuffer_cv,
	const torch::Tensor& imageBuffer_cv,
	const bool render_geo,
	const bool debug,
	const torch::Tensor& config,
	const torch::Tensor& label);

std::tuple<int, int, int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
ConvertDepthCUDA(
    const torch::Tensor& DepthMap,
    torch::Tensor& ref_Mask,
    torch::Tensor& src_Mask,
    const torch::Tensor& viewmatrix,
    const torch::Tensor& transmatrix,
    const bool debug,
	const float ref_fx, 
	const float ref_fy,
    const float src_fx, 
	const float src_fy,
    const float ref_cx,
    const float ref_cy,
	const float src_cx,
    const float src_cy,
    const int ref_height,
    const int ref_width,
    const int src_height,
    const int src_width,
	const bool use_mask);
		
torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix);

torch::Tensor patch_offsets(int h_patch_size, torch::Device device);