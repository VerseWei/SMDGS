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

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <random>
#include <chrono>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include "cuda_converter/config.h"
#include "cuda_converter/converter.h"
#include "rasterize_points.h"
#include <fstream>
#include <string>
#include <functional>
#include <cuda_runtime.h>        // Include the CUDA runtime header for cudaSetDevice()

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

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
	const torch::Tensor& label)
{
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }
  
  // Determine which device the tensor is on
  auto device_avai = means3D.device();
  int device_index = device_avai.index(); // Get the index of the device
  // Set the current CUDA device to the device where 'points' is located
  cudaSetDevice(device_index);

  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;

  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

  torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
  torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
  torch::Tensor pixels = torch::zeros({2 * P, 1}, means3D.options());
  torch::Tensor out_observe = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
  torch::Tensor out_opac = torch::full({1, H, W}, 0.0, float_opts);
  torch::Tensor out_opac_s = torch::full({1, H, W}, 0.0, float_opts);
  torch::Tensor out_all_map = torch::full({NUM_ALL_MAP, H, W}, 0, float_opts);
  torch::Tensor out_plane_depth = torch::full({1, H, W}, 0, float_opts);
  
  torch::Device device(torch::kCUDA, device_index);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer_rt = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer_rt = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer_rt = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc_rt = resizeFunctional(geomBuffer_rt);
  std::function<char*(size_t)> binningFunc_rt = resizeFunctional(binningBuffer_rt);
  std::function<char*(size_t)> imgFunc_rt = resizeFunctional(imgBuffer_rt);
  
  int rendered = 0;
  if(P != 0)
  {
	  int M = 0;
	  if(sh.size(0) != 0)
	  {
		M = sh.size(1);
      }
	  rendered = CudaRasterizer::Rasterizer::forward(
	    geomFunc_rt,
		binningFunc_rt,
		imgFunc_rt,
	    P, degree, M,
		background.contiguous().data<float>(),
		W, H,
		means3D.contiguous().data<float>(),
		sh.contiguous().data_ptr<float>(),
		colors.contiguous().data<float>(), 
		opacity.contiguous().data<float>(), 
		scales.contiguous().data_ptr<float>(),
		scale_modifier,
		rotations.contiguous().data_ptr<float>(),
		cov3D_precomp.contiguous().data<float>(), 
		all_map.contiguous().data<float>(), 
		sky_pixels.contiguous().data<float>(),
		viewmatrix.contiguous().data<float>(), 
		projmatrix.contiguous().data<float>(),
		campos.contiguous().data<float>(),
		tan_fovx,
		tan_fovy,
		prefiltered,
		config.contiguous().data<float>(),
		out_color.contiguous().data<float>(),
		pixels.contiguous().data<float>(),
		radii.contiguous().data<int>(),
		out_observe.contiguous().data<int>(),
		out_opac.contiguous().data<float>(),
		out_all_map.contiguous().data<float>(),
		out_plane_depth.contiguous().data<float>(),
		out_opac_s.contiguous().data<float>(),
		label.contiguous().data<float>(),
		render_geo,
		debug);
  }
  return std::make_tuple(rendered, out_color, radii, pixels, out_observe, out_opac, out_all_map, out_plane_depth, out_opac_s, geomBuffer_rt, binningBuffer_rt, imgBuffer_rt);
}

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
	const torch::Tensor& imgBuffer_cv)
{
  const int P = means3D.size(0);
  const int H = dL_dout_color.size(1);
  const int W = dL_dout_color.size(2);
  
  int M = 0;
  if(sh.size(0) != 0)
  {	
	M = sh.size(1);
  }

  // Determine which device the tensor is on
  auto device = means3D.device();
  int device_index = device.index(); // Get the index of the device
  // Set the current CUDA device to the device where 'points' is located
  cudaSetDevice(device_index);

  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D_abs = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means3D.options());
  torch::Tensor dL_dall_map = torch::zeros({P, NUM_ALL_MAP}, means3D.options());
  torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
  torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
  torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());
  torch::Tensor depth = torch::full({P, 1}, 0.0, means3D.options());
  
  if(P != 0)
  {  
	  CudaRasterizer::Rasterizer::backwardwithconvert(P, degree, M, R,
	  background.contiguous().data<float>(),
	  all_map_pixels.contiguous().data<float>(),
	  W, H, 
	  means3D.contiguous().data<float>(),
	  sh.contiguous().data<float>(),
	  colors.contiguous().data<float>(),
	  all_maps.contiguous().data<float>(),
	  sky_pixels.contiguous().data<float>(),
	  scales.data_ptr<float>(),
	  scale_modifier,
	  rotations.data_ptr<float>(),
	  cov3D_precomp.contiguous().data<float>(),
	  viewmatrix.contiguous().data<float>(),
	  projmatrix.contiguous().data<float>(),
	  campos.contiguous().data<float>(),
	  tan_fovx,
	  tan_fovy,
	  radii.contiguous().data<int>(),
	  reinterpret_cast<char*>(geomBuffer_rt.contiguous().data_ptr()),
	  reinterpret_cast<char*>(binningBuffer_rt.contiguous().data_ptr()),
	  reinterpret_cast<char*>(imageBuffer_rt.contiguous().data_ptr()),
	  dL_dout_color.contiguous().data<float>(),
	  dL_dout_opac.contiguous().data<float>(),
	  dL_dout_all_map.contiguous().data<float>(),
	  dL_dout_plane_depth.contiguous().data<float>(),
	  dL_dout_opac_s.contiguous().data<float>(),
	  label.contiguous().data<float>(),
	  dL_dmeans2D.contiguous().data<float>(),
	  dL_dmeans2D_abs.contiguous().data<float>(),
	  dL_dconic.contiguous().data<float>(),  
	  dL_dopacity.contiguous().data<float>(),
	  dL_dcolors.contiguous().data<float>(),
	  dL_dmeans3D.contiguous().data<float>(),
	  dL_dcov3D.contiguous().data<float>(),
	  dL_dsh.contiguous().data<float>(),
	  dL_dscales.contiguous().data<float>(),
	  dL_drotations.contiguous().data<float>(),
	  depth.contiguous().data<float>(),
	  dL_dall_map.contiguous().data<float>(),
	  render_geo,
	  debug,
	  config.contiguous().data<float>(),
	  dL_dout_proj_depth.contiguous().data<float>(),
	  dL_dout_proj_normal.contiguous().data<float>(),
	  viewmatrix_cv.contiguous().data<float>(),
	  transmatrix_cv.contiguous().data<float>(),
	  debug_cv,
	  ref_fx,
	  ref_fy,
	  src_fx,
	  src_fy,
	  ref_cx,
	  ref_cy,
	  src_cx,
	  src_cy,
	  ref_height,
	  ref_width,
	  src_height,
	  src_width,
	  num_projected,
	  MaxNumVertices,
	  MaxNumFaces,
	  vertices.contiguous().data<float>(),
	  rays.contiguous().data<float>(),
	  faces.contiguous().data<int>(),
	  pixel_label.contiguous().data<int>(),
	  faceNormal.contiguous().data<float>(),
	  reinterpret_cast<char*>(geomBuffer_cv.contiguous().data_ptr()),
	  reinterpret_cast<char*>(binningBuffer_cv.contiguous().data_ptr()),
	  reinterpret_cast<char*>(imgBuffer_cv.contiguous().data_ptr()));
  }

  return std::make_tuple(dL_dmeans2D, dL_dmeans2D_abs, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_dsh, dL_dscales, dL_drotations, depth, dL_dall_map);
}

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
	const torch::Tensor& geomBuffer_rt,
	const int R,
	const torch::Tensor& binningBuffer_rt,
	const torch::Tensor& imageBuffer_rt,
	const bool render_geo,
	const bool debug,
	const torch::Tensor& config,
	const torch::Tensor& label) 
{
  const int P = means3D.size(0);
  const int H = dL_dout_color.size(1);
  const int W = dL_dout_color.size(2);
  
  int M = 0;
  if(sh.size(0) != 0)
  {	
	M = sh.size(1);
  }

  // Determine which device the tensor is on
  auto device = means3D.device();
  int device_index = device.index(); // Get the index of the device
  // Set the current CUDA device to the device where 'points' is located
  cudaSetDevice(device_index);

  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D_abs = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means3D.options());
  torch::Tensor dL_dall_map = torch::zeros({P, NUM_ALL_MAP}, means3D.options());
  torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
  torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
  torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());
  torch::Tensor depth = torch::full({P, 1}, 0.0, means3D.options());
  
  if(P != 0)
  {  
	  CudaRasterizer::Rasterizer::backward(P, degree, M, R,
	  background.contiguous().data<float>(),
	  all_map_pixels.contiguous().data<float>(),
	  W, H, 
	  means3D.contiguous().data<float>(),
	  sh.contiguous().data<float>(),
	  colors.contiguous().data<float>(),
	  all_maps.contiguous().data<float>(),
	  sky_pixels.contiguous().data<float>(),
	  scales.data_ptr<float>(),
	  scale_modifier,
	  rotations.data_ptr<float>(),
	  cov3D_precomp.contiguous().data<float>(),
	  viewmatrix.contiguous().data<float>(),
	  projmatrix.contiguous().data<float>(),
	  campos.contiguous().data<float>(),
	  tan_fovx,
	  tan_fovy,
	  radii.contiguous().data<int>(),
	  reinterpret_cast<char*>(geomBuffer_rt.contiguous().data_ptr()),
	  reinterpret_cast<char*>(binningBuffer_rt.contiguous().data_ptr()),
	  reinterpret_cast<char*>(imageBuffer_rt.contiguous().data_ptr()),
	  dL_dout_color.contiguous().data<float>(),
	  dL_dout_opac.contiguous().data<float>(),
	  dL_dout_all_map.contiguous().data<float>(),
	  dL_dout_plane_depth.contiguous().data<float>(),
	  dL_dout_opac_s.contiguous().data<float>(),
	  label.contiguous().data<float>(),
	  dL_dmeans2D.contiguous().data<float>(),
	  dL_dmeans2D_abs.contiguous().data<float>(),
	  dL_dconic.contiguous().data<float>(),  
	  dL_dopacity.contiguous().data<float>(),
	  dL_dcolors.contiguous().data<float>(),
	  dL_dmeans3D.contiguous().data<float>(),
	  dL_dcov3D.contiguous().data<float>(),
	  dL_dsh.contiguous().data<float>(),
	  dL_dscales.contiguous().data<float>(),
	  dL_drotations.contiguous().data<float>(),
	  depth.contiguous().data<float>(),
	  dL_dall_map.contiguous().data<float>(),
	  render_geo,
	  debug,
	  config.contiguous().data<float>());
  }

  return std::make_tuple(dL_dmeans2D, dL_dmeans2D_abs, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_dsh, dL_dscales, dL_drotations, depth, dL_dall_map);
}

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
    const bool use_mask)
{
  // Determine which device the tensor is on
  auto device_avai = DepthMap.device();
  int device_index = device_avai.index(); // Get the index of the device
  // Set the current CUDA device to the device where 'DepthMap' is located
  cudaSetDevice(device_index);

  const int MaxNumVertices = ref_height * ref_width;
  const int MaxNumFaces = 2 * (ref_height - 1) * (ref_width - 1);
  const int MaxNumPixels = src_height * src_width;

  auto int32_opts = DepthMap.options().dtype(torch::kInt32);
  auto float_opts = DepthMap.options().dtype(torch::kFloat32);
  auto int64_opts = DepthMap.options().dtype(torch::kInt64);

  torch::Device device(torch::kCUDA, device_index);
  torch::TensorOptions options(torch::kByte);

  torch::Tensor vertices_ref = torch::full({3, MaxNumVertices}, 0.0, float_opts);
  torch::Tensor vertices = torch::full({3, MaxNumVertices}, 0.0, float_opts);
  torch::Tensor points_image = torch::full({2, ref_height, ref_width}, 0.0, float_opts);
  torch::Tensor rays = torch::full({3, MaxNumVertices}, 0.0, float_opts);
  torch::Tensor faces = torch::full({3, MaxNumFaces}, 0, int32_opts);
  torch::Tensor faceIndicator = torch::full({MaxNumFaces}, 1, int32_opts);
  torch::Tensor pixel_label = torch::full({src_height * src_width}, 0, int32_opts);
  torch::Tensor faceNormal_ref = torch::full({3, MaxNumFaces}, 0.0, float_opts);
  torch::Tensor faceNormal = torch::full({3, MaxNumFaces}, 0.0, float_opts);
  torch::Tensor geomBuffer_cv = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer_cv = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer_cv = torch::empty({0}, options.device(device));
  torch::Tensor projected_depth = torch::full({1, src_height, src_width}, 0.0, float_opts);
  torch::Tensor projected_normal = torch::full({3, src_height, src_width}, 0.0, float_opts);

  std::function<char*(size_t)> geomFunc_cv = resizeFunctional(geomBuffer_cv);
  std::function<char*(size_t)> binningFunc_cv = resizeFunctional(binningBuffer_cv);
  std::function<char*(size_t)> imgFunc_cv = resizeFunctional(imgBuffer_cv);

  if (!use_mask) {
    ref_Mask.fill_(1);
    src_Mask.fill_(1);
  }

  int64_t ref_points = ref_Mask.sum().item<int64_t>();
  int num_projected = 0;
  if (ref_points != 0)
  {
    CudaConverter::Converter::depth2mesh2depth(
      geomFunc_cv,
      binningFunc_cv,
      imgFunc_cv,
      ref_height, ref_width,
      src_height, src_width,
      ref_fx, ref_fy,
      src_fx, src_fy,
      ref_cx, ref_cy,
	  src_cx, src_cy,
      viewmatrix.contiguous().data<float>(),
      transmatrix.contiguous().data<float>(),
      DepthMap.contiguous().data<float>(),
      ref_Mask.contiguous().data<int>(),
      src_Mask.contiguous().data<int>(),
	  vertices_ref.contiguous().data<float>(),
      vertices.contiguous().data<float>(),
	  points_image.contiguous().data<float>(),
	  rays.contiguous().data<float>(),
      faces.contiguous().data<int>(),
      faceIndicator.contiguous().data<int>(),
      pixel_label.contiguous().data<int>(),
	  faceNormal_ref.contiguous().data<float>(),
      faceNormal.contiguous().data<float>(),
      projected_depth.contiguous().data<float>(),
      projected_normal.contiguous().data<float>(),
      MaxNumVertices,
      MaxNumFaces,
      MaxNumPixels,
	  num_projected,
      debug);

    cudaDeviceSynchronize();
  }

  return std::make_tuple(num_projected, MaxNumVertices, MaxNumFaces, projected_depth, projected_normal, vertices, rays, faces, pixel_label, faceNormal_ref, faceNormal, geomBuffer_cv, binningBuffer_cv, imgBuffer_cv);

}

torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix)
{ 

  // Determine which device the tensor is on
  auto device = means3D.device();
  int device_index = device.index(); // Get the index of the device
  // Set the current CUDA device to the device where 'points' is located
  cudaSetDevice(device_index);

  const int P = means3D.size(0);
  
  torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));
 
  if(P != 0)
  {
	CudaRasterizer::Rasterizer::markVisible(P,
		means3D.contiguous().data<float>(),
		viewmatrix.contiguous().data<float>(),
		projmatrix.contiguous().data<float>(),
		present.contiguous().data<bool>());
  }
  
  return present;
}

torch::Tensor patch_offsets(int h_patch_size, torch::Device device) {
	auto offsets = torch::arange(-h_patch_size, h_patch_size + 1, torch::TensorOptions().dtype(torch::kInt32).device(device));
	auto mesh = torch::meshgrid({offsets, offsets}, /*indexing=*/"xy");
	auto stacked = torch::stack({mesh[0], mesh[1]}, -1);

	return stacked.view({1, -1, 2});
}