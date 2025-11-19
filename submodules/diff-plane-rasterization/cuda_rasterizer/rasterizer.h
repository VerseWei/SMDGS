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

#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <vector>
#include <functional>

namespace CudaRasterizer
{
	class Rasterizer
	{
	public:

		static void markVisible(
			int P,
			float* means3D,
			float* viewmatrix,
			float* projmatrix,
			bool* present);

		static int forward(
			std::function<char* (size_t)> geometryBuffer_rt,
			std::function<char* (size_t)> binningBuffer_rt,
			std::function<char* (size_t)> imageBuffer_rt,
			const int P, int D, int M,
			const float* background,
			const int width, int height,
			const float* means3D,
			const float* shs,
			const float* colors_precomp,
			const float* opacities,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* all_map,
			const float* sky_pixels,
			const float* viewmatrix,
			const float* projmatrix,
			const float* cam_pos,
			const float tan_fovx, float tan_fovy,
			const bool prefiltered,
			float* config,
			float* out_color,
			float* pixels,
			int* radii,
			int* out_observe,
			float* out_opac,
			float* out_all_map,
			float* out_plane_depth,
			float* out_opac_s,
			const float* label,
			const bool render_geo,
			bool debug = false);

		static void backward(
			const int P, int D, int M, int R,
			const float* background,
			const float* all_map_pixels,
			const int width, int height,
			const float* means3D,
			const float* shs,
			const float* colors_precomp,
			const float* all_maps,
			const float* sky_pixels,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float* campos,
			const float tan_fovx, float tan_fovy,
			const int* radii,
			char* geom_buffer_rt,
			char* binning_buffer_rt,
			char* image_buffer_rt,
			const float* dL_dpix,
			const float* dL_dpixopac,
			const float* dL_dout_all_map,
			const float* dL_dout_plane_depth,
			const float* dL_dpixopac_s,
			const float* label,
			float* dL_dmean2D,
			float* dL_dmean2D_abs,
			float* dL_dconic,
			float* dL_dopacity,
			float* dL_dcolor,
			float* dL_dmean3D,
			float* dL_dcov3D,
			float* dL_dsh,
			float* dL_dscale,
			float* dL_drot,
			float* depth,
			float* dL_dall_map,
			const bool render_geo,
			bool debug,
			float* config);

		static void backwardwithconvert(
			const int P, int D, int M, int R,
			const float* background,
			const float* all_map_pixels,
			const int width, int height,
			const float* means3D,
			const float* shs,
			const float* colors_precomp,
			const float* all_maps,
			const float* sky_pixels,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float* campos,
			const float tan_fovx, float tan_fovy,
			const int* radii,
			char* geom_buffer_rt,
			char* binning_buffer_rt,
			char* img_buffer_rt,
			const float* dL_dpix,
			const float* dL_dpixopac,
			float* dL_dout_all_map,
			float* dL_dout_plane_depth,
			const float* dL_dpixopac_s,
			const float* label,
			float* dL_dmean2D,
			float* dL_dmean2D_abs,
			float* dL_dconic,
			float* dL_dopacity,
			float* dL_dcolor,
			float* dL_dmean3D,
			float* dL_dcov3D,
			float* dL_dsh,
			float* dL_dscale,
			float* dL_drot,
			float* depth,
			float* dL_dall_map,
			const bool render_geo,
			bool debug,
			float* config,
			const float* dL_dout_proj_depth,
			const float* dL_dout_proj_normal,
			const float* viewmatrix_cv,
			const float* transmatrix_cv,
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
			const float* vertices,
			const float* rays,
			const int* faces,
			const int* pixel_label,
			const float* faceNormal,
			char* geom_buffer_cv,
			char* binning_buffer_cv,
			char* img_buffer_cv);
	};
};

#endif