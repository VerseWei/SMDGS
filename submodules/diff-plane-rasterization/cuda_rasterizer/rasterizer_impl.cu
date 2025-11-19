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

#include "rasterizer_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"

// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb_rt(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Gaussians that pass it.
__global__ void checkFrustum(int P,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool* present)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 p_view;
	present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
}

__global__ void addpixeldepthgradbymultincc(
	float* __restrict__ dL_dout_plane_depth,
	const float* __restrict__ dL_dgrid,
	const float* __restrict__ viewmatrix,
	const float* __restrict__ transmatrix,
	const int ref_height,
	const int ref_width,
	const float src_fx,
	const float src_fy,
	const int MaxNumVertices,
	const int total_patch_size,
	const int sample_num,
	const float* __restrict__ ref_grid,
	const float* __restrict__ vertices)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= sample_num * total_patch_size)
		return;

	const float2 pixf = { ref_grid[idx * 2], ref_grid[idx * 2 + 1] };
	const float2 dL_dps = { dL_dgrid[idx * 2], dL_dgrid[idx * 2 + 1] };
	const int2 pix = { (int)pixf.x, (int)pixf.y };

	if (pix.x < 0 || pix.x >= ref_width || pix.y < 0 || pix.y >= ref_height)
		return;

	const int pix_id = pix.y * ref_width + pix.x;
	float3 As;
	As.x = vertices[0 * MaxNumVertices + pix_id];
	As.y = vertices[1 * MaxNumVertices + pix_id];
	As.z = vertices[2 * MaxNumVertices + pix_id];

	const float3 dAs_dArz = { transmatrix[0] * viewmatrix[2] + transmatrix[4] * viewmatrix[6] + transmatrix[8] * viewmatrix[10],
							  transmatrix[1] * viewmatrix[2] + transmatrix[5] * viewmatrix[6] + transmatrix[9] * viewmatrix[10],
							  transmatrix[2] * viewmatrix[2] + transmatrix[6] * viewmatrix[6] + transmatrix[10] * viewmatrix[10] };

	float2 dps_dArz;
	dps_dArz.x = (As.z * dAs_dArz.x - As.x * dAs_dArz.z) / ((As.z + 0.0000001f) * (As.z + 0.0000001f));
	dps_dArz.y = (As.z * dAs_dArz.y - As.y * dAs_dArz.z) / ((As.z + 0.0000001f) * (As.z + 0.0000001f));

	atomicAdd(&(dL_dout_plane_depth[pix_id]), src_fx * dL_dps.x * dps_dArz.x + src_fy * dL_dps.y * dps_dArz.y); 

}

__global__ void addpixeldepthgradbymultigeo(
	float* __restrict__ dL_dout_plane_depth,
	const float* __restrict__  dL_dDs,
	const float* __restrict__ dL_dNs,
	const float* __restrict__ viewmatrix_cv,
	const float* __restrict__ transmatrix_cv,
	const int src_height,
	const int src_width,
	const float src_fx,
	const float src_fy,
	const int MaxNumVertices,
	const int MaxNumFaces,
	const float* __restrict__ rays,
	const int* __restrict__ faces,
	const float* __restrict__ faceNormal,
	const float* __restrict__ vertices_src,
	const float* __restrict__ vertices2D,
	const uint32_t* __restrict__ interp_id,
	const float* __restrict__ interp_weight,
	const uint32_t* __restrict__ interp_marker)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= src_height * src_width)
		return;
	if (interp_marker[idx] > 0)
	{
		const int coll_id = interp_id[idx];
		const int ref_pixel_A = faces[0 * MaxNumFaces + coll_id];
		const int ref_pixel_B = faces[1 * MaxNumFaces + coll_id];
		const int ref_pixel_C = faces[2 * MaxNumFaces + coll_id];

		if (dL_dDs[idx] != 0)
		{
			const float3 weight = { interp_weight[3 * idx + 0], interp_weight[3 * idx + 1], interp_weight[3 * idx + 2] };
			const float _alpha = weight.x;
			const float _beta = weight.y;
			const float _gamma = weight.z;
			const float3 dAsx_dAw = { transmatrix_cv[0], transmatrix_cv[4], transmatrix_cv[8] };
			const float3 dBsx_dBw = dAsx_dAw;
			const float3 dCsx_dCw = dAsx_dAw;
			const float3 dAsy_dAw = { transmatrix_cv[1], transmatrix_cv[5], transmatrix_cv[9] };
			const float3 dBsy_dBw = dAsy_dAw;
			const float3 dCsy_dCw = dAsy_dAw;
			const float3 dAsz_dAw = { transmatrix_cv[2], transmatrix_cv[6], transmatrix_cv[10] };
			const float3 dBsz_dBw = dAsz_dAw;
			const float3 dCsz_dCw = dAsz_dAw;
			const float3 dAw_dArz = { viewmatrix_cv[2], viewmatrix_cv[6], viewmatrix_cv[10] };
			const float3 dBw_dBrz = dAw_dArz;
			const float3 dCw_dCrz = dAw_dArz;
			
			atomicAdd(&(dL_dout_plane_depth[ref_pixel_A]), dL_dDs[idx] * _alpha * (dAsz_dAw.x * dAw_dArz.x + dAsz_dAw.y * dAw_dArz.y + dAsz_dAw.z * dAw_dArz.z));
			atomicAdd(&(dL_dout_plane_depth[ref_pixel_B]), dL_dDs[idx] * _beta * (dBsz_dBw.x * dBw_dBrz.x + dBsz_dBw.y * dBw_dBrz.y + dBsz_dBw.z * dBw_dBrz.z));
			atomicAdd(&(dL_dout_plane_depth[ref_pixel_C]), dL_dDs[idx] * _gamma * (dCsz_dCw.x * dCw_dCrz.x + dCsz_dCw.y * dCw_dCrz.y + dCsz_dCw.z * dCw_dCrz.z));					 

			const float3 As = { vertices_src[9 * coll_id + 0], vertices_src[9 * coll_id + 1], vertices_src[9 * coll_id + 2] };
			const float3 Bs = { vertices_src[9 * coll_id + 3], vertices_src[9 * coll_id + 4], vertices_src[9 * coll_id + 5] };
			const float3 Cs = { vertices_src[9 * coll_id + 6], vertices_src[9 * coll_id + 7], vertices_src[9 * coll_id + 8] };

			float2 a = { vertices2D[6 * coll_id + 0], vertices2D[6 * coll_id + 1] };  
            float2 b = { vertices2D[6 * coll_id + 2], vertices2D[6 * coll_id + 3] };
            float2 c = { vertices2D[6 * coll_id + 4], vertices2D[6 * coll_id + 5] };

			const uint2 pix = { (int)idx % src_width, (int)idx / src_width };
			const float x = (float)pix.x;
			const float y = (float)pix.y;

			const float3 dxa_dAs = {src_fx / As.z, 0, -src_fx * As.x / (As.z * As.z)};      
			const float3 dya_dAs = {0, src_fy / As.z, -src_fy * As.y / (As.z * As.z)};   
			const float3 dxb_dBs = {src_fx / Bs.z, 0, -src_fx * Bs.x / (Bs.z * Bs.z)};
			const float3 dyb_dBs = {0, src_fy / Bs.z, -src_fy * Bs.y / (Bs.z * Bs.z)};
			const float3 dxc_dCs = {src_fx / Cs.z, 0, -src_fx * Cs.x / (Cs.z * Cs.z)};
			const float3 dyc_dCs = {0, src_fy / Cs.z, -src_fy * Cs.y / (Cs.z * Cs.z)};
			
			const float N = (b.x - a.x) * (y - a.y) - (x - a.x) * (b.y - a.y);  
			const float D = (b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y);
			const float eps = 1e-8f;
			const float D_safe = (fabsf(D) < eps) ? copysignf(eps, D) : D;
			const float D2 = 1.0f / (D_safe * D_safe);

			const float gamma = N / D_safe;
			const float B = x - a.x - gamma * (c.x - a.x);
			const float S = b.x - a.x;
			const float S_safe = (fabsf(S) < eps) ? copysignf(eps, S) : S;
			const float S1 = 1.0f / S_safe;
			const float S2 = 1.0f / (S_safe * S_safe);
			const float beta = B / S_safe;
			const float alpha = 1.0f - beta - gamma;
			
			float2 dgamma_da = {(b.y - y) * D - N * (b.y - c.y), (x - b.x) * D - N * (c.x - b.x)};
			float2 dgamma_db = {(y - a.y) * D - N * (c.y - a.y), -(x - a.x) * D + N * (c.x - a.x)};
			float2 dgamma_dc = {N * (b.y - a.y), -N * (b.x - a.x)};
			dgamma_da.x *= D2;
			dgamma_da.y *= D2;
			dgamma_db.x *= D2;
			dgamma_db.y *= D2;
			dgamma_dc.x *= D2;
			dgamma_dc.y *= D2;

			const float2 dbeta_da = {(-1 - (c.x - a.x) * dgamma_da.x + gamma) * S1 + B * S2, -(c.x - a.x) * dgamma_da.y * S1};
			const float2 dbeta_db = {-(c.x - a.x) * dgamma_db.x * S1 - B * S2, -(c.x - a.x) * dgamma_db.y * S1};
			const float2 dbeta_dc = {(-(c.x - a.x) * dgamma_dc.x - gamma) * S1, -(c.x - a.x) * dgamma_dc.y * S1};
			
			float2 dalpha_da = { - dbeta_da.x - dgamma_da.x, - dbeta_da.y - dgamma_da.y };
			float2 dalpha_db = { - dbeta_db.x - dgamma_db.x, - dbeta_db.y - dgamma_db.y };
			float2 dalpha_dc = { - dbeta_dc.x - dgamma_dc.x, - dbeta_dc.y - dgamma_dc.y };

			const float3 dalpha_dAs = {dalpha_da.x * dxa_dAs.x + dalpha_da.y * dya_dAs.x,
									   dalpha_da.x * dxa_dAs.y + dalpha_da.y * dya_dAs.y,
									   dalpha_da.x * dxa_dAs.z + dalpha_da.y * dya_dAs.z};
			const float3 dalpha_dBs = {dalpha_db.x * dxb_dBs.x + dalpha_db.y * dyb_dBs.x,
									   dalpha_db.x * dxb_dBs.y + dalpha_db.y * dyb_dBs.y,
									   dalpha_db.x * dxb_dBs.z + dalpha_db.y * dyb_dBs.z};
			const float3 dalpha_dCs = {dalpha_dc.x * dxc_dCs.x + dalpha_dc.y * dyc_dCs.x,
									   dalpha_dc.x * dxc_dCs.y + dalpha_dc.y * dyc_dCs.y,
									   dalpha_dc.x * dxc_dCs.z + dalpha_dc.y * dyc_dCs.z};
			
			const float3 dbeta_dAs = {dbeta_da.x * dxa_dAs.x + dbeta_da.y * dya_dAs.x,
									  dbeta_da.x * dxa_dAs.y + dbeta_da.y * dya_dAs.y,
									  dbeta_da.x * dxa_dAs.z + dbeta_da.y * dya_dAs.z};
			const float3 dbeta_dBs = {dbeta_db.x * dxb_dBs.x + dbeta_db.y * dyb_dBs.x,
									  dbeta_db.x * dxb_dBs.y + dbeta_db.y * dyb_dBs.y,
								      dbeta_db.x * dxb_dBs.z + dbeta_db.y * dyb_dBs.z};
			const float3 dbeta_dCs = {dbeta_dc.x * dxc_dCs.x + dbeta_dc.y * dyc_dCs.x,
									  dbeta_dc.x * dxc_dCs.y + dbeta_dc.y * dyc_dCs.y,
									  dbeta_dc.x * dxc_dCs.z + dbeta_dc.y * dyc_dCs.z};

			const float3 dgamma_dAs = {dgamma_da.x * dxa_dAs.x + dgamma_da.y * dya_dAs.x,
									   dgamma_da.x * dxa_dAs.y + dgamma_da.y * dya_dAs.y,
									   dgamma_da.x * dxa_dAs.z + dgamma_da.y * dya_dAs.z};
			const float3 dgamma_dBs = {dgamma_db.x * dxb_dBs.x + dgamma_db.y * dyb_dBs.x,
									   dgamma_db.x * dxb_dBs.y + dgamma_db.y * dyb_dBs.y,
									   dgamma_db.x * dxb_dBs.z + dgamma_db.y * dyb_dBs.z};
			const float3 dgamma_dCs = {dgamma_dc.x * dxc_dCs.x + dgamma_dc.y * dyc_dCs.x,
									   dgamma_dc.x * dxc_dCs.y + dgamma_dc.y * dyc_dCs.y,
									   dgamma_dc.x * dxc_dCs.z + dgamma_dc.y * dyc_dCs.z};

			const float3 dAs_dArz = {dot3(dAsx_dAw, dAw_dArz), dot3(dAsy_dAw, dAw_dArz), dot3(dAsz_dAw, dAw_dArz)};
			const float3 dBs_dBrz = dAs_dArz;
			const float3 dCs_dCrz = dAs_dArz;

			const float dalpha_dArz = dot3(dalpha_dAs, dAs_dArz);
			const float dalpha_dBrz = dot3(dalpha_dBs, dBs_dBrz);
			const float dalpha_dCrz = dot3(dalpha_dCs, dCs_dCrz);

			const float dbeta_dArz = dot3(dbeta_dAs, dAs_dArz);
			const float dbeta_dBrz = dot3(dbeta_dBs, dBs_dBrz);
			const float dbeta_dCrz = dot3(dbeta_dCs, dCs_dCrz);

			const float dgamma_dArz = dot3(dgamma_dAs, dAs_dArz);
			const float dgamma_dBrz = dot3(dgamma_dBs, dBs_dBrz);
			const float dgamma_dCrz = dot3(dgamma_dCs, dCs_dCrz);

			atomicAdd(&(dL_dout_plane_depth[ref_pixel_A]), dL_dDs[idx] * (As.z * dalpha_dArz + Bs.z * dbeta_dArz + Cs.z * dgamma_dArz));
			atomicAdd(&(dL_dout_plane_depth[ref_pixel_A]), dL_dDs[idx] * (As.z * dalpha_dBrz + Bs.z * dbeta_dBrz + Cs.z * dgamma_dBrz));
			atomicAdd(&(dL_dout_plane_depth[ref_pixel_A]), dL_dDs[idx] * (As.z * dalpha_dCrz + Bs.z * dbeta_dCrz + Cs.z * dgamma_dCrz));
		}
	}
}

// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeys_rt(
	int P,
	const float2* points_xy,
	const float* depths,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	int* radii,
	dim3 grid)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Generate no key/value pair for invisible Gaussians
	if (radii[idx] > 0)
	{
		// Find this Gaussian's offset in buffer for writing keys/values.
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		uint2 rect_min, rect_max;

		getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);

		// For each tile that the bounding rect overlaps, emit a 
		// key/value pair. The key is |  tile ID  |      depth      |,
		// and the value is the ID of the Gaussian. Sorting the values 
		// with this key yields Gaussian IDs in a list, such that they
		// are first sorted by tile and then by depth. 
		for (int y = rect_min.y; y < rect_max.y; y++)
		{
			for (int x = rect_min.x; x < rect_max.x; x++)
			{
				uint64_t key = y * grid.x + x;
				key <<= 32;
				key |= *((uint32_t*)&depths[idx]);
				gaussian_keys_unsorted[off] = key;
				gaussian_values_unsorted[off] = idx;
				off++;
			}
		}
	}
}

// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;
}

// Mark Gaussians as visible/invisible, based on view frustum testing
void CudaRasterizer::Rasterizer::markVisible(
	int P,
	float* means3D,
	float* viewmatrix,
	float* projmatrix,
	bool* present)
{
	checkFrustum << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		viewmatrix, projmatrix,
		present);
}

CudaRasterizer::GeometryState_rt CudaRasterizer::GeometryState_rt::fromChunk(char*& chunk, size_t P)
{
	GeometryState_rt geom_rt;
	obtain(chunk, geom_rt.depths, P, 128);
	obtain(chunk, geom_rt.clamped, P * 3, 128);
	obtain(chunk, geom_rt.internal_radii, P, 128);
	obtain(chunk, geom_rt.means2D, P, 128);
	obtain(chunk, geom_rt.cov3D, P * 6, 128);
	obtain(chunk, geom_rt.conic_opacity, P, 128);
	obtain(chunk, geom_rt.rgb, P * 3, 128);
	obtain(chunk, geom_rt.tiles_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom_rt.scan_size, geom_rt.tiles_touched, geom_rt.tiles_touched, P);
	obtain(chunk, geom_rt.scanning_space, geom_rt.scan_size, 128);
	obtain(chunk, geom_rt.point_offsets, P, 128);
	return geom_rt;
}

CudaRasterizer::ImageState_rt CudaRasterizer::ImageState_rt::fromChunk(char*& chunk, size_t N)
{
	ImageState_rt img_rt;
	obtain(chunk, img_rt.accum_alpha, N, 128);
	obtain(chunk, img_rt.accum_normal, 3 * N, 128);
	obtain(chunk, img_rt.accum_dist, N, 128);
	obtain(chunk, img_rt.n_contrib, N, 128);
	obtain(chunk, img_rt.ranges, N, 128);
	obtain(chunk, img_rt.accum_alpha_s, N, 128);
	return img_rt;
}

CudaRasterizer::BinningState_rt CudaRasterizer::BinningState_rt::fromChunk(char*& chunk, size_t P)
{
	BinningState_rt binning_rt;
	obtain(chunk, binning_rt.point_list, P, 128);
	obtain(chunk, binning_rt.point_list_unsorted, P, 128);
	obtain(chunk, binning_rt.point_list_keys, P, 128);
	obtain(chunk, binning_rt.point_list_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning_rt.sorting_size,
		binning_rt.point_list_keys_unsorted, binning_rt.point_list_keys,
		binning_rt.point_list_unsorted, binning_rt.point_list, P);
	obtain(chunk, binning_rt.list_sorting_space, binning_rt.sorting_size, 128);
	return binning_rt;
}

CudaRasterizer::GeometryState_cv CudaRasterizer::GeometryState_cv::fromChunk(char*& chunk, size_t P)
{
	GeometryState_cv geom_cv;
	obtain(chunk, geom_cv.pixels_touched, P, 128);
    obtain(chunk, geom_cv.bounding_box, 4 * P, 128);
    obtain(chunk, geom_cv.vertices2D, 6 * P, 128);
    obtain(chunk, geom_cv.verticesZ, 3 * P, 128);
	obtain(chunk, geom_cv.vertices_src, 9 * P, 128);
    cub::DeviceScan::InclusiveSum(nullptr, geom_cv.scan_size, geom_cv.pixels_touched, geom_cv.pixels_touched, P);
	obtain(chunk, geom_cv.scanning_space, geom_cv.scan_size, 128);
	obtain(chunk, geom_cv.point_offsets, P, 128);
	return geom_cv;
}

CudaRasterizer::BinningState_cv CudaRasterizer::BinningState_cv::fromChunk(char*& chunk, size_t P)
{
	BinningState_cv binning_cv;
	obtain(chunk, binning_cv.face_list, P, 128);
	obtain(chunk, binning_cv.face_list_unsorted, P, 128);
	obtain(chunk, binning_cv.face_list_keys, P, 128);
	obtain(chunk, binning_cv.face_list_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning_cv.sorting_size,
		binning_cv.face_list_keys_unsorted, binning_cv.face_list_keys,
		binning_cv.face_list_unsorted, binning_cv.face_list, P);
	obtain(chunk, binning_cv.list_sorting_space, binning_cv.sorting_size, 128);
	return binning_cv;
}

CudaRasterizer::ImageState_cv CudaRasterizer::ImageState_cv::fromChunk(char*& chunk, size_t N)
{
	ImageState_cv img_cv;
	obtain(chunk, img_cv.ranges, N, 128);
    obtain(chunk, img_cv.interp_id, N, 128);
    obtain(chunk, img_cv.interp_weight, 3 * N, 128);
    obtain(chunk, img_cv.interp_marker, N, 128);
	return img_cv;
}


// Forward rendering procedure for differentiable rasterization
// of Gaussians.
int CudaRasterizer::Rasterizer::forward(
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
	bool debug)
{
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	size_t chunk_size = required<GeometryState_rt>(P);
	char* chunkptr = geometryBuffer_rt(chunk_size);
	GeometryState_rt geomState_rt = GeometryState_rt::fromChunk(chunkptr, P);

	if (radii == nullptr)
	{
		radii = geomState_rt.internal_radii;
	}

	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Dynamically resize image-based auxiliary buffers during training
	size_t img_chunk_size = required<ImageState_rt>(width * height);
	char* img_chunkptr = imageBuffer_rt(img_chunk_size);
	ImageState_rt imgState_rt = ImageState_rt::fromChunk(img_chunkptr, width * height);

	if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}

	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
	CHECK_CUDA(FORWARD::preprocess(
		P, D, M,
		means3D,
		(glm::vec3*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		opacities,
		shs,
		geomState_rt.clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, projmatrix,
		(glm::vec3*)cam_pos,
		width, height,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		radii,
		geomState_rt.means2D,
		geomState_rt.depths,
		geomState_rt.cov3D,
		geomState_rt.rgb,
		geomState_rt.conic_opacity,
		tile_grid,
		geomState_rt.tiles_touched,
		prefiltered,
		config
	), debug)

	// Compute prefix sum over full list of touched tile counts by Gaussians
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]           每个高斯覆盖多少tile的前项和
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState_rt.scanning_space, geomState_rt.scan_size, geomState_rt.tiles_touched, geomState_rt.point_offsets, P), debug)

	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	int num_rendered;
	CHECK_CUDA(cudaMemcpy(&num_rendered, geomState_rt.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

	size_t binning_chunk_size = required<BinningState_rt>(num_rendered);
	char* binning_chunkptr = binningBuffer_rt(binning_chunk_size);
	BinningState_rt binningState_rt = BinningState_rt::fromChunk(binning_chunkptr, num_rendered);

	// For each instance to be rendered, produce adequate [ tile | depth ] key 
	// and corresponding dublicated Gaussian indices to be sorted
	duplicateWithKeys_rt << <(P + 255) / 256, 256 >> > (
		P,
		geomState_rt.means2D,
		geomState_rt.depths,
		geomState_rt.point_offsets,
		binningState_rt.point_list_keys_unsorted,
		binningState_rt.point_list_unsorted,
		radii,
		tile_grid)
	CHECK_CUDA(, debug)

	int bit = getHigherMsb_rt(tile_grid.x * tile_grid.y);

	// Sort complete list of (duplicated) Gaussian indices by keys
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState_rt.list_sorting_space,
		binningState_rt.sorting_size,
		binningState_rt.point_list_keys_unsorted, binningState_rt.point_list_keys,
		binningState_rt.point_list_unsorted, binningState_rt.point_list,
		num_rendered, 0, 32 + bit), debug)

	CHECK_CUDA(cudaMemset(imgState_rt.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);

	// Identify start and end of per-tile workloads in sorted list
	if (num_rendered > 0)
		identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
			num_rendered,
			binningState_rt.point_list_keys,
			imgState_rt.ranges);
	CHECK_CUDA(, debug)

	// Let each tile blend its range of Gaussians independently in parallel
	const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState_rt.rgb;
	CHECK_CUDA(FORWARD::render(
		tile_grid, block,
		imgState_rt.ranges,
		binningState_rt.point_list,
		width, height,
		focal_x, focal_y,
		float(width*0.5f), float(height*0.5f),
		viewmatrix,
		cam_pos,
		geomState_rt.means2D,
		feature_ptr,
		all_map,
		sky_pixels,
		geomState_rt.conic_opacity,
		imgState_rt.accum_alpha,
		imgState_rt.accum_alpha_s,
		imgState_rt.accum_normal,
		imgState_rt.accum_dist,
		imgState_rt.n_contrib,
		background,
		out_color,
		pixels,
		out_observe,
		out_opac,
		out_all_map,
		out_plane_depth,
		out_opac_s,
		label,
		render_geo,
		config), debug)

	return num_rendered;
}

// Produce necessary gradients for optimization, corresponding
// to forward render pass
void CudaRasterizer::Rasterizer::backward(
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
	float* config)
{
	GeometryState_rt geomState_rt = GeometryState_rt::fromChunk(geom_buffer_rt, P);
	BinningState_rt binningState_rt = BinningState_rt::fromChunk(binning_buffer_rt, R);
	ImageState_rt imgState_rt = ImageState_rt::fromChunk(img_buffer_rt, width * height);

	if (radii == nullptr)
	{
		radii = geomState_rt.internal_radii;
	}

	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Compute loss gradients w.r.t. 2D mean position, conic matrix,
	// opacity and RGB of Gaussians from per-pixel loss gradients.
	// If we were given precomputed colors and not SHs, use them.
	const float* color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState_rt.rgb;
	CHECK_CUDA(BACKWARD::render(
		tile_grid,
		block,
		imgState_rt.ranges,
		binningState_rt.point_list,
		width, height,
		focal_x, focal_y,
		background,
		geomState_rt.means2D,
		geomState_rt.conic_opacity,
		color_ptr,
		all_maps,
		all_map_pixels,
		sky_pixels,
		imgState_rt.accum_alpha,
		imgState_rt.accum_alpha_s,
		imgState_rt.accum_normal,
		imgState_rt.accum_dist,
		imgState_rt.n_contrib,
		dL_dpix,
		dL_dpixopac,
		dL_dout_all_map,
		dL_dout_plane_depth,
		dL_dpixopac_s,
		label,
		(float3*)dL_dmean2D,
		(float3*)dL_dmean2D_abs,
		(float4*)dL_dconic,
		dL_dopacity,
		dL_dcolor,
		dL_dall_map,
		render_geo,
		config), debug)

	// Take care of the rest of preprocessing. Was the precomputed covariance
	// given to us or a scales/rot pair? If precomputed, pass that. If not,
	// use the one we computed ourselves.
	const float* cov3D_ptr = (cov3D_precomp != nullptr) ? cov3D_precomp : geomState_rt.cov3D;
	CHECK_CUDA(BACKWARD::preprocess(P, D, M,
		(float3*)means3D,
		radii,
		shs,
		geomState_rt.clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		cov3D_ptr,
		viewmatrix,
		projmatrix,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		(glm::vec3*)campos,
		(float3*)dL_dmean2D,
		dL_dconic,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_dcov3D,
		dL_dsh,
		(glm::vec3*)dL_dscale,
		(glm::vec4*)dL_drot,
		depth,
		config), debug)
}


// Produce necessary gradients for optimization, corresponding
// to forward render pass
void CudaRasterizer::Rasterizer::backwardwithconvert(
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
	char* img_buffer_cv)
{
	GeometryState_cv geomState_cv = GeometryState_cv::fromChunk(geom_buffer_cv, MaxNumFaces);
	BinningState_cv binningState_cv = BinningState_cv::fromChunk(binning_buffer_cv, num_projected);
	ImageState_cv imgState_cv = ImageState_cv::fromChunk(img_buffer_cv, src_width * src_height);

	// Compute the gradient of pixel-by-pixel depth induced by multi-view geometric consistency loss.
	addpixeldepthgradbymultigeo << <(src_height * src_width + 255) / 256, 256 >> > (
		dL_dout_plane_depth,
		dL_dout_proj_depth,
		dL_dout_proj_normal,
		viewmatrix_cv,
		transmatrix_cv,
		src_height,
		src_width,
		src_fx, src_fy,
		MaxNumVertices,
		MaxNumFaces,
		rays,
		faces,
		faceNormal,
		geomState_cv.vertices_src,
		geomState_cv.vertices2D,
		imgState_cv.interp_id,
		imgState_cv.interp_weight,
		imgState_cv.interp_marker);
	CHECK_CUDA(, debug_cv);	

	GeometryState_rt geomState_rt = GeometryState_rt::fromChunk(geom_buffer_rt, P);
	BinningState_rt binningState_rt = BinningState_rt::fromChunk(binning_buffer_rt, R);
	ImageState_rt imgState_rt = ImageState_rt::fromChunk(img_buffer_rt, width * height);

	if (radii == nullptr)
	{
		radii = geomState_rt.internal_radii;
	}

	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Compute loss gradients w.r.t. 2D mean position, conic matrix,
	// opacity and RGB of Gaussians from per-pixel loss gradients.
	// If we were given precomputed colors and not SHs, use them.
	const float* color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState_rt.rgb;
	CHECK_CUDA(BACKWARD::render(
		tile_grid,
		block,
		imgState_rt.ranges,
		binningState_rt.point_list,
		width, height,
		focal_x, focal_y,
		background,
		geomState_rt.means2D,
		geomState_rt.conic_opacity,
		color_ptr,
		all_maps,
		all_map_pixels,
		sky_pixels,
		imgState_rt.accum_alpha,
		imgState_rt.accum_alpha_s,
		imgState_rt.accum_normal,
		imgState_rt.accum_dist,
		imgState_rt.n_contrib,
		dL_dpix,
		dL_dpixopac,
		dL_dout_all_map,
		dL_dout_plane_depth,
		dL_dpixopac_s,
		label,
		(float3*)dL_dmean2D,
		(float3*)dL_dmean2D_abs,
		(float4*)dL_dconic,
		dL_dopacity,
		dL_dcolor,
		dL_dall_map,
		render_geo,
		config), debug)

	// Take care of the rest of preprocessing. Was the precomputed covariance
	// given to us or a scales/rot pair? If precomputed, pass that. If not,
	// use the one we computed ourselves.
	const float* cov3D_ptr = (cov3D_precomp != nullptr) ? cov3D_precomp : geomState_rt.cov3D;
	CHECK_CUDA(BACKWARD::preprocess(P, D, M,
		(float3*)means3D,
		radii,
		shs,
		geomState_rt.clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		cov3D_ptr,
		viewmatrix,
		projmatrix,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		(glm::vec3*)campos,
		(float3*)dL_dmean2D,
		dL_dconic,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_dcov3D,
		dL_dsh,
		(glm::vec3*)dL_dscale,
		(glm::vec4*)dL_drot,
		depth,
		config), debug)
}