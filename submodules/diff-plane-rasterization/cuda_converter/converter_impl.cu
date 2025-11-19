#include "converter_impl.h"
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

#include "auxiliary.h"
#include "building.h"
#include "projecting.h"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

uint32_t getHigherMsb_cv(uint32_t n)
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

// Generates one key/value pair for all face / pixel overlaps. 
// Run once per face (1:N mapping).
__global__ void duplicateWithKeys_cv(
	int P,
    const int W, int H,
	const uint32_t* offsets,
	uint64_t* face_keys_unsorted,
	uint32_t* face_values_unsorted,
    uint32_t* pixels_touched,
    uint32_t* bounding_box,
    float* vertices2D)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
        return;

	// Generate no key/value pair for invisible Gaussians
	if (pixels_touched[idx] > 0)
	{   
		// Find this face's offset in buffer for writing keys/values.
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		int minX = bounding_box[4 * idx + 0];
        int minY = bounding_box[4 * idx + 1];
        int maxX = bounding_box[4 * idx + 2];
        int maxY = bounding_box[4 * idx + 3];
        float2 point0_image = { vertices2D[6 * idx + 0], vertices2D[6 * idx + 1] };
        float2 point1_image = { vertices2D[6 * idx + 2], vertices2D[6 * idx + 3] };
        float2 point2_image = { vertices2D[6 * idx + 4], vertices2D[6 * idx + 5] };

		// Record key/value pair. The key is | pixels ID | pixels_touched |,
		// and the value is the ID of the pixel. Sorting the values 
		// with this key yields pixel IDs in a list, such that they
		// are first sorted by face and then by pixels_touched. 
        for (int y = minY; y < maxY; y++)
        {
            for (int x = minX; x < maxX; x++)
            {   
                float2 pixel_xy = { (float)x, (float)y };
                bool inTriangle = PointInTriangle(point0_image, point1_image, point2_image, pixel_xy);
                if (inTriangle) {
                    uint64_t key = y * W + x;
                    key <<= 32;
                    key |= *((uint32_t*)&pixels_touched[idx]);
                    face_keys_unsorted[off] = key;
                    face_values_unsorted[off] = idx;
                    off++;
                }
            }
        }
	}
}

// Check keys to see if it is at the start/end of one pixel's range in 
// the full sorted list. If yes, write start/end of this pixel. 
// Run once per instanced (duplicated) face ID.
__global__ void identifyFaceRanges(int L, uint64_t* face_list_keys, uint2* ranges, int* pixel_label)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read pixel ID from key. Update start/end of tile range if at limit.
	uint64_t key = face_list_keys[idx];
	uint32_t currpixel = key >> 32;
	if (idx == 0)
		ranges[currpixel].x = 0;
	else
	{
		uint32_t prevpixel = face_list_keys[idx - 1] >> 32;
		if (currpixel != prevpixel)
		{
			ranges[prevpixel].y = idx;
			ranges[currpixel].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currpixel].y = L;
    pixel_label[currpixel] = 1;
}

CudaConverter::GeometryState_cv CudaConverter::GeometryState_cv::fromChunk(char*& chunk, size_t P)
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

CudaConverter::BinningState_cv CudaConverter::BinningState_cv::fromChunk(char*& chunk, size_t P)
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

CudaConverter::ImageState_cv CudaConverter::ImageState_cv::fromChunk(char*& chunk, size_t N)
{
	ImageState_cv img_cv;
	obtain(chunk, img_cv.ranges, N, 128);
    obtain(chunk, img_cv.interp_id, N, 128);
    obtain(chunk, img_cv.interp_weight, 3 * N, 128);
    obtain(chunk, img_cv.interp_marker, N, 128);
	return img_cv;
}

void CudaConverter::Converter::depth2mesh2depth(
    std::function<char* (size_t)> geometryBuffer_cv,
    std::function<char* (size_t)> binningBuffer_cv,
    std::function<char* (size_t)> imageBuffer_cv,
    const int ref_height,
    const int ref_width,
    const int src_height,
    const int src_width,
    const float ref_focal_x, 
    const float ref_focal_y,
    const float src_focal_x,
    const float src_focal_y,
    const float ref_principal_x,
    const float ref_principal_y,
    const float src_principal_x,
    const float src_principal_y,
    const float* viewmatrix,
    const float* transmatrix,
    const float* depth,
    const int* ref_mask,
    const int* src_mask,
    float* vertices_ref,
    float* vertices,
    float* points_image,
    float* rays,
    int* faces,
    int* faceIndicator,
    int* pixel_label,
    float* faceNormal_ref,
    float* faceNormal,
    float* depthmap,
    float* normalmap,
    const int MaxNumVertices,
    const int MaxNumFaces,
    const int MaxNumPixels,
    int num_projected,
    const bool debug)
{
    size_t chunk_size = required<GeometryState_cv>(MaxNumFaces);
	char* chunkptr = geometryBuffer_cv(chunk_size);
	GeometryState_cv geomState_cv = GeometryState_cv::fromChunk(chunkptr, MaxNumFaces);

    // Dynamically resize image-based auxiliary buffers during training.
	size_t img_chunk_size = required<ImageState_cv>(src_width * src_height);
	char* img_chunkptr = imageBuffer_cv(img_chunk_size);
	ImageState_cv imgState_cv = ImageState_cv::fromChunk(img_chunkptr, src_width * src_height);

    // Utilize depth to build mesh.
    // printf("Build mesh...\n");
	CHECK_CUDA(BUILDING::buildmesh::mesh(
		ref_width, ref_height,
        src_width, src_height,
		ref_focal_x, ref_focal_y,
        src_focal_x, src_focal_y,
		ref_principal_x, ref_principal_y,
        src_principal_x, src_principal_y,
		viewmatrix,
        transmatrix,
        depth,
        ref_mask,
        vertices_ref,
        vertices,
        points_image,
        rays,
        faces,
        faceIndicator,
        MaxNumVertices,
        MaxNumFaces), debug);

    // Process every face, touched pixel numbers and ranges.
    // printf("Mesh postprocess...\n");
    CHECK_CUDA(BUILDING::buildmesh::postprocess(
        src_width, src_height,
        vertices_ref,
        vertices,
        points_image,
        faces,
        faceIndicator,
        faceNormal_ref,
        faceNormal,
        MaxNumVertices,
        MaxNumFaces,
        geomState_cv.pixels_touched,
        geomState_cv.bounding_box,
        geomState_cv.vertices2D,
        geomState_cv.verticesZ,
        geomState_cv.vertices_src), debug);

    // Compute prefix sum over full list of touched pixels counts by faces.
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8].
    // printf("Compute inclusive sum...\n");
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState_cv.scanning_space, geomState_cv.scan_size, geomState_cv.pixels_touched, geomState_cv.point_offsets, MaxNumFaces), debug);

    // Retrieve total number of face instances to launch and resize aux buffers.
	CHECK_CUDA(cudaMemcpy(&num_projected, geomState_cv.point_offsets + MaxNumFaces - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

    size_t binning_chunk_size = required<BinningState_cv>(num_projected);
	char* binning_chunkptr = binningBuffer_cv(binning_chunk_size);
	BinningState_cv binningState_cv = BinningState_cv::fromChunk(binning_chunkptr, num_projected);

    // For each instance to be projected, produce adequate [ pixel | pixels_touched ] key .
	// and corresponding dublicated Gaussian indices to be sorted.
    // printf("Duplicate with keys...\n");
	duplicateWithKeys_cv << <(MaxNumFaces + 255) / 256, 256 >> > (
		MaxNumFaces,
        src_width, src_height,
        geomState_cv.point_offsets,
        binningState_cv.face_list_keys_unsorted,
        binningState_cv.face_list_unsorted,
        geomState_cv.pixels_touched,
        geomState_cv.bounding_box,
        geomState_cv.vertices2D);
	CHECK_CUDA(, debug);

    // printf("Get higher msb...\n");
    int bit = getHigherMsb_cv(src_width * src_height);

    // Sort complete list of (duplicated) face indices by keys.
    // printf("Sort pairs...\n");
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState_cv.list_sorting_space,
		binningState_cv.sorting_size,
		binningState_cv.face_list_keys_unsorted, binningState_cv.face_list_keys,
		binningState_cv.face_list_unsorted, binningState_cv.face_list,
		num_projected, 0, 32 + bit), debug); 

    // Alocate memory for imageState.ranges, to save start and end bit in face_list.
    // printf("CUDA memory set...\n");
    CHECK_CUDA(cudaMemset(imgState_cv.ranges, 0, src_width * src_height * sizeof(uint2)), debug);

    // Identify start and end of per-tile workloads in sorted list.
    // printf("Identify face ranges...\n");
	if (num_projected > 0)
		identifyFaceRanges << <(num_projected + 255) / 256, 256 >> > (
			num_projected,
			binningState_cv.face_list_keys,
			imgState_cv.ranges,
            pixel_label);
	CHECK_CUDA(, debug);

    // Project depth map by mesh.
    // printf("Project depth map...\n");
    CHECK_CUDA(PROJECTING::projectdepth::depth(
        MaxNumFaces,
        MaxNumPixels,
        geomState_cv.vertices2D,
        geomState_cv.verticesZ,
		imgState_cv.ranges,
        imgState_cv.interp_id,
        imgState_cv.interp_weight,
        imgState_cv.interp_marker,
		binningState_cv.face_list,
        src_width,
        src_height,
        src_focal_x,
        src_focal_y,
        src_principal_x,
        src_principal_y,
        transmatrix,
        faces,
        pixel_label,
        faceNormal,
        depthmap,
        normalmap,
        src_mask), debug);
        
}
