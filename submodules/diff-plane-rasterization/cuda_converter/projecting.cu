
#include "projecting.h"
#include "auxiliary.h"
#include <cmath>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

template <uint32_t CHANNELS, uint32_t MAPS>
__global__ void depthCUDA(
    const int MaxNumFaces,
    const int MaxNumPixels,
    const float* __restrict__ vertices2D,
    const float* __restrict__ verticesZ,
    const uint2* __restrict__ ranges,
    uint32_t* __restrict__ interp_id,
    float* __restrict__ interp_weight,
    uint32_t* __restrict__ interp_marker,
	const uint32_t* __restrict__ face_list,
    const int src_width,
    const int src_height,
    const float src_focal_x,
    const float src_focal_y,
    const float src_principal_x,
    const float src_principal_y,
    const float* __restrict__ viewmatrix,
    const int* __restrict__ faces,
    const int* __restrict__ pixel_label,
    const float* __restrict__ faceNormal,
    float* __restrict__ depthmap,
    float* __restrict__ normalmap,
    const int* __restrict__ mask)
{   
    auto idx = cg::this_grid().thread_rank();
	if (idx >= MaxNumPixels)
		return;
    interp_marker[idx] = 0;

    // Check if this thread is associated with a valid pixel or outside.
    bool in_mask = mask[idx] > 0;
    bool in_project = pixel_label[idx] > 0;

    const uint2 pix = { (int)idx % src_width, (int)idx / src_width };
    const float2 pixf = { (float)pix.x, (float)pix.y };

    if (in_mask && in_project)
    {
        // Load start/end range of IDs to process in bit sorted list.
        uint2 range = ranges[idx];
        int toDo = range.y - range.x;
        if (toDo > 0)
        {   
            float tmp_seta = 0.f;
            for (int j = 0; j < toDo; j++)
            {   
                int coll_id = face_list[range.x + j];

                const float3 normal = { faceNormal[0 * MaxNumFaces + coll_id], faceNormal[1 * MaxNumFaces + coll_id], faceNormal[2 * MaxNumFaces + coll_id] };
                const float length_normal = sqrt(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
                const float3 dir_normal = { normal.x / length_normal, normal.y / length_normal, normal.z / length_normal };
                const float3 ray = { (pixf.x - src_principal_x) / src_focal_x, (pixf.y - src_principal_y) / src_focal_y, 1 };
                const float length_ray = sqrt(ray.x * ray.x + ray.y * ray.y + ray.z * ray.z);
                const float3 dir_ray = { ray.x / length_ray, ray.y / length_ray, ray.z / length_ray };
                float seta = std::acos(dir_normal.x * dir_ray.x + dir_normal.y * dir_ray.y + dir_normal.z * dir_ray.z);

                tmp_seta = seta;

                float2 a = { vertices2D[6 * coll_id + 0], vertices2D[6 * coll_id + 1] };
                float2 b = { vertices2D[6 * coll_id + 2], vertices2D[6 * coll_id + 3] };
                float2 c = { vertices2D[6 * coll_id + 4], vertices2D[6 * coll_id + 5] };
                float d_a = verticesZ[3 * coll_id + 0];
                float d_b = verticesZ[3 * coll_id + 1];
                float d_c = verticesZ[3 * coll_id + 2];

                float xp = pix.x, yp = pix.y;
                float xa = a.x, ya = a.y;
                float xb = b.x, yb = b.y;
                float xc = c.x, yc = c.y;
                float gamma = ((xb - xa) * (yp - ya) - (xp - xa) * (yb - ya)) / 
                            ((xb - xa) * (yc - ya) - (xc - xa) * (yb - ya));
                float beta = (xp - xa - gamma * (xc - xa)) / (xb - xa);
                float alpha = 1.0f - beta - gamma;

                float d_interp = alpha * d_a + beta * d_b + gamma * d_c;

                if (d_interp > 0 )
                {   
                    if (depthmap[idx] == 0)
                    {
                        depthmap[idx] = d_interp;
                        normalmap[0 * MaxNumPixels + idx] = normal.x;
                        normalmap[1 * MaxNumPixels + idx] = normal.y;
                        normalmap[2 * MaxNumPixels + idx] = normal.z;
                        interp_marker[idx] = 1;
                        interp_id[idx] = coll_id;
                        tmp_seta = seta;
                        interp_weight[3 * idx + 0] = alpha;
                        interp_weight[3 * idx + 1] = beta;
                        interp_weight[3 * idx + 2] = gamma;
                    }
                    else
                    {   
                        if (d_interp < depthmap[idx]) {
                            depthmap[idx] = d_interp;
                            normalmap[0 * MaxNumPixels + idx] = normal.x;
                            normalmap[1 * MaxNumPixels + idx] = normal.y;
                            normalmap[2 * MaxNumPixels + idx] = normal.z;
                            interp_marker[idx] = 1;
                            interp_id[idx] = coll_id;
                            tmp_seta = seta;
                            interp_weight[3 * idx + 0] = alpha;
                            interp_weight[3 * idx + 1] = beta;
                            interp_weight[3 * idx + 2] = gamma;
                        } 
                    }
                }
            }
            if (tmp_seta <= 1.57f)
                interp_marker[idx] = 0;
        }
    }
}

template <uint32_t CHANNELS, uint32_t MAPS>
__global__ void samplegridCUDA(
    const int ref_width,
    const int ref_height,
    const int MaxNumVertices,
    const int sample_num,
    const int patch_size,
    float* __restrict__ points_image,
    float* __restrict__ ref_grid,
    float* __restrict__ src_grid,
    const int* __restrict__ offsets,
    const int* __restrict__ sample_mask,
    const int* __restrict__ sample_mask_prefix_sum)
{
    auto idx = cg::this_grid().thread_rank();
	if (idx >= MaxNumVertices)
		return;
    const bool is_sample = sample_mask[idx] > 0;
    const int total_patch_size = (2 * patch_size + 1) * (2 * patch_size + 1);
    const int sample_idx = sample_mask_prefix_sum[idx];
    const int2 ref_pix = { (int)(idx % ref_width), (int)(idx / ref_width) };
    if (is_sample) {
        if ((ref_pix.x - patch_size) < 0 || (ref_pix.x + patch_size) >= ref_width || (ref_pix.y - patch_size) < 0 || (ref_pix.y + patch_size) >= ref_height)
            return;
        int tmp_idx = sample_idx * total_patch_size * 2;
        for (int i = 0; i < total_patch_size; i++) {
            int grid_x = ref_pix.x + offsets[2 * i];
            int grid_y = ref_pix.y + offsets[2 * i + 1];
            ref_grid[tmp_idx + 2 * i] = (float)grid_x;
            ref_grid[tmp_idx + 2 * i + 1] = (float)grid_y;
            src_grid[tmp_idx + 2 * i] = points_image[grid_y * ref_width + grid_x];
            src_grid[tmp_idx + 2 * i + 1] = points_image[MaxNumVertices + grid_y * ref_width + grid_x];
        }
    }
}

namespace PROJECTING
{
    void projectdepth::depth(
            const int MaxNumFaces,
            const int MaxNumPixels,
            const float* vertices2D,
            const float* verticesZ,
            const uint2* ranges,
            uint32_t* interp_id,
            float* interp_weight,
            uint32_t* interp_marker,
            const uint32_t* face_list,
            const int src_width,
            const int src_height,
            const float src_focal_x,
            const float src_focal_y,
            const float src_principal_x,
            const float src_principal_y,
            const float* viewmatrix,
            const int* faces,
            const int* pixel_label,
            const float* faceNormal,
            float* depthmap,
            float* normalmap,
            const int* mask)
        {
            depthCUDA<NUM_CHANNELS, NUM_MAPS> << <(MaxNumPixels + 255) / 256, 256>> > (
                MaxNumFaces,
                MaxNumPixels,
                vertices2D,
                verticesZ,
                ranges,
                interp_id,
                interp_weight,
                interp_marker,
                face_list,
                src_width,
                src_height,
                src_focal_x,
                src_focal_y,
                src_principal_x,
                src_principal_y,
                viewmatrix,
                faces,
                pixel_label,
                faceNormal,
                depthmap,
                normalmap,
                mask);
        }

    void projectdepth::samplegrid(
        const int ref_width,
        const int ref_height,
        const int MaxNumVertices,
        const int sample_num,
        const int patch_size,
        float* points_image,
        float* ref_grid,
        float* src_grid,
        const int* offsets,
        const int* sample_mask,
        const int* sample_mask_prefix_sum)
    {
        samplegridCUDA<NUM_CHANNELS, NUM_MAPS> << <(MaxNumVertices + 255) / 256, 256>> > (
            ref_width,
            ref_height,
            MaxNumVertices,
            sample_num,
            patch_size,
            points_image,
            ref_grid,
            src_grid,
            offsets,
            sample_mask,
            sample_mask_prefix_sum);
    }
}

