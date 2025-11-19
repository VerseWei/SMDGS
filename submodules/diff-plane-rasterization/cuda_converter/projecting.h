
#ifndef CUDA_CONVERTER_PROJECTING_H_INCLUDED
#define CUDA_CONVERTER_PROJECTING_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <cmath>

namespace PROJECTING
{
    class projectdepth
    {
    public:
        static void depth(
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
            const int* mask);

        static void samplegrid(
            const int ref_width,
            const int ref_height,
            const int MaxNumvertices,
            const int sample_num,
            const int patch_size,
            float* points_image,
            float* ref_grid,
            float* src_grid,
            const int* offsets,
            const int* sample_mask,
            const int* sample_mask_prefix_sum);
    };
    
}

#endif