
#ifndef CUDA_CONVERTER_BUILDING_H_INCLUDED
#define CUDA_CONVERTER_BUILDING_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <cmath>

namespace BUILDING
{
    class buildmesh
    {
    public:
        static void mesh(
            const int ref_width,
            const int ref_height,
            const int src_width,
            const int src_height,
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
            const int* mask,
            float* vertices_ref,
            float* vertices,
            float* points_image,
            float* rays,
            int* faces,
            int* faceIndicator,
            const int MaxNumVertices,
            const int MaxNumFaces);

        static void postprocess(
            const int width,
            const int height,
            float* vertices_ref,
            float* vertices,
            float* points_image,
            int* faces,
            int* faceIndicator,
            float* faceNormal_ref,
            float* faceNormal,
            const int MaxNumVertices,
            const int MaxNumFaces,
            uint32_t* pixels_touched,
            uint32_t* bounding_box,
            float* vertices2D,
            float* verticesZ,
            float* vertices_src);
    };
    
}

#endif