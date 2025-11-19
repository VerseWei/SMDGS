
#ifndef CUDA_CONVERTER_H_INCLUDED
#define CUDA_CONVERTER_H_INCLUDED

#include <vector>
#include <functional>

namespace CudaConverter
{
	class Converter
	{
	public:

		static void depth2mesh2depth(
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
			const bool debug);
	};
};

#endif