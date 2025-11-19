
#pragma once

#include <iostream>
#include <vector>
#include "converter.h"
#include <cuda_runtime_api.h>

namespace CudaConverter
{   
    template <typename T>
	static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment)
	{
		std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
		ptr = reinterpret_cast<T*>(offset);
		chunk = reinterpret_cast<char*>(ptr + count);
	}

    struct GeometryState_cv
	{
		uint32_t* pixels_touched;
        uint32_t* bounding_box;
        float* vertices2D;
        float* verticesZ;
		float* vertices_src;
        size_t scan_size;
		char* scanning_space;
		uint32_t* point_offsets;

		static GeometryState_cv fromChunk(char*& chunk, size_t P);
	};

    struct BinningState_cv
	{
		size_t sorting_size;
		uint64_t* face_list_keys_unsorted;
		uint64_t* face_list_keys;
		uint32_t* face_list_unsorted;
		uint32_t* face_list;
		char* list_sorting_space;

		static BinningState_cv fromChunk(char*& chunk, size_t P);
	};

    struct ImageState_cv
	{
		uint2* ranges;
		uint32_t* interp_id;
		float* interp_weight;
		uint32_t* interp_marker;

		static ImageState_cv fromChunk(char*& chunk, size_t N);
	};

    template<typename T> 
	size_t required(size_t P)
	{
		char* size = nullptr;
		T::fromChunk(size, P);
		return ((size_t)size) + 128;
	}
}