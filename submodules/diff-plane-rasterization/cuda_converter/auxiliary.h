
#ifndef CUDA_CONVERTER_AUXILIARY_H_INCLUDED
#define CUDA_CONVERTER_AUXILIARY_H_INCLUDED

#include "config.h"
#include "stdio.h"
#include <cmath>

#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)

__forceinline__ __device__ float ndc2Pix(float v, int S)
{
	return ((v + 1.0) * S - 1.0) * 0.5;
}

__forceinline__ __device__ float3 invTransformPoint4x3(const float3& p, const float* matrix)
{   
    float3 transformed = {
        matrix[0] * (p.x - matrix[12]) + matrix[1] * (p.y - matrix[13]) + matrix[2] * (p.z - matrix[14]),
        matrix[4] * (p.x - matrix[12]) + matrix[5] * (p.y - matrix[13]) + matrix[6] * (p.z - matrix[14]),
        matrix[8] * (p.x - matrix[12]) + matrix[9] * (p.y - matrix[13]) + matrix[10] * (p.z - matrix[14]),
    };
	
    return transformed;

}

__forceinline__ __device__ float3 transformPoint4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],       // W2C * p 相机系下坐标
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
	};

	return transformed;
}

__forceinline__ __device__ float2 transformPoint4x4(const float3& p, const float* matrix, const int W, const int H)
{
	float4 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],        // W2C * p 相机系下坐标
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
		matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]
	};
    float p_w = 1.0f / (transformed.w + 0.0000001f);
	float3 p_proj = { transformed.x * p_w, transformed.y * p_w, transformed.z * p_w };    // ndc系
    float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };                  // 像素系

	return point_image;
}

__forceinline__ __device__ float4 transformPoint4x4NDC(const float3& p, const float* matrix)
{
	float4 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],        // W2C * p 相机系下坐标
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
		matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]
	};

	return transformed;
}

__forceinline__ __device__ float TriangleArea(const float2& A, const float2& B, const float2& C) 
{
    const float x1 = A.x;
    const float y1 = A.y;
    const float x2 = B.x;
    const float y2 = B.y;
    const float x3 = C.x;
    const float y3 = C.y;

    return fabs(((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0f));
}

__forceinline__ __device__ bool PointInTriangle(const float2& A, const float2& B, const float2& C, const float2& P)
{
    float eps = 1e-3;
    float Area_ABC = TriangleArea(A, B, C);
    float Area_PBC = TriangleArea(P, B, C);
    float Area_PAC = TriangleArea(A, P, C);
    float Area_PAB = TriangleArea(A, P, B);

    return fabs(Area_ABC - (Area_PAB + Area_PAC + Area_PBC)) < eps;
}

__forceinline__ __device__ float3 ComputeFaceNormal(const float3& A, const float3& B, const float3& C)
{
    float3 edge1 = { B.x - A.x, B.y - A.y, B.z - A.z };
    float3 edge2 = { C.x - B.x, C.y - B.y, C.z - B.z };

    float3 normal;
    normal.x = edge1.y * edge2.z - edge1.z * edge2.y;
    normal.y = edge1.z * edge2.x - edge1.x * edge2.z;
    normal.z = edge1.x * edge2.y - edge1.y * edge2.x;

    float length = sqrtf(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
    if (length > 0.0f) {
        normal.x /= length;
        normal.y /= length;
        normal.z /= length;
    }

    return normal;
}

#define CHECK_CUDA(A, debug) \
A; if(debug) { \
auto ret = cudaDeviceSynchronize(); \
if (ret != cudaSuccess) { \
std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); \
throw std::runtime_error(cudaGetErrorString(ret)); \
} \
}

#endif