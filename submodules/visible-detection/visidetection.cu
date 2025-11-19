
#include <torch/extension.h>
#include <chrono>
#include <cmath>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

__device__ float GetNormalAngle(const float3& ref_normal, const float3& src_normal, bool need_normalize) {
    float dot_val = ref_normal.x * src_normal.x + ref_normal.y * src_normal.y + ref_normal.z * src_normal.z;
    float cos_theta = dot_val;
    if (need_normalize) {
        float ref_norm = sqrtf(ref_normal.x * ref_normal.x + ref_normal.y * ref_normal.y + ref_normal.z * ref_normal.z);
        float src_norm = sqrtf(src_normal.x * src_normal.x + src_normal.y * src_normal.y + src_normal.z * src_normal.z);
        if (ref_norm < 1e-6f || src_norm < 1e-6f)
            return 0.0f;
        cos_theta = dot_val / (ref_norm * src_norm);
    }
    cos_theta = fmaxf(-1.0f, fminf(1.0f, cos_theta));

    return acosf(cos_theta) * 180 / 3.14159;
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

__global__ void excute_detect_cuda(
    const int ref_height,
    const int ref_width,
    const int src_height,
    const int src_width,
    const int num_faces,
    const int num_pixels,
    float* __restrict__ depths,
    float* __restrict__ face_normals,
    float* __restrict__ viewmatrix,
    float* __restrict__ ref_K,
    float* __restrict__ transmatrix,
    float* __restrict__ src_K,
    bool* __restrict__ prior_mask,
    bool* __restrict__ project_mask,
    bool* __restrict__ visible_mask
) {
    auto idx = cg::this_grid().thread_rank();
	if (idx >= num_pixels)
		return;

    if (!prior_mask[idx])
        return;

    int2 p = make_int2((int)(idx % ref_width), (int)(idx / ref_width));
    float2 pixf = { (float)p.x, (float)p.y };

    if (p.x == 0 || p.x == ref_width - 1 || p.y == 0 || p.y == ref_height - 1)
        return;

    float* first_row = face_normals + (2 * (ref_width - 1) * (p.y - 1) + 2 * (p.x - 1)) * 3;
    float* second_row = face_normals + (2 * (ref_width - 1) * p.y + 2 * (p.x - 1) + 1) * 3;

    float3* first_vecs  = reinterpret_cast<float3*>(first_row);
    float3* second_vecs = reinterpret_cast<float3*>(second_row);

    float3 merged[6];
    merged[0] = first_vecs[0];
    merged[1] = first_vecs[1];
    merged[2] = first_vecs[2];
    merged[3] = second_vecs[0];
    merged[4] = second_vecs[1];
    merged[5] = second_vecs[2];

    float3 vertex_normal;
    int cout = 0;
    for (int i = 0; i < 6; i++) {
        float3 normal = merged[i];
        float length = sqrtf(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
        if (length > 0) {
            normal.x /= length;
            normal.y /= length;
            normal.z /= length;
            vertex_normal.x += normal.x;
            vertex_normal.y += normal.y;
            vertex_normal.z += normal.z;
            cout++;
        }
    }

    bool vertex_valid = false;
    float ref_fx = ref_K[0];
    float ref_fy = ref_K[4];
    float ref_cx = ref_K[2];
    float ref_cy = ref_K[5];
    if (cout > 0) {
        float length = sqrtf(vertex_normal.x * vertex_normal.x + vertex_normal.y * vertex_normal.y + vertex_normal.z * vertex_normal.z);
        vertex_normal.x /= length;
        vertex_normal.y /= length;
        vertex_normal.z /= length;

        float3 ray = { (pixf.x - ref_cx) / ref_fx, (pixf.y - ref_cy) / ref_fy, 1 };
        float ray_length = sqrtf(ray.x * ray.x + ray.y * ray.y + ray.z * ray.z);
        ray.x /= ray_length;
        ray.y /= ray_length;
        ray.z /= ray_length;
        float3 inv_ray = { -ray.x, -ray.y, -ray.z };

        float angle = GetNormalAngle(inv_ray, vertex_normal, false);
        if (angle < 85.0f) {
            vertex_valid = true;
        }
    }

    if (vertex_valid) {
        float3 ray = { (pixf.x - ref_cx) / ref_fx, (pixf.y - ref_cy) / ref_fy, 1 };
        float3 point_ref_cam = { ray.x * depths[idx], ray.y * depths[idx], depths[idx] };
        float3 point_world = invTransformPoint4x3(point_ref_cam, viewmatrix);
        float3 point_src_cam = transformPoint4x3(point_world, transmatrix);

        if (point_src_cam.z < 0.02)
            return;

        float src_fx = src_K[0];
        float src_fy = src_K[4];
        float src_cx = src_K[2];
        float src_cy = src_K[5];
        float2 point_image = { src_fx * point_src_cam.x / point_src_cam.z + src_cx,
                               src_fy * point_src_cam.y / point_src_cam.z + src_cy };
        if (point_image.x > 0 && point_image.x < src_width - 1 && point_image.y > 0 && point_image.y < src_height - 1) {
            int tempPoints[4];
            tempPoints[0] = (int)(point_image.x) + int(point_image.y) * src_width;
            tempPoints[1] = (int)(point_image.x + 1) + int(point_image.y) * src_width;
            tempPoints[2] = (int)(point_image.x) + int(point_image.y + 1) * src_width;
            tempPoints[3] = (int)(point_image.x + 1) + int(point_image.y + 1) * src_width;
            if (project_mask[tempPoints[0]] && project_mask[tempPoints[1]] && project_mask[tempPoints[2]] && project_mask[tempPoints[3]]) {
                visible_mask[idx] = true;
            }
        }
    }
}


torch::Tensor excute_detect(torch::Tensor depth, torch::Tensor faceNormal, torch::Tensor ref_pose, torch::Tensor ref_K, torch::Tensor src_pose, torch::Tensor src_K, torch::Tensor prior_mask, torch::Tensor project_mask)
{
    cudaSetDevice(0);

    const int ref_height = prior_mask.size(1);
    const int ref_width = prior_mask.size(2);
    const int src_height = project_mask.size(1);
    const int src_width = project_mask.size(2);
    const int NumFaces = faceNormal.size(1);
    const int NumPixels = ref_height * ref_width;
    auto bool_opts = prior_mask.options().dtype(torch::kBool);

    torch::Tensor visible_mask = torch::full({1, ref_height, ref_width}, false, bool_opts);

    excute_detect_cuda<< <(NumPixels + 255) / 256, 256>> >(
        ref_height,
        ref_width,
        src_height,
        src_width,
        NumFaces,
        NumPixels,
        depth.contiguous().data<float>(),
        faceNormal.contiguous().data<float>(),
        ref_pose.contiguous().data<float>(),
        ref_K.contiguous().data<float>(),
        src_pose.contiguous().data<float>(),
        src_K.contiguous().data<float>(),
        prior_mask.contiguous().data<bool>(),
        project_mask.contiguous().data<bool>(),
        visible_mask.contiguous().data<bool>()
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    cudaDeviceSynchronize();

    return visible_mask;
}