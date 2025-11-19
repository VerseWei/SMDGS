
#include "building.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cmath>
namespace cg = cooperative_groups;

// Main build mesh method, project every pixel of depth to 3D space,
// build mesh by all 3D points and record,
// vertices(CHANNELS, MaxNumVertices) are 3D world coordinates,
// faces(MAPS, MaxNumFaces) are vertices ids,
// faceIndicator(MaxNumFaces, ) are faces indicators.
template <uint32_t CHANNELS, uint32_t MAPS>
__global__ void meshCUDA(
    const int W,
    const int H,
    const int WW,
    const int HH,
    const float ref_focal_x,
    const float ref_focal_y,
    const float src_focal_x,
    const float src_focal_y,
    const float ref_principal_x,
    const float ref_principal_y,
    const float src_principal_x,
    const float src_principal_y,
    const float* __restrict__ viewmatrix,
    const float* __restrict__ transmatrix,
    const float* __restrict__ depth,
    const int* __restrict__ mask,
    float* __restrict__ vertices_ref,
    float* __restrict__ vertices,
    float* __restrict__ points_image,
    float* __restrict__ rays,
    int* __restrict__ faces,
    int* __restrict__ faceIndicator,
    const int MaxNumVertices,
    const int MaxNumFaces)
{

    // Identify current block and associated min/max pixel range.
    auto idx = cg::this_grid().thread_rank();
	if (idx >= MaxNumVertices)
		return;
    
    const uint2 pix = { (int)idx % W, (int)idx / W };
    const float2 pixf = { (float)pix.x, (float)pix.y };
    const float3 ray = { (pixf.x - ref_principal_x) / ref_focal_x, (pixf.y - ref_principal_y) / ref_focal_y, 1 };

    // Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W && pix.y < H;
    bool spread_face = pix.x < (W -1) && pix.y < (H - 1);

    // Project depth to 3D space.
    bool in_mask = mask[idx] > 0;
    float p_view = depth[idx];
    bool outlier = (p_view <= 0.0f);
    if (inside)
    {   
        float3 point_ref_cam = { ray.x * p_view,
                                 ray.y * p_view,
                                 p_view };
        float3 point_world = invTransformPoint4x3(point_ref_cam, viewmatrix);
        float3 point_src_cam = transformPoint4x3(point_world, transmatrix);
        vertices_ref[0 * MaxNumVertices + idx] = point_world.x;                // 3D point in reference camera space
        vertices_ref[1 * MaxNumVertices + idx] = point_world.y;
        vertices_ref[2 * MaxNumVertices + idx] = point_world.z;
        vertices[0 * MaxNumVertices + idx] = point_src_cam.x;                // 3D point in nearest camera space
        vertices[1 * MaxNumVertices + idx] = point_src_cam.y;
        vertices[2 * MaxNumVertices + idx] = point_src_cam.z;
        float2 point_image = { src_focal_x * point_src_cam.x / (point_src_cam.z + 0.0000001f) + src_principal_x,
                               src_focal_y * point_src_cam.y / (point_src_cam.z + 0.0000001f) + src_principal_y };
        points_image[idx] = point_image.x;                                   // 2D point in nearest screen space
        points_image[MaxNumVertices + idx] = point_image.y;
        rays[0 * MaxNumVertices + idx] = ray.x;
        rays[1 * MaxNumVertices + idx] = ray.y;
        rays[2 * MaxNumVertices + idx] = ray.z;
    }

    // Build faces by vertices.
    const uint2 face_id = { 2 * ((W - 1) * pix.y + pix.x), 2 * ((W - 1) * pix.y + pix.x) + 1 };
    if (inside && spread_face)
    {
        faces[0 * MaxNumFaces + face_id.x] = idx + 1;
        faces[1 * MaxNumFaces + face_id.x] = idx;
        faces[2 * MaxNumFaces + face_id.x] = idx + W;
        faces[0 * MaxNumFaces + face_id.y] = idx + W + 1;
        faces[1 * MaxNumFaces + face_id.y] = idx + 1;
        faces[2 * MaxNumFaces + face_id.y] = idx + W;
    }

    // Remove outlier, and only mark valid faces.
    if (!in_mask || outlier)
    {
        if (pix.x > 0 && pix.x < (W - 1) && pix.y > 0 && pix.y < (H - 1)) {
            faceIndicator[face_id.x] = 0;
            faceIndicator[face_id.x - 1] = 0;
            faceIndicator[face_id.x - 2] = 0;
            faceIndicator[face_id.x - (2 * (W - 1))] = 0;
            faceIndicator[face_id.x - (2 * (W - 1)) - 1] = 0;
            faceIndicator[face_id.x - (2 * (W - 1)) + 1] = 0;
        } else if (pix.x == 0 && pix.y == 0) {
            faceIndicator[face_id.x] = 0;
        } else if (pix.x == W - 1 && pix.y == 0) {
            faceIndicator[face_id.x - 1] = 0;
            faceIndicator[face_id.x - 2] = 0;
        } else if (pix.x == 0 && pix.y == H - 1) {
            faceIndicator[face_id.x - (2 * (W - 1))] = 0;
            faceIndicator[face_id.x - (2 * (W - 1)) + 1] = 0;
        } else if (pix.x == W - 1 && pix.y == H - 1) {
            faceIndicator[face_id.x - (2 * (W - 1)) - 1] = 0;
        } else if (pix.x == 0) {
            faceIndicator[face_id.x] = 0;
            faceIndicator[face_id.x - (2 * (W - 1))] = 0;
            faceIndicator[face_id.x - (2 * (W - 1)) + 1] = 0;
        } else if (pix.x == W - 1) {
            faceIndicator[face_id.x - 1] = 0;
            faceIndicator[face_id.x - 2] = 0;
            faceIndicator[face_id.x - (2 * (W - 1)) - 1] = 0;
        } else if (pix.y == 0) {
            faceIndicator[face_id.x] = 0;
            faceIndicator[face_id.x - 1] = 0;
            faceIndicator[face_id.x - 2] = 0;
        } else if (pix.y == H -1) {
            faceIndicator[face_id.x - (2 * (W - 1))] = 0;
            faceIndicator[face_id.x - (2 * (W - 1)) - 1] = 0;
            faceIndicator[face_id.x - (2 * (W - 1)) + 1] = 0;
        }
    }
}
    


template <uint32_t CHANNELS, uint32_t MAPS>
__global__ void postprocessCUDA(
    const int W,
    const int H,
    float* __restrict__ vertices_ref,
    float* __restrict__ vertices,
    float* __restrict__ points_image,
    int* __restrict__ faces,
    int* __restrict__ faceIndicator,
    float* __restrict__ faceNormal_ref,
    float* __restrict__ faceNormal,
    const int MaxNumVertices,
    const int MaxNumFaces,
    uint32_t* __restrict__ pixels_touched,
    uint32_t* __restrict__ bounding_box,
    float* __restrict__ vertices2D,
    float* __restrict__ verticesZ,
    float* __restrict__ vertices_src)
{
    auto idx = cg::this_grid().thread_rank();
	if (idx >= MaxNumFaces)
		return;

    // Compute face normal in reference camera.
    float3 vertex0_ref = { vertices_ref[faces[idx]], vertices_ref[MaxNumVertices + faces[idx]], vertices_ref[2 * MaxNumVertices + faces[idx]] };
    float3 vertex1_ref = { vertices_ref[faces[MaxNumFaces + idx]], vertices_ref[MaxNumVertices + faces[MaxNumFaces + idx]], vertices_ref[2 * MaxNumVertices + faces[MaxNumFaces + idx]] };
    float3 vertex2_ref = { vertices_ref[faces[2 * MaxNumFaces + idx]], vertices_ref[MaxNumVertices + faces[2 * MaxNumFaces + idx]], vertices_ref[2 * MaxNumVertices + faces[2 * MaxNumFaces + idx]] };
    float3 normal_ref = ComputeFaceNormal(vertex0_ref, vertex1_ref, vertex2_ref);
    faceNormal_ref[0 * MaxNumFaces + idx] = normal_ref.x;
    faceNormal_ref[1 * MaxNumFaces + idx] = normal_ref.y;
    faceNormal_ref[2 * MaxNumFaces + idx] = normal_ref.z;


	// Initialize touched pixels to 0. If this isn't changed,
	// this face will not be processed further.
	pixels_touched[idx] = 0;

    // Invalid face, return.
    if (faceIndicator[idx] == 0)
        return;
    
    // Transform three vertices to camera coordinates.
    float3 vertex0_cam = { vertices[faces[idx]], vertices[MaxNumVertices + faces[idx]], vertices[2 * MaxNumVertices + faces[idx]] };
    float3 vertex1_cam = { vertices[faces[MaxNumFaces + idx]], vertices[MaxNumVertices + faces[MaxNumFaces + idx]], vertices[2 * MaxNumVertices + faces[MaxNumFaces + idx]] };
    float3 vertex2_cam = { vertices[faces[2 * MaxNumFaces + idx]], vertices[MaxNumVertices + faces[2 * MaxNumFaces + idx]], vertices[2 * MaxNumVertices + faces[2 * MaxNumFaces + idx]] };

    // Transform three vertices to screen space.
    float2 point0_image = { points_image[faces[idx]], points_image[MaxNumVertices + faces[idx]] };
    float2 point1_image = { points_image[faces[MaxNumFaces + idx]], points_image[MaxNumVertices + faces[MaxNumFaces + idx]] };
    float2 point2_image = { points_image[faces[2 * MaxNumFaces + idx]], points_image[MaxNumVertices + faces[2 * MaxNumFaces + idx]] };

    // Any vertex is not inside frustum, renturn.
    if (vertex0_cam.z < 0.02f || vertex1_cam.z < 0.02f || vertex2_cam.z < 0.02f)
        return;
    
    // Any vertex is not inside pixel plane, return.
    bool in_screen0 = (point0_image.x >= 0 && point0_image.x < W && point0_image.y >= 0 && point0_image.y < H);
    bool in_screen1 = (point1_image.x >= 0 && point1_image.x < W && point1_image.y >= 0 && point1_image.y < H);
    bool in_screen2 = (point2_image.x >= 0 && point2_image.x < W && point2_image.y >= 0 && point2_image.y < H);

    if (!in_screen0 || !in_screen1 || !in_screen2)
        return;

    // Compute 2D bounding box of a face.
    float min_x = min(min(point0_image.x, point1_image.x), point2_image.x);
    float max_x = max(max(point0_image.x, point1_image.x), point2_image.x);
    float min_y = min(min(point0_image.y, point1_image.y), point2_image.y);
    float max_y = max(max(point0_image.y, point1_image.y), point2_image.y);

    int minX = std::ceil(min_x), minY = std::ceil(min_y);
    int maxX = std::ceil(max_x), maxY = std::ceil(max_y);
    if (minX == maxX || minY == maxY)
        return;

    // Add pixels in triangle to pixels_touched.
    for (int row = minY; row < maxY; row++)
    {
        for (int col = minX; col < maxX; col++)
        {
            float2 pixel_xy = { (float)col, (float)row };
            bool inTriangle = PointInTriangle(point0_image, point1_image, point2_image, pixel_xy);
            if (inTriangle) {
                pixels_touched[idx]++;
            }
        }
    }

    // Record aabb if this face touches at least one pixel. 
    if (pixels_touched[idx] > 0) {
        bounding_box[4 * idx + 0] = minX;
        bounding_box[4 * idx + 1] = minY;
        bounding_box[4 * idx + 2] = maxX;
        bounding_box[4 * idx + 3] = maxY;
        vertices2D[6 * idx + 0] = point0_image.x;
        vertices2D[6 * idx + 1] = point0_image.y;
        vertices2D[6 * idx + 2] = point1_image.x;
        vertices2D[6 * idx + 3] = point1_image.y;
        vertices2D[6 * idx + 4] = point2_image.x;
        vertices2D[6 * idx + 5] = point2_image.y;
        float3 normal = ComputeFaceNormal(vertex0_cam, vertex1_cam, vertex2_cam);
        faceNormal[0 * MaxNumFaces + idx] = normal.x;
        faceNormal[1 * MaxNumFaces + idx] = normal.y;
        faceNormal[2 * MaxNumFaces + idx] = normal.z;
        verticesZ[3 * idx + 0] = vertex0_cam.z;
        verticesZ[3 * idx + 1] = vertex1_cam.z;
        verticesZ[3 * idx + 2] = vertex2_cam.z;
        vertices_src[9 * idx + 0] = vertex0_cam.x;
        vertices_src[9 * idx + 1] = vertex0_cam.y;
        vertices_src[9 * idx + 2] = vertex0_cam.z;
        vertices_src[9 * idx + 3] = vertex1_cam.x;
        vertices_src[9 * idx + 4] = vertex1_cam.y;
        vertices_src[9 * idx + 5] = vertex1_cam.z;
        vertices_src[9 * idx + 6] = vertex2_cam.x;
        vertices_src[9 * idx + 7] = vertex2_cam.y;
        vertices_src[9 * idx + 8] = vertex2_cam.z;
    }

}

namespace BUILDING
{
    void buildmesh::mesh(
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
            const int MaxNumFaces)
        {   
            meshCUDA<NUM_CHANNELS, NUM_MAPS> << <(MaxNumVertices + 255) / 256, 256>> > (
                ref_width,
                ref_height,
                src_width,
                src_height,
                ref_focal_x,
                ref_focal_y,
                src_focal_x,
                src_focal_y,
                ref_principal_x,
                ref_principal_y,
                src_principal_x,
                src_principal_y,
                viewmatrix,
                transmatrix,
                depth,
                mask,
                vertices_ref,
                vertices,
                points_image,
                rays,
                faces,
                faceIndicator,
                MaxNumVertices,
                MaxNumFaces);
        }

    void buildmesh::postprocess(
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
            float* vertices_src)
        {
            postprocessCUDA<NUM_CHANNELS, NUM_MAPS> << <(MaxNumFaces + 255) / 256, 256>> > (
                width,
                height,
                vertices_ref,
                vertices,
                points_image,
                faces,
                faceIndicator,
                faceNormal_ref,
                faceNormal,
                MaxNumVertices,
                MaxNumFaces,
                pixels_touched,
                bounding_box,
                vertices2D,
                verticesZ,
                vertices_src);
        }
} // namespace BUILDING 

