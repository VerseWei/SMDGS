#include <torch/extension.h>
#include <vector>

torch::Tensor excute_detect(
  torch::Tensor depth,
  torch::Tensor faceNormal,
  torch::Tensor ref_pose,
  torch::Tensor ref_K,
  torch::Tensor src_pose,
  torch::Tensor src_K,
  torch::Tensor prior_mask,
  torch::Tensor project_mask
);

torch::Tensor visible_detection(
  torch::Tensor depth,
  torch::Tensor faceNormal,    // in ref_cam
  torch::Tensor ref_pose,
  torch::Tensor ref_K,
  torch::Tensor src_pose,
  torch::Tensor src_K,
  torch::Tensor prior_mask,
  torch::Tensor project_mask
) {
  return excute_detect(depth, faceNormal, ref_pose, ref_K, src_pose, src_K, prior_mask, project_mask);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // bundle adjustment kernels
  m.def("visible_detection", &visible_detection, "multi-view visible detection");
}