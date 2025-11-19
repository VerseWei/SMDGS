import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import List, Optional
from pathlib import Path
import os
from natsort import natsorted
import torch
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import pairwise
from sklearn.neighbors import KDTree


def get_filename_list(input_dir: Path, ends_with: Optional[str] = None) -> List:
    filenames = os.listdir(input_dir)
    if ends_with is not None:
        filenames = [input_dir / name for name in filenames if name.lower().endswith(ends_with)]
    else:
        filenames = [input_dir / name for name in filenames]
    filenames = natsorted(filenames)\
    
    return filenames

def fusion_sam_masks(sam_filenames, nonzero_depth_mask):
    nonzero_depth_mask = nonzero_depth_mask.permute(2, 0, 1)      # [1, H, W]
    nonzero_depth_mask = nonzero_depth_mask.cpu().numpy()
    masks = [cv2.imread(str(filename), cv2.IMREAD_GRAYSCALE) for filename in sam_filenames]
    masks = [(mask > 0).astype(np.uint8) for mask in masks]  #
    masks = np.stack(masks)
    mask = masks * nonzero_depth_mask
    mask = mask[np.any(mask != 0, axis=(1, 2))]

    return mask

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

# def divide_mask_into_compact_patches(masks_tensor, max_elements_per_patch, k_nearest=20):
#     M, H, W = masks_tensor.shape
#     patches = []
#     patch_labels = []
    
#     for label in range(M):
#         mask = masks_tensor[label]  # 当前标签的mask
#         y_indices, x_indices = np.nonzero(mask)  # 获取mask中所有非零像素的坐标
#         num_pixels = len(y_indices)
#         print(y_indices.shape, x_indices.shape)

#         # 进行聚类，按 max_elements_per_patch 划分 patch
#         for start in range(0, num_pixels, max_elements_per_patch):
#             end = min(start + max_elements_per_patch, num_pixels)
#             patch_coords = (y_indices[start:end], x_indices[start:end])

#             # 创建一个新的 patch mask
#             patch_mask = np.zeros((H, W), dtype=np.uint8)
#             patch_mask[patch_coords] = 1
#             patches.append(patch_mask)
#             patch_labels.append(label)

#     patches_tensor = np.array(patches)[:, np.newaxis]  # [N, H, W, 1]
    
#     # 找到每个 patch 的 k_nearest 最近邻
#     N = patches_tensor.shape[0]
#     neighbors = []

#     # 获取所有非零像素的坐标
#     # for patch in patches_tensor:
#     #     print(np.count_nonzero(patch), patch.shape)
#     patch_centers = [np.argwhere(patch.squeeze()) for patch in patches_tensor]
#     patch_centers_flat = [np.mean(center, axis=0) for center in patch_centers]  # 计算每个 patch 的中心点
#     patch_centers_flat = np.array(patch_centers_flat)  # 转换为数组


#     for label in range(M):
#         # 在当前 label 中构建 KD 树
#         label_indices = [i for i in range(N) if patch_labels[i] == label]  # 当前label包含的patch的索引
#         if len(label_indices) == 0:
#             continue
        
#         tree = KDTree(patch_centers_flat[label_indices])
#         print(label)
#         for i in label_indices:
#             current_patch_center = patch_centers_flat[i].reshape(1, -1)
#             distances, nearest_indices = tree.query(current_patch_center, k=min(k_nearest, len(label_indices)-1))
#             label_indices = np.array(label_indices)

#             # 合并邻近 patches
#             nearest_indices = label_indices[nearest_indices.flatten()]  # 映射回原索引
#             neighbor_mask = np.sum(patches_tensor[nearest_indices], axis=0) if nearest_indices.size > 0 else np.zeros((H, W, 1), dtype=np.uint8)
#             neighbors.append(neighbor_mask)

#     neibor_patches_tensor = np.array(neighbors)[:, np.newaxis]  # [N, H, W, 1]

#     return  neibor_patches_tensor, patches_tensor, N


def divide_mask_into_compact_patches(masks_tensor, max_elements_per_patch, k_nearest=20):
    N, height, width = masks_tensor.shape
    patch_size = int(np.sqrt(max_elements_per_patch))
    offsets = [(-patch_size / 2, 0), (patch_size / 2, 0), (0, -patch_size / 2), (0, patch_size / 2)]
    patch_num_h = height // patch_size if (height % patch_size < 0.5 * patch_size) else height // patch_size + 1
    patch_num_w = width // patch_size if (width % patch_size < 0.5 * patch_size) else width // patch_size + 1
    patch_num = patch_num_h * patch_num_w

    patch_centers = np.array([(y, x)
                              for y in range(patch_size // 2, height, patch_size)
                              for x in range(patch_size // 2, width, patch_size)])
    patch_count = np.zeros(N, dtype=int)
    mask_patch_centers = [[] for _ in range(N)]

    for mask_idx in range(N):
        mask = masks_tensor[mask_idx]
        for center in patch_centers:
            y, x = center[0], center[1]
            if any(
                0 <= int(y + dy) < height and 0 <= int(x + dx) < width and mask[int(y + dy), int(x + dx)] > 0
                for dy, dx in offsets
            ):
                patch_count[mask_idx] += 1
                mask_patch_centers[mask_idx].append(center)

    mask_patch_centers = [np.array(patch_list) for patch_list in mask_patch_centers if len(patch_list) > 0]
    patch_count = patch_count[patch_count > 0]
    # 如果没有有效补丁，避免 np.vstack 报错
    if mask_patch_centers:
        mask_patch_centers = np.vstack(mask_patch_centers)
    else:
        mask_patch_centers = np.empty((0, 2))  # 假设每个补丁中心是 (y, x)，所以维度为 2

    # 计算最大坐标
    patch_center_max_h = np.max(mask_patch_centers[:, 0]) if mask_patch_centers.size > 0 else 0
    patch_center_max_w = np.max(mask_patch_centers[:, 1]) if mask_patch_centers.size > 0 else 0

    # 计算补丁计数的累积和并生成起始索引
    indices = np.cumsum(patch_count)
    starts = np.insert(indices[:-1], 0, 0)

    # 处理切片
    sliced_centers = []
    for i in range(len(patch_count)):
        if i < len(mask_patch_centers):  # 确保索引不超出范围
            sliced_centers.append(mask_patch_centers[starts[i]:indices[i]])
        else:
            sliced_centers.append(np.empty((0, 2)))  # 如果没有对应的补丁中心，返回空数组

    mask_neibor_patch = np.zeros((patch_num, height, width), dtype=np.uint8)
    mask_unique_patch = np.zeros((patch_num, height, width), dtype=np.uint8)
    current_index = 0
    for i, count in enumerate(patch_count):
        centers = sliced_centers[i]
        k_nearest = min(k_nearest, len(centers) - 1)
        nbrs = NearestNeighbors(n_neighbors=k_nearest + 1, algorithm='auto').fit(centers)
        distances, indices = nbrs.kneighbors(centers)
        for j in range(count):
            combined_mask = np.zeros((height, width), dtype=np.uint8)
            unique_mask = np.zeros((height, width), dtype=np.uint8)
            for idx, neighbor_idx in enumerate(indices[j]):
                neighbor_center = centers[neighbor_idx]
                y_n_center, x_n_center = neighbor_center
                y_n_start = max(y_n_center - patch_size // 2, 0)
                x_n_start = max(x_n_center - patch_size // 2, 0)
                y_n_end = min(y_n_start + patch_size, height) if min(y_n_start + patch_size, height) != patch_center_max_h else height
                x_n_end = min(x_n_start + patch_size, width) if min(x_n_start + patch_size, width) != patch_center_max_w else width
                combined_mask[y_n_start: y_n_end, x_n_start: x_n_end] = 1
                if idx == 0:
                    unique_mask[y_n_start: y_n_end, x_n_start: x_n_end] = 1
            mask_neibor_patch[current_index + j] = combined_mask
            mask_unique_patch[current_index + j] = unique_mask
        current_index += count

    neibor_patches_tensor = torch.from_numpy(mask_neibor_patch)
    neibor_patches_tensor = neibor_patches_tensor.unsqueeze(-1)
    
    unique_patches_tensor = torch.from_numpy(mask_unique_patch)
    unique_patches_tensor = unique_patches_tensor.unsqueeze(-1)
    
    return neibor_patches_tensor, unique_patches_tensor, patch_num