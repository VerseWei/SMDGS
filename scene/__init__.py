#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
import numpy as np
import torch
import torch.nn.functional as F
import cv2

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, prepare_data=True, resolution_scales=[1.0], task="train"):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.source_path = args.source_path

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "colmap/sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, args.create_mask, args.mask_type)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        self.cameras_extent, self.cameras_center = scene_info.nerf_normalization["radius"], scene_info.nerf_normalization["center"] # [1], [3]
        print(f"cameras_extent {self.cameras_extent}, cameras_center {self.cameras_center}")

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)   # Multi-res consistent random shuffling

        self.multi_view_num_filter = args.multi_view_num_filter
        self.multi_view_num_ncc = args.multi_view_num_ncc
        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

            if prepare_data:
                print("Reading multiview file for checking aligned depth")    
                self.read_multiview_pairs(resolution_scale)
                print("Computing probability maps for aligned depth")
                self.compute_aligned_depth_pmap(args, resolution_scale)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent, self.cameras_center)

        self.gaussians.config = torch.tensor(self.gaussians.config, dtype=torch.float32, device=args.data_device)

        if task == "train":
            self.train_camera_names = [cam.image_name for cam in self.train_cameras[resolution_scale]]
            print("Computing nearest train cameras for multiview check")
            self.compute_nearest_train_cams(args, resolution_scale)
        elif task == "render":
            with open(os.path.join(self.model_path, "multi_train_view_ncc.json"), 'r', encoding="utf-8") as file:
                lines = file.readlines()
            ref_names = set(json.loads(line)["ref_name"] for line in lines)
            self.train_cameras[resolution_scale] = [
                cam for cam in self.train_cameras[resolution_scale] if cam.image_name in ref_names
            ]

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
    def read_multiview_pairs(self, resolution_scale):
        pair_path = os.path.join(self.source_path, "colmap/pair.txt")
        with open(pair_path, 'r') as f:
            lines = f.readlines()
        multiview_data = []
        pair_start_idx = 1
        while pair_start_idx < len(lines):
            ref_image_info = lines[pair_start_idx].strip().split()
            ref_image_name = ref_image_info[1]
            ref_cam = [cam for cam in self.train_cameras[resolution_scale] if cam.image_name == ref_image_name]
            ref_cam = ref_cam[0]

            neighbor_line = lines[pair_start_idx + 1].strip().split()
            num_neighbors = int(neighbor_line[0])
            neighbors = []
            for i in range(min(num_neighbors, self.multi_view_num_filter)):
                neighbor_image_name = neighbor_line[2 + i * 3]
                neighbors.append(neighbor_image_name)
                ref_cam.nearest_names_filter.append(neighbor_image_name)
            multiview_data.append({
                "ref_name": ref_image_name,
                "nearest_name": neighbors
            })
            pair_start_idx += 2
        with open(os.path.join(self.model_path, "multi_view_filter.json"), 'w') as f:
            for entry in multiview_data:
                f.write(json.dumps(entry) + '\n')
    
    def reproject_with_depth(self, args, depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
        intrinsics_ref = intrinsics_ref.to(torch.float32)
        intrinsics_src = intrinsics_src.to(torch.float32)
        extrinsics_ref = extrinsics_ref
        extrinsics_src = extrinsics_src
        w_ref, h_ref = depth_ref.shape[1], depth_ref.shape[0]
        w_src, h_src = depth_src.shape[1], depth_src.shape[0]

        y_ref, x_ref = torch.meshgrid(torch.arange(0, h_ref, device=args.data_device), torch.arange(0, w_ref, device=args.data_device))  
        x_ref, y_ref = x_ref.reshape([-1]), y_ref.reshape([-1])  # p
        xyz_ref = torch.matmul(torch.linalg.inv(intrinsics_ref),
                            torch.vstack((x_ref, y_ref, torch.ones_like(x_ref))) * depth_ref.reshape([-1]))  
        xyz_2src = torch.matmul(torch.matmul(extrinsics_src, torch.inverse(extrinsics_ref)),
                            torch.vstack((xyz_ref, torch.ones_like(x_ref))))[:3] 
        K_xyz_src = torch.matmul(intrinsics_src, xyz_2src)
        xy_src = K_xyz_src[:2] / (K_xyz_src[2:3] + 1e-8) 
        x_src = xy_src[0].reshape([h_ref, w_ref])        
        y_src = xy_src[1].reshape([h_ref, w_ref])

        grid = torch.stack((x_src / (w_src - 1) * 2 - 1, y_src / (h_src - 1) * 2 - 1), dim=-1).unsqueeze(0)   # [1, h_ref, w_ref, 2]
        depth_src = depth_src.unsqueeze(0).unsqueeze(0)                                                       # [1, 1, h_src, w_src]
        sampled_depth_src = F.grid_sample(depth_src, grid, mode='bilinear', align_corners=False).squeeze()    # [1, 1, h_ref, w_ref]
        xyz_src = torch.matmul(torch.inverse(intrinsics_src),
                            torch.vstack((xy_src, torch.ones_like(x_ref))) * sampled_depth_src.reshape([-1])) 

        xyz_reprojected = torch.matmul(extrinsics_ref, torch.matmul(torch.inverse(extrinsics_src),
                                    torch.vstack((xyz_src, torch.ones_like(x_ref)))))[:3]  
        depth_reprojected = xyz_reprojected[2].reshape([h_ref, w_ref])          # [h_ref, w_ref]
        K_xyz_reprojected = torch.matmul(intrinsics_ref, xyz_reprojected)
        xy_reprojected = K_xyz_reprojected[:2] / (K_xyz_reprojected[2:3] + 1e-8)  
        x_reprojected = xy_reprojected[0].reshape([h_ref, w_ref])               # [h_ref, w_ref]
        y_reprojected = xy_reprojected[1].reshape([h_ref, w_ref])               # [h_ref, w_ref]

        valid_mask = (x_src > 0) & (x_src < w_src) & (y_src > 0) & (y_src < h_src) \
                    & (depth_ref > 0) & (depth_reprojected > 0)  

        return depth_reprojected, x_reprojected, y_reprojected, x_src, y_src, valid_mask

    def compute_nearest_train_cams(self, args, resolution_scale):
        self.world_view_transforms = []
        camera_centers = []
        center_rays = []
        for id, cur_cam in enumerate(self.train_cameras[resolution_scale]):
            self.world_view_transforms.append(cur_cam.world_view_transform)
            camera_centers.append(cur_cam.camera_center)
            R = torch.tensor(cur_cam.R).float().to(args.data_device)
            T = torch.tensor(cur_cam.T).float().to(args.data_device)
            center_ray = torch.tensor([0.0,0.0,1.0]).float().to(args.data_device)
            center_ray = center_ray @ R.transpose(-1,-2)
            center_rays.append(center_ray)
        self.world_view_transforms = torch.stack(self.world_view_transforms)
        camera_centers = torch.stack(camera_centers, dim=0)
        center_rays = torch.stack(center_rays, dim=0)
        center_rays = torch.nn.functional.normalize(center_rays, dim=-1)

        diss = torch.norm(camera_centers[:,None] - camera_centers[None], dim=-1).detach().cpu().numpy()  # 相机两两之间的距离
        tmp = torch.sum(center_rays[:,None] * center_rays[None], dim=-1)
        angles = torch.arccos(tmp)*180/3.14159
        angles = angles.detach().cpu().numpy()       # 两两相机射线之间的夹角
        with open(os.path.join(self.model_path, "multi_train_view_ncc.json"), 'w') as file:
            for id, cur_cam in enumerate(self.train_cameras[resolution_scale]):
                sorted_indices = np.lexsort((angles[id], diss[id]))
                mask = ( angles[id][sorted_indices] < args.multi_view_max_angle) & \
                    (diss[id][sorted_indices] > args.multi_view_min_dis) & \
                    (diss[id][sorted_indices] < args.multi_view_max_dis)
                sorted_indices = sorted_indices[mask]
                multi_view_num_ncc = min(self.multi_view_num_ncc, len(sorted_indices))
                json_d = {'ref_name' : cur_cam.image_name, 'nearest_name': []}
                for index in sorted_indices[:multi_view_num_ncc]:
                    cur_cam.nearest_names_ncc.append(self.train_cameras[resolution_scale][index].image_name)
                    json_d["nearest_name"].append(self.train_cameras[resolution_scale][index].image_name)
                json_str = json.dumps(json_d, separators=(',', ':'))
                file.write(json_str)
                file.write('\n')

    def compute_aligned_depth_pmap(self, args, resolution_scale):
        if args.aligned_depth and args.comput_pmap:
            pmaps_dir = os.path.join(args.model_path, "aligned_pmaps")
            os.makedirs(pmaps_dir, exist_ok=True)
            print("computing depth probability map")
            for id, cur_cam in enumerate(self.train_cameras[resolution_scale]):
                nearest_cam = []
                nearest_names = cur_cam.nearest_names_filter
                for nearest_name in nearest_names:
                    src_cam = [cam for cam in self.getTrainCameras(resolution_scale) if cam.image_name == nearest_name]
                    nearest_cam.append(src_cam[0])
                nearest_cam = None if len(nearest_cam) == 0 else nearest_cam
                    
                ref_intrinsics = cur_cam.get_intrinsics()
                ref_extrinsics = cur_cam.get_extrinsics()
                ref_depth_est = cur_cam.get_alignedD(args.use_mask)
                ref_depth_est = ref_depth_est.squeeze()

                if nearest_cam != None:
                    w_ref, h_ref = ref_depth_est.shape[1], ref_depth_est.shape[0]
                    depth_mask = (ref_depth_est > 0).to(args.data_device)
                    dynamic_consistency = torch.zeros((h_ref, w_ref), device=args.data_device)
                    num_consistent = torch.zeros((h_ref, w_ref), device=args.data_device)
                    for src_cam in nearest_cam:
                        src_intrinsics = src_cam.get_intrinsics()
                        src_extrinsics = src_cam.get_extrinsics()
                        src_depth_est = src_cam.get_alignedD(args.use_mask)
                        src_depth_est = src_depth_est.squeeze()
                        y_ref, x_ref = torch.meshgrid(torch.arange(0, h_ref, device=args.data_device), torch.arange(0, w_ref, device=args.data_device), indexing='ij')        # p
                        depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src, mask_reprojected = self.reproject_with_depth(args, ref_depth_est, ref_intrinsics, ref_extrinsics,    
                                                                                                                                src_depth_est, src_intrinsics, src_extrinsics)    # d', p', p', q, q
                        dist = torch.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)
                        depth_diff = torch.abs(depth_reprojected - ref_depth_est)            # [H, W]
                        relative_depth_diff = depth_diff / (ref_depth_est + 1e-8)            # [H, W]
                        err_mask = (dist < 5) & (relative_depth_diff < 0.02)

                        if not torch.any(err_mask):
                            continue
                        tmp_index = dist + 200 * relative_depth_diff
                        dynamic_consistency[err_mask] += torch.exp(-tmp_index[err_mask])
                        num_consistent[err_mask] += 1

                    if torch.any(num_consistent):
                        multi_view_mask = (num_consistent >= 1) #& (dynamic_consistency > 0.1 * num_consistent)
                        remove_mask = depth_mask & (~multi_view_mask)
                        cur_cam.consist_mask = multi_view_mask[None].clone().detach().cpu()
                            
                        if args.save_color_pmaps and id % 10 == 0:
                            image_array = np.zeros((h_ref, w_ref, 3), dtype=np.uint8)
                            image_array[multi_view_mask.cpu().numpy()] = [255, 255, 255]
                            image_array[remove_mask.cpu().numpy()] = [0, 0, 255]  # OpenCV 使用 BGR 顺序
                            save_path = os.path.join(pmaps_dir, f'{os.path.splitext(os.path.basename(cur_cam.image_path))[0]}.png')
                            cv2.imwrite(save_path, image_array)

        elif args.aligned_depth and not args.comput_pmap:
            pmaps_dir = os.path.join(args.model_path, "aligned_pmaps")
            os.makedirs(pmaps_dir, exist_ok=True)
            print("Using default depth probability map")
            for id, cur_cam in enumerate(self.train_cameras[resolution_scale]):
                if cur_cam.get_alignedD() is not None:
                    ref_depth_est = cur_cam.get_alignedD(args.use_mask)
                    ref_depth_est = ref_depth_est.squeeze()
                    multi_view_mask = (ref_depth_est > 0)
                    cur_cam.consist_mask = multi_view_mask.clone().detach().cpu()
                    if args.save_color_pmaps and id % 10 == 0:
                        image_array = np.zeros((h_ref, w_ref, 3), dtype=np.uint8)
                        image_array[multi_view_mask.cpu().numpy()] = [255, 255, 255]
                        save_path = os.path.join(pmaps_dir, f'{os.path.splitext(os.path.basename(cur_cam.image_path))[0]}.png')
                        cv2.imwrite(save_path, image_array)