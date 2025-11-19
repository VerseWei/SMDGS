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
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import imageio
import skimage
import cv2
from concurrent.futures import ThreadPoolExecutor

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    K: None
    prcppoint: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    mask: np.array
    mono: np.array
    alignedD: None
    mono_filter: np.array


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius, "center": center}

def load_objectmask(path):
    alpha = imageio.imread(path, pilmode='F')
    alpha = skimage.img_as_float32(alpha) / 255
    white = alpha > 0
    alpha[white] = 1
    return alpha

def load_skymask(path):
    alpha = imageio.imread(path, pilmode='F')
    sky = (alpha > 150)
    alpha[sky] = 0
    alpha[~sky] = 255
    alpha = skimage.img_as_float32(alpha) / 255
    return alpha

def read_monoData(path):
    _, ext = os.path.splitext(path)
    mono = np.load(path)
    if ext == ".npz":
        mono = mono[mono.files[0]]
        mono = np.squeeze(mono)
        mono = mono[np.newaxis, ...]
    if len(mono.shape) == 4:
        mono = mono[0]
    elif len(mono.shape) == 2:
        mono = mono[None]
    return mono

def load_poses(pose_path, num):
    poses = []
    with open(pose_path, "r") as f:
        lines = f.readlines()
    for i in range(num):
        line = lines[i]
        c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
        c2w[:3,3] = c2w[:3,3] * 10.0
        w2c = np.linalg.inv(c2w)
        w2c = w2c
        poses.append(w2c)
    poses = np.stack(poses, axis=0)
    return poses

def create_mask(image_path):
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    rect = (0, 0, width - 1, height - 1)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    mask = np.zeros(image.shape[:2], np.uint8)

    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8') * 255

    return mask2

def save_mask(image_path, output_path):
    mask_image = create_mask(image_path)
    cv2.imwrite(output_path, mask_image)

def create_masks_for_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    image_formats = ('.jpg', '.jpeg', '.png')
    files = [f for f in os.listdir(input_folder) if f.lower().endswith(image_formats)]

    with ThreadPoolExecutor() as executor:
        futures = []
        for filename in files:
            image_path = os.path.join(input_folder, filename)
            image_name = os.path.basename(image_path).split(".")[0]
            output_path = os.path.join(output_folder, f"{image_name}.png")
            futures.append(executor.submit(save_mask, image_path, output_path))
        for future in futures:
            future.result()  # Wait for all tasks to complete

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, create_mask, mask_type):
    cam_infos = []

    if not os.path.exists(f'{images_folder}/../../mask') and create_mask:
        print("Creating mask by OpenCV")
        create_masks_for_folder(images_folder, f'{images_folder}/../../mask')

    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)
        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
            prcppoint = np.array([intr.params[1] / width, intr.params[2] / height])
            fx = focal_length_x
            fy = focal_length_x
            cx = intr.params[1]
            cy = intr.params[2]
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
            prcppoint = np.array([intr.params[2] / width, intr.params[3] / height])
            fx = focal_length_x
            fy = focal_length_y
            cx = intr.params[2]
            cy = intr.params[3]
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        elif intr.model=="SIMPLE_RADIAL":
            f, cx, cy, r = intr.params
            FovY = focal2fov(f, height)
            FovX = focal2fov(f, width)
            prcppoint = np.array([cx / width, cy / height])
            # undistortion
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
            D = np.array([r, 0, 0, 0])  # Only radial distortion
            image_undistorted = cv2.undistort(image_cv, K, D, None)
            image_undistorted = cv2.cvtColor(image_undistorted, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image_undistorted)
        else:

            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        try:
            alignedD = np.load(f'{images_folder}/../../output/local_aligned/{image_name}.npy')
            alignedD = np.transpose(alignedD, (2, 0, 1))
            alignedD = np.clip(alignedD, 0, None)
        except  FileNotFoundError:
            alignedD = None

        try:
            monoN = read_monoData(f'{images_folder}/../../mono_normals/{image_name}.npy')
            try:
                monoD = read_monoData(f'{images_folder}/../../mono_depths/{image_name}.npz')
            except FileNotFoundError:
                monoD = np.zeros_like(monoN[:1])
            mono = np.concatenate([monoN, monoD], 0)
        except FileNotFoundError:
            mono = None

        if mask_type == "sky":
            try:
                mask = load_skymask(f'{images_folder}/../../mask/{image_name}.png')[None]      # [1, H, W]    取反scene为白，sky为黑
            except FileNotFoundError:
                mask = np.ones([1, image.size[1], image.size[0]]).astype(np.float32)
        elif mask_type == "object":
            try:
                mask = load_objectmask(f'{images_folder}/../../mask/{image_name}.png')[None]
            except FileNotFoundError:
                mask = np.ones([1, image.size[1], image.size[0]]).astype(np.float32)
        
        try:
            mono_filter = load_objectmask(f'{images_folder}/../../output/mono_filter/{image_name}.png')[None]
        except FileNotFoundError:
            mono_filter = np.ones([1, image.size[1], image.size[0]]).astype(np.float32)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, K=K, prcppoint=prcppoint, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height, mask=mask, mono=mono, alignedD=alignedD, mono_filter=mono_filter)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T

    positions = np.nan_to_num(positions, nan=0.0)
    colors = np.nan_to_num(colors, nan=0.0)
    normals = np.nan_to_num(normals, nan=0.0)

    valid = np.all(np.abs(positions) <= 1000, axis=1)
    positions = positions[valid]
    colors = colors[valid]
    normals = normals[valid]

    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb, normals = None):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    if normals is None:
        normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, create_mask, mask_type, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "colmap/sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "colmap/sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "colmap/sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "colmap/sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
    reading_dir = "colmap/images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir), create_mask=create_mask, mask_type=mask_type)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
    
    js_file = f"{path}/split.json"
    train_list = None
    test_list = None
    if os.path.exists(js_file):
        with open(js_file) as file:
            meta = json.load(file)
            train_list = meta["train"]
            test_list = meta["test"]
            print(f"train_list {len(train_list)}, test_list {len(test_list)}")

    if train_list is not None:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if c.image_name in train_list]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if c.image_name in test_list]
        print(f"train_cam_infos {len(train_cam_infos)}, test_cam_infos {len(test_cam_infos)}")
    elif eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "colmap/sparse/0/points3D.ply")
    bin_path = os.path.join(path, "colmap/sparse/0/points3D.bin")
    txt_path = os.path.join(path, "colmap/sparse/0/points3D.txt")
    if not os.path.exists(ply_path) or True:
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}