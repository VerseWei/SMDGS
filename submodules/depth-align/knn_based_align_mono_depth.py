from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
import tyro
from utils.utils import depth_path_to_tensor, get_filename_list, depth_filtering, load_mask
# from global_align_depth import colmap_sfm_points_to_depths
from rich.console import Console
from rich.progress import track
from utils.rich_utils import CONSOLE
import cv2
import matplotlib
import pycolmap
import os
from cuml.neighbors import NearestNeighbors
from time import time
from typing import Literal
import collections
from typing import Dict, Literal, Optional
import struct

CameraModel = collections.namedtuple("CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple("Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple("Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])

class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)

CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12),
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model) for camera_model in CAMERA_MODELS])
CAMERA_MODEL_NAMES = dict([(camera_model.model_name, camera_model) for camera_model in CAMERA_MODELS])


CONSOLE = Console(width=120)
GLOBAL_BATCH_SIZE = 30
LOCAL_BATCH_SIZE = 1000
KNN_BATCH_SIZE = 1000000
K_NEIGHBORS = 30
gpu_device = "cuda" if torch.cuda.is_available() else "cpu"   # cuml only surpport default cuda 0
device = "cpu"

@dataclass
class AlignDepths:
    """Align mono depth to sfm scale
    """

    data: Path
    """Input dataset path"""
    iterations: int = 1000
    """Number of grad descent iterations to align depths"""
    depth_filtering: bool = True
    """Run mono depth filter"""
    save_filter: bool = True
    """Save mono depth filter"""
    sfm_filtering: bool = True
    """Run sfm depth filter"""
    use_mask: bool =  False
    """Align with mask"""
    segmask_type: str = "object"  # ["sky", "object"]
    """The type of segment mask"""
    skip_colmap_to_depths: bool = True
    """Skip colmap to sfm step"""
    align_method: Literal["closed_form", "robust_iteration"] = "robust_iteration"
    logo_info: bool = False
    sigma: float = 0.1

    def main(self) -> None:
        if self.logo_info:
            CONSOLE.print("Aligning sfm depth maps with mono estimates")
        # Align sparse sfm depth maps with mono depth maps
        global_batch_size = GLOBAL_BATCH_SIZE

        colmap_path = str(self.data / Path("colmap/sparse/0"))
        segmask_path = str(self.data / Path("mask"))
        output_path = str(self.data / Path("output"))
        reconstruction = pycolmap.Reconstruction(colmap_path)
        img_to_key = {image.name: key for key, image in reconstruction.images.items()}
        suffixs = [Path(image_name).suffix for image_name in img_to_key.keys()]

        sfm_depth_path = self.data / Path("sfm_depths")
        CONSOLE.print("Generating sfm depth maps from sparse colmap reconstruction")
        colmap_sfm_points_to_depths(
            recon_dir=Path(colmap_path),
            output_dir=Path(sfm_depth_path)
        )
        sfm_depth_filenames = get_filename_list(
            image_dir=self.data / Path("sfm_depths"), ends_with=(".npy", ".npz")
        )
        sfm_dir = self.data / Path("sfm_depths")
        self.sfm_filtering = False
        mono_depth_filenames = get_filename_list(
            image_dir=self.data / Path("mono_depths"), ends_with=(".npy", ".npz")
        )
        if os.path.exists(segmask_path):
            seg_mask_filenames = get_filename_list(
                image_dir=Path(segmask_path), ends_with=(".png")
            )

        num_frames = len(sfm_depth_filenames)

        cmap = matplotlib.colormaps.get_cmap('Spectral_r')
        
        for batch_index in range(0, num_frames, global_batch_size):
            batch_sfm_frames = sfm_depth_filenames[
                batch_index : batch_index + global_batch_size
            ]
            batch_sfm_set = set(batch_sfm_frames)
            batch_mono_frames = [frame for frame in mono_depth_filenames if (sfm_dir / frame.stem).with_suffix('.npy') in batch_sfm_set]
            mono_depth_filenames = [frame for frame in mono_depth_filenames if (sfm_dir / frame.stem).with_suffix('.npy') not in batch_sfm_set]
            if os.path.exists(segmask_path):
                batch_segmask_frames = [frame for frame in seg_mask_filenames if (sfm_dir / frame.stem).with_suffix('.npy') in batch_sfm_set]
                seg_mask_filenames = [frame for frame in seg_mask_filenames if (sfm_dir / frame.stem).with_suffix('.npy') not in batch_sfm_set]

            with torch.no_grad():
                mono_depth_tensors = []
                mono_scale_masks = []
                sfm_depth_tensors = []
                sfm_filters = []
                intrinsics_tensors = []

                for frame_index in range(len(batch_sfm_frames)):
                    sfm_frame = batch_sfm_frames[frame_index]
                    mono_frame = batch_mono_frames[frame_index]

                    mono_depth = depth_path_to_tensor(
                        mono_frame,
                        return_color=False,
                        scale_factor=0.001 if mono_frame.suffix == ".png" else 1,
                    )  # note that npy depth maps are in meters

                    if os.path.exists(segmask_path) and self.use_mask:
                        segmask_frame = batch_segmask_frames[frame_index]
                        mono_scale_mask = load_mask(segmask_frame, self.segmask_type)
                        mono_scale_mask = torch.tensor(mono_scale_mask, dtype=torch.bool)
                    else:
                        mono_scale_mask = (mono_depth < 5 * torch.mean(mono_depth)).to(torch.bool)
                    
                    mono_depth_tensors.append(mono_depth)
                    mono_scale_masks.append(mono_scale_mask)
                    save_mono_scale_mask = mono_scale_mask.cpu().numpy().squeeze().astype(np.uint8)
                    file_name = mono_frame.stem
                    save_mono_scale_mask_path = os.path.join(output_path, os.path.join('mono_scale_mask', file_name + '.png'))
                    os.makedirs(os.path.dirname(save_mono_scale_mask_path), exist_ok=True)
                    cv2.imwrite(save_mono_scale_mask_path, save_mono_scale_mask * 255)

                    image_id = img_to_key.get(f'{file_name}{suffixs[0]}')
                    Tw2c = reconstruction.images[image_id].projection_matrix()
                    Tw2c = np.concatenate((Tw2c, [[0, 0, 0, 1]]), axis=0)  
                    camera = reconstruction.cameras[reconstruction.images[image_id].camera_id]
                    fx = camera.focal_length_x
                    fy = camera.focal_length_y
                    cx = camera.principal_point_x
                    cy = camera.principal_point_y

                    intrinsic = torch.tensor([[fx, 0, cx],
                                              [0, fy, cy],
                                              [0, 0, 1]], device=device)
                    intrinsics_tensors.append(intrinsic)

                    sfm_depth = depth_path_to_tensor(
                        sfm_frame, return_color=False, scale_factor=1
                    ).to(device)

                    if self.sfm_filtering:
                        save_sfm_filter_path = os.path.join(output_path, os.path.join('sfm_filter', file_name + '.png'))
                        os.makedirs(os.path.dirname(save_sfm_filter_path), exist_ok=True)
                        _, sfm_filter = depth_filtering(sfm_depth, intrinsic, save_sfm_filter_path, gpu_device, True, torch.ones_like(sfm_depth, dtype=bool), mode="sparse")    # [H, W, 1]
                        sfm_filter = sfm_filter.to(device)
                    else:
                        sfm_filter = ((sfm_depth > 0) & (sfm_depth < torch.quantile(sfm_depth[sfm_depth > 0], 0.98))).to(torch.bool)
            
                    if self.logo_info:
                        CONSOLE.print(
                            f"{file_name} sfmpoints: {torch.count_nonzero(sfm_depth).item()} valid: {torch.count_nonzero(sfm_filter).item()}"
                        )

                    sfm_depth = sfm_depth * sfm_filter
                    sfm_depth_tensors.append(sfm_depth)               
                    sfm_filters.append(sfm_filter)
                    save_sfm_filtered_path = os.path.join(output_path, os.path.join('sfm_filtered', file_name + '.npy'))
                    os.makedirs(os.path.dirname(save_sfm_filtered_path), exist_ok=True)

            # KD Tree based local alignment
            for idx in track(
                range(len(mono_depth_tensors)),
                description="Knn-based local alignment..."
            ):  
                intrinsic = intrinsics_tensors[idx].to(device)
                sfm_depth = sfm_depth_tensors[idx].to(device)           # [H, W, 1]
                sfm_filter = sfm_filters[idx].to(device)
                mono_depth = mono_depth_tensors[idx].to(device)
                mono_filter = mono_scale_masks[idx].to(device)
                
                # depth filtering
                if self.depth_filtering:
                    file_name = batch_mono_frames[idx].stem
                    save_mono_filter_path = os.path.join(output_path, os.path.join('mono_filter', file_name + '.png'))
                    os.makedirs(os.path.dirname(save_mono_filter_path), exist_ok=True)
                    _, mono_filter = depth_filtering(mono_depth, intrinsic, save_mono_filter_path, gpu_device, self.save_filter, mono_filter, mode="dense")    # [H, W, 1]
                    mono_filter = mono_filter.to(device)
                local_batch_size = LOCAL_BATCH_SIZE
                knn_batch_size = KNN_BATCH_SIZE
                num_pixels = torch.count_nonzero(mono_filter)

                # Local alignment
                if torch.count_nonzero(mono_filter) != 0:
                    y_mono, x_mono = torch.meshgrid(torch.arange(0, mono_depth.shape[0], device=device), torch.arange(0, mono_depth.shape[1], device=device))  
                    x_mono, y_mono = x_mono.reshape([-1]), y_mono.reshape([-1])
                    xyz_mono = torch.matmul(torch.linalg.inv(intrinsic), torch.vstack((x_mono, y_mono, torch.ones_like(x_mono))) * mono_depth.reshape([-1]).to(device)) # [3, H*W=N]

                    knn_filter = (mono_filter & sfm_filter).to(device)                       # [H, W, 1]
                    knn_filter_flat = knn_filter.reshape([-1])                               # [N]
                    mono_filter_flat = mono_filter.reshape([-1])                             # [N] 
                    mono_3D_indices = torch.nonzero(mono_filter, as_tuple=False).squeeze().to(device)   # [K, 3]

                    mono_3D = xyz_mono[:, knn_filter_flat]                                              # [3, M]
                    knn_3D_indices = torch.nonzero(knn_filter, as_tuple=False).squeeze().to(device)     # [M, 3]
                    

                    if torch.sum(knn_filter) >  K_NEIGHBORS:
                        knn = NearestNeighbors(n_neighbors=K_NEIGHBORS)
                        knn.fit(mono_3D.T.to(gpu_device))
                        knn_xyz_mono = xyz_mono[:, mono_filter_flat].T
                        ids = []
                        for batch_index in range(0, num_pixels, knn_batch_size):
                            knn_xyz_ = knn_xyz_mono[batch_index: min(batch_index + knn_batch_size, num_pixels)]
                            _, id = knn.kneighbors(knn_xyz_.to(gpu_device))
                            ids.append(torch.from_numpy(id.get()))
                        ids = torch.cat(ids, dim=0).to(device)          # [K, K_NEIGHBORS]
                        
                        pixels_id = knn_3D_indices[ids]                                            # [K, K_NEIGHBORS, 3] 

                        pixels_id_row = pixels_id[..., 0]
                        pixels_id_col = pixels_id[..., 1]

                        Predict_depth = mono_depth[pixels_id_row, pixels_id_col, 0]     # [K, K_NEIGHBORS]
                        Target_depth = sfm_depth[pixels_id_row, pixels_id_col, 0]       # [K, K_NEIGHBORS]

                        scale = []
                        shift = []
                        for batch_index in range(0, num_pixels, local_batch_size):
                            predepth = Predict_depth[batch_index: batch_index + local_batch_size if batch_index + local_batch_size < num_pixels else num_pixels]
                            tardepth = Target_depth[batch_index: batch_index + local_batch_size if batch_index + local_batch_size < num_pixels else num_pixels]
                            if self.align_method == "closed_form":
                                scale_, shift_ = local_compute_scale_and_shift(predepth, tardepth, gpu_device)      # [local_batch_size, 1]
                            elif self.align_method == "robust_iteration":
                                scale_, shift_ = robust_scale_shift(predepth.to(gpu_device), tardepth.to(gpu_device), sigma=self.sigma)
                            scale.append(scale_.to(device))
                            shift.append(shift_.to(device))
                        scale = torch.cat(scale, dim=0)
                        shift = torch.cat(shift, dim=0)

                        mono_indices = mono_3D_indices[:, 0] * mono_depth.shape[1] + mono_3D_indices[:, 1]
                        mono_indices = mono_indices.long()

                        local_mono = mono_depth.view(-1)
                        local_mono[mono_indices] *= scale
                        local_mono[mono_indices] += shift
                        local_mono = local_mono.view(mono_depth.shape[0], mono_depth.shape[1], 1)
                        local_mono[~mono_filter] = 0
                        local_mono, mono_filter = depth_filtering(local_mono, intrinsic, None, gpu_device, False, mono_filter, mode="dense")
                        local_mono = local_mono.to(device)
                        mono_filter = mono_filter.to(device)

                        eval_mask = (local_mono > 0) & (sfm_depth > 0)
                        ref_diff = (local_mono - sfm_depth).abs() / (sfm_depth + 1e-8).clip(0, 1)
                        avg = torch.mean(ref_diff[eval_mask]) * 100
                        med = torch.median(ref_diff[eval_mask]) * 100

                        if self.logo_info:
                            CONSOLE.print(
                                f"[bold yellow]Average  relative depth alignment error for batch depths is: {avg:5f} %, and median value is: {med:5f} % "
                            )
                    else:
                        local_mono = torch.zeros_like(mono_depth, device=device)
                else:
                    local_mono = torch.zeros_like(mono_depth, device=device)    

                local_mono_np = local_mono.detach().cpu().numpy()
                save_local_path = os.path.join(output_path, os.path.join('local_aligned', file_name + '.npy'))
                os.makedirs(os.path.dirname(save_local_path), exist_ok=True)
                np.save(save_local_path, local_mono_np)
                local_mono_np =  local_mono_np.clip(0, 20).squeeze(-1)
                local_mono_np = (local_mono_np - local_mono_np.min()) / (local_mono_np.max() - local_mono_np.min() + 1e-8)
                local_mono_np = (cmap(local_mono_np)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
                cv2.imwrite(save_local_path.replace('npy', 'png'), local_mono_np)

            torch.cuda.empty_cache()             

def local_compute_scale_and_shift(prediction, target, device):
    prediction = prediction.to(device)
    target = target.to(device)
    a_00 = torch.sum(prediction * prediction, dim=1)
    a_01 = torch.sum(prediction, dim=1)
    a_11 = prediction.shape[1]

    b_0 = torch.sum(prediction * target, dim=1)
    b_1 = torch.sum(target, dim=1)

    det = a_00 * a_11 - a_01 * a_01

    mask = (det == 0)

    x_0 = (a_11 * b_0 - a_01 * b_1) / det
    x_1 = (-a_01 * b_0 + a_00 * b_1) / det

    x_0[mask] = 0
    x_1[mask] = 0

    return x_0, x_1

def robust_scale_shift(P, T_gt, sigma=0.1, num_iters=10):

    N, M = P.shape
    device = P.device
    s = torch.ones(N, device=device)
    t = torch.zeros(N, device=device)

    for _ in range(num_iters):
        pred = s[:, None] * P + t[:, None]
        residual = pred - T_gt  # [N, M]

        w = 1.0 / (1.0 + (residual ** 2) / ((sigma * T_gt) ** 2))

        W = w
        X = P
        Y = T_gt

        sum_w = torch.sum(W, dim=1) + 1e-8

        sum_wx = torch.sum(W * X, dim=1)
        sum_wy = torch.sum(W * Y, dim=1)
        sum_wxy = torch.sum(W * X * Y, dim=1)
        sum_wxx = torch.sum(W * X * X, dim=1)

        denom = sum_w * sum_wxx - sum_wx ** 2 + 1e-8
        s = (sum_w * sum_wxy - sum_wx * sum_wy) / denom
        t = (sum_wy - s * sum_wx) / sum_w

    return s, t

def colmap_sfm_points_to_depths(
    recon_dir: Path,
    output_dir: Path,
    min_depth: float = 0.001,
    max_depth: float = 1000,
    max_repoj_err: float = 2.5,
    min_n_visible: int = 5,
) -> Dict[int, Path]:
    """Converts COLMAP's points3d.bin to sparse depth maps

    Args:
        recon_dir: Path to the reconstruction directory, e.g. "sparse/0"
        output_dir: Path to the output directory.
        verbose: If True, logs progress of depth image creation.
        min_depth: Discard points closer than this to the camera.
        max_depth: Discard points farther than this from the camera.
        max_repoj_err: Discard points with reprojection error greater than this
          amount (in pixels).
        min_n_visible: Discard 3D points that have been triangulated with fewer
          than this many frames.
        include_depth_debug: Also include debug images showing depth overlaid
          upon RGB.

    Returns:
        Depth file paths indexed by COLMAP image id
    """
    depth_scale_to_integer_factor = 1

    if (recon_dir / "points3D.bin").exists():
        ptid_to_info = read_points3D_binary(recon_dir / "points3D.bin")
        cam_id_to_camera = read_cameras_binary(recon_dir / "cameras.bin")
        im_id_to_image = read_images_binary(recon_dir / "images.bin")
    elif (recon_dir / "points3D.txt").exists():
        ptid_to_info = read_points3D_text(recon_dir / "points3D.txt")
        cam_id_to_camera = read_cameras_text(recon_dir / "cameras.txt")
        im_id_to_image = read_images_text(recon_dir / "images.txt")
    # Only support first camera
    CAMERA_ID = 1
    W = cam_id_to_camera[CAMERA_ID].width
    H = cam_id_to_camera[CAMERA_ID].height

    iter_images = iter(im_id_to_image.items())

    for im_id, im_data in track(iter_images, description="..."):
        # TODO(1480) BEGIN delete when abandoning colmap_parsing_utils
        pids = [pid for pid in im_data.point3D_ids if pid != -1]
        xyz_world = np.array([ptid_to_info[pid].xyz for pid in pids])
        # delete
        # xyz_world = np.array([p.xyz for p in ptid_to_info.values()])
        rotation = qvec2rotmat(im_data.qvec)

        z = (rotation @ xyz_world.T)[-1] + im_data.tvec[-1]
        errors = np.array([ptid_to_info[pid].error for pid in pids])
        n_visible = np.array([len(ptid_to_info[pid].image_ids) for pid in pids])
        uv = np.array(
            [
                im_data.xys[i]
                for i in range(len(im_data.xys))
                if im_data.point3D_ids[i] != -1
            ]
        )

        idx = np.where(
            (z >= min_depth)
            & (z <= max_depth)
            & (errors <= max_repoj_err)
            & (n_visible >= min_n_visible)
            & (uv[:, 0] >= 0)
            & (uv[:, 0] < W)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < H)
        )
        z = z[idx]
        uv = uv[idx]

        uu, vv = uv[:, 0].astype(int), uv[:, 1].astype(int)
        depth = np.zeros((H, W), dtype=np.float32)
        depth[vv, uu] = z

        depth_img = depth_scale_to_integer_factor * depth

        out_name = Path(str(im_data.name)).stem
        depth_path = output_dir / out_name

        save_depth(
            depth=depth_img, depth_path=depth_path, scale_factor=1, verbose=False
        )

def read_cameras_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasText(const std::string& path)
        void Reconstruction::ReadCamerasText(const std::string& path)
    """
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(id=camera_id, model=model, width=width, height=height, params=params)
    return cameras


def read_cameras_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, num_bytes=8 * num_params, format_char_sequence="d" * num_params)
            cameras[camera_id] = Camera(
                id=camera_id, model=model_name, width=width, height=height, params=np.array(params)
            )
        assert len(cameras) == num_cameras
    return cameras

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def write_cameras_text(cameras, path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasText(const std::string& path)
        void Reconstruction::ReadCamerasText(const std::string& path)
    """
    HEADER = (
        "# Camera list with one line of data per camera:\n"
        + "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        + "# Number of cameras: {}\n".format(len(cameras))
    )
    with open(path, "w") as fid:
        fid.write(HEADER)
        for _, cam in cameras.items():
            to_write = [cam.id, cam.model, cam.width, cam.height, *cam.params]
            line = " ".join([str(elem) for elem in to_write])
            fid.write(line + "\n")

def save_depth(depth, depth_path, verbose=True, scale_factor=1) -> None:
    """helper to save metric depths

    Args:
        depth: image to save (numpy, Tensor)
        depth_path: path to save
        verbose: whether to print save path
        scale_factor: depth metric scaling factor

    Returns:
        None
    """
    if torch.is_tensor(depth):
        depth = depth.float() / scale_factor
        depth = depth.detach().cpu().numpy()
    else:
        depth = depth / scale_factor
    if not Path(os.path.dirname(depth_path)).exists():
        Path(os.path.dirname(depth_path)).mkdir(parents=True)
    if verbose:
        print("saving to: ", depth_path)
    np.save(depth_path, depth)

def read_images_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    """
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack([tuple(map(float, elems[0::3])), tuple(map(float, elems[1::3]))])
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = Image(
                    id=image_id,
                    qvec=qvec,
                    tvec=tvec,
                    camera_id=camera_id,
                    name=image_name,
                    xys=xys,
                    point3D_ids=point3D_ids,
                )
    return images


def read_images_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = b""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":  # look for the ASCII 0 entry
                image_name += current_char
                current_char = read_next_bytes(fid, 1, "c")[0]
            image_name = image_name.decode("utf-8")
            num_points2D = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24 * num_points2D, format_char_sequence="ddq" * num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])), tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=image_name,
                xys=xys,
                point3D_ids=point3D_ids,
            )
    return images

def read_points3D_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    points3D = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                point3D_id = int(elems[0])
                xyz = np.array(tuple(map(float, elems[1:4])))
                rgb = np.array(tuple(map(int, elems[4:7])))
                error = float(elems[7])
                image_ids = np.array(tuple(map(int, elems[8::2])))
                point2D_idxs = np.array(tuple(map(int, elems[9::2])))
                points3D[point3D_id] = Point3D(
                    id=point3D_id, xyz=xyz, rgb=rgb, error=error, image_ids=image_ids, point2D_idxs=point2D_idxs
                )
    return points3D


def read_points3D_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            binary_point_line_properties = read_next_bytes(fid, num_bytes=43, format_char_sequence="QdddBBBd")
            point3D_id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(fid, num_bytes=8 * track_length, format_char_sequence="ii" * track_length)
            image_ids = np.array(tuple(map(int, track_elems[0::2])))
            point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
            points3D[point3D_id] = Point3D(
                id=point3D_id, xyz=xyz, rgb=rgb, error=error, image_ids=image_ids, point2D_idxs=point2D_idxs
            )
    return points3D

def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )

if __name__ == "__main__":
    tyro.cli(AlignDepths).main()
