
import os
import torch
import random
import numpy as np
from random import randint
from utils.loss_utils import l1_loss, ssim, cos_loss, lncc_weight
from utils.graphics_utils import patch_offsets
from gaussian_renderer import render, network_gui
import sys, time
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import cv2
from tqdm import tqdm
from utils.image_utils import psnr, normal2curv
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import torch.nn.functional as F
from visible_detection import visible_detection
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
import time
import torch.nn.functional as F

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
setup_seed(22)

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    gaussians = GaussianModel(dataset)
    scene = Scene(dataset, gaussians)
    use_mask = dataset.use_mask
    contrib_densify = dataset.contrib_densify
    gaussians.training_setup(opt)

    if not use_mask:
        print("Training without mask")
    else:
        print("Training with mask")
    
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=dataset.data_device)
    bg = torch.rand((3), device=dataset.data_device) if opt.random_background else background
    debug_path = os.path.join(scene.model_path, "debug")
    os.makedirs(debug_path, exist_ok=True)

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    train_viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_multi_view_geo_for_log = 0.0
    geo_depth_loss = None
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):

        iter_start.record()
        gaussians.update_learning_rate(iteration)
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random camera
        if not train_viewpoint_stack:
            train_viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = train_viewpoint_stack.pop(randint(0, len(train_viewpoint_stack)-1))

        # Pick a nearest camera
        if iteration > opt.multi_view_weight_from_iter and len(viewpoint_cam.nearest_names_ncc) > 0:
            nearest_name = random.sample(viewpoint_cam.nearest_names_ncc, 1)
            nearest_cam = [cam for cam in scene.getTrainCameras() if cam.image_name == nearest_name[0]]
            nearest_cam = None if len(nearest_cam) == 0 else nearest_cam[0]

        gt_image, gt_image_gray = viewpoint_cam.get_image(bg, with_mask = False)
        mask_gt = viewpoint_cam.get_gtMask(use_mask)
        monoN = None

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        if iteration > opt.multi_view_weight_from_iter and args.warp:
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg, contrib_densify=contrib_densify, depth_threshold=opt.depth_threshold*scene.cameras_extent, \
                                 nearest_camera=nearest_cam, use_mask=use_mask, convert_depth=True)
        else:
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg, contrib_densify=contrib_densify, depth_threshold=opt.depth_threshold*scene.cameras_extent)
        image, viewspace_point_tensor, viewspace_point_tensor_abs, visibility_filter, radii, depth_normal, normal, opac_s, depth = \
            render_pkg["render"], render_pkg["viewspace_points"],  render_pkg["viewspace_points_abs"], render_pkg["visibility_filter"], render_pkg["radii"], \
            render_pkg["depth_normal"], render_pkg["rendered_normal"], render_pkg["opac_s"], render_pkg["plane_depth"]

        if gaussians.skybox:
            visibility_filter[:gaussians.skybox_points] = False

        # Loss
        ssim_loss = (1.0 - ssim(image, gt_image))
        Ll1 = l1_loss(image, gt_image)
        image_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss
        loss = image_loss.clone()

        geo_factor = 5

        if use_mask:
            geom_prior_mask = (depth.detach() < geo_factor * scene.cameras_extent).float()
        else:
            geom_prior_mask = torch.ones_like(depth.detach()).to(dataset.data_device)

        # smooth loss
        if dataset.smooth and iteration > 7000:
            curv_nm = normal2curv(monoN, mask_gt)
            image_weight = (1 - curv_nm.clamp(0,1).detach()) ** 2
            loss_surface = (geom_prior_mask * image_weight * mask_gt * ((depth_normal - normal).abs().sum(0))).mean()             
            loss += opt.lambda_4 * loss_surface

        # opacity loss
        if dataset.scene_opac and iteration < 15_000:
            loss_opac = (opac_s * (1 - mask_gt)).mean()
            loss += opt.lambda_6 * loss_opac

        # prior normal loss
        if iteration < opt.geom_prior_until_iter and dataset.prior_normal:
            monoN = viewpoint_cam.get_mono(use_mask)[:3]
            consist_weight = torch.exp(-((1 - torch.sum(normal.detach() * monoN, dim=0, keepdim=True)) ** 2) / (0.5 ** 2))
            loss_monoN = cos_loss(normal, monoN, weight=mask_gt*geom_prior_mask*consist_weight)                  
            curv_nr = normal2curv(normal, mask_gt)
            curv_nm = normal2curv(monoN, mask_gt)
            loss_curv = l1_loss(curv_nr, curv_nm, mask_gt*geom_prior_mask*consist_weight)
            loss += opt.lambda_3 * (0.02 * (2 - (iteration / opt.geom_prior_until_iter)) * loss_monoN + 0.002 * loss_curv)

        # prior depth loss
        if dataset.aligned_depth and iteration < opt.geom_prior_until_iter and dataset.prior_depth:       
            alignedD = viewpoint_cam.get_alignedD(use_mask)
            consist_mask = viewpoint_cam.get_pmaps(use_mask)            
            depth_err = torch.abs((alignedD - depth) / (alignedD + depth + 1e-8))
            depth_weight = (1.0 / torch.exp(100 * depth_err)).detach()
            LalignedD = (depth_weight * torch.abs(alignedD - depth))[consist_mask.to(torch.bool) & geom_prior_mask.to(torch.bool)].mean()
            loss += opt.lambda_2 * (1 / scene.cameras_extent) * LalignedD

        # scale loss
        if visibility_filter.sum() > 0:
            scale = gaussians.get_scaling[visibility_filter]
            sorted_scale, _ = torch.sort(scale, dim=-1)
            min_scale_loss = sorted_scale[...,0]
            loss += opt.lambda_1 * min_scale_loss.mean()

        # debug
        if (iteration % 100 == 0) or (iteration == first_iter):    
            show_mask = mask_gt
            gt_img_show = ((gt_image).permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
            img_show = ((image).permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
            normal_show = (((normal * show_mask+1.0)*0.5).permute(1,2,0).clamp(0,1)*255).detach().cpu().numpy().astype(np.uint8)
            if monoN is None:
                monoN = torch.ones_like(normal, device="cuda")
            mono_normal_show = (((monoN+1.0)*0.5).permute(1,2,0).clamp(0,1)*255).detach().cpu().numpy().astype(np.uint8)
            depth_normal_show = (((depth_normal * show_mask+1.0)*0.5).permute(1,2,0).clamp(0,1)*255).detach().cpu().numpy().astype(np.uint8)
            depth_i = (render_pkg['plane_depth'] * show_mask).squeeze().detach().cpu().numpy().clip(0, 30)
            depth_i = (depth_i - depth_i.min()) / (depth_i.max() - depth_i.min() + 1e-20)
            depth_i = (depth_i * 255).clip(0, 255).astype(np.uint8)
            depth_color = cv2.applyColorMap(depth_i, cv2.COLORMAP_JET)
            row0 = np.concatenate([gt_img_show, img_show, depth_color], axis=1)                      # debug      gt_img   rendered_img   rendered_depth  
            row1 = np.concatenate([mono_normal_show, normal_show, depth_normal_show], axis=1)        #            mono_normal  rendered_normal depth_normal
            image_to_show = np.concatenate([row0, row1], axis=0)
            cv2.imwrite(os.path.join(debug_path, "%05d"%iteration + "_" + viewpoint_cam.image_name + ".jpg"), image_to_show)

        # multi-view loss
        if iteration > opt.multi_view_weight_from_iter and dataset.warp:
            projected_depthmap, projected_normalmap = render_pkg["projected_depthmap"], render_pkg["projected_normalmap"]
            projected_mask = (projected_depthmap != 0)
            depth_noise = torch.zeros_like(projected_depthmap, device=dataset.data_device)
            nearest_render_pkg = render(nearest_cam, gaussians, pipe, bg, return_plane=True, return_depth_normal=False)
            near_depth = nearest_render_pkg['plane_depth']
            near_normal = nearest_render_pkg['rendered_normal']
            projected_normalmap = torch.nn.functional.normalize(projected_normalmap, p=2, dim=0)
            near_normal = torch.nn.functional.normalize(near_normal, p=2, dim=0)
            depth_noise[projected_mask] = torch.abs((projected_depthmap[projected_mask] - near_depth[projected_mask]) / \
                                                    (projected_depthmap[projected_mask].detach() + near_depth[projected_mask].detach()))
            multi_view_loss_mask = (depth_noise.detach() > 0) & (depth_noise.detach() < 0.01) & (near_depth.detach() < geo_factor * scene.cameras_extent)

            geo_weight = (1.0 / torch.exp(100 * depth_noise)).detach()
            geo_depth_loss = (geo_weight * torch.abs(projected_depthmap - near_depth))[multi_view_loss_mask].mean()        
            loss += opt.lambda_5 * (1 / scene.cameras_extent) * geo_depth_loss           
            
            visible_mask = visible_detection(render_pkg["plane_depth"], render_pkg["face_normal"], viewpoint_cam.world_view_transform, viewpoint_cam.get_k(),
                                             nearest_cam.world_view_transform, nearest_cam.get_k(), mask_gt.to(torch.bool), projected_mask)
            if iteration % 100 == 0:
                multi_view_mask_show = (multi_view_loss_mask*255).float().squeeze().detach().cpu().numpy().astype(np.uint8)
                visible_mask_show = (visible_mask*255).float().squeeze().detach().cpu().numpy().astype(np.uint8)
                show_mask = np.concatenate([multi_view_mask_show, visible_mask_show], axis=1)
                cv2.imwrite(os.path.join(debug_path, "%05d"%iteration + "_" + viewpoint_cam.image_name + ".png"), show_mask)

            if visible_mask.sum() > 0:
                patch_size = dataset.patch_size
                sample_num = dataset.sample_num
                total_patch_size = (dataset.patch_size * 2 + 1) ** 2

                with torch.no_grad():
                    ix, iy = torch.meshgrid(
                        torch.arange(viewpoint_cam.image_width), torch.arange(viewpoint_cam.image_height), indexing='xy')
                    pixels = torch.stack([ix, iy], dim=-1).float().to(depth.device)
                    pixels = pixels.reshape(-1, 2)
                    v_mask = visible_mask.reshape(-1)
                    valid_indices = torch.arange(v_mask.shape[0], device="cuda")[v_mask]
                    if v_mask.sum().item() < sample_num:
                        sample_num = v_mask.sum().item()
                    index = np.random.choice(v_mask.sum().item(), sample_num, replace = False)
                    valid_indices = valid_indices[index]
                    sample_mask = torch.zeros(v_mask.shape[0], dtype=torch.bool, device="cuda")
                    sample_mask[valid_indices] = True

                    offsets = patch_offsets(patch_size, "cuda")
                    ori_pixels_patch = pixels[sample_mask].reshape(-1, 1, 2) + offsets.float()
                    ori_pixels_patch[:, :, 0] = ori_pixels_patch[:, :, 0].clamp(0, viewpoint_cam.image_width - 1)
                    ori_pixels_patch[:, :, 1] = ori_pixels_patch[:, :, 1].clamp(0, viewpoint_cam.image_height - 1)
                    pixels_patch = ori_pixels_patch.clone()

                    pixels_patch[:, :, 0] = 2 * pixels_patch[:, :, 0] / (viewpoint_cam.image_width - 1) - 1.0
                    pixels_patch[:, :, 1] = 2 * pixels_patch[:, :, 1] / (viewpoint_cam.image_height - 1) - 1.0

                    ori_rays = ori_pixels_patch.clone()
                    ori_rays[:, :, 0] = (ori_rays[:, :, 0] - viewpoint_cam.Cx) / viewpoint_cam.Fx
                    ori_rays[:, :, 1] = (ori_rays[:, :, 1] - viewpoint_cam.Cy) / viewpoint_cam.Fy

                    ref_gray_val = F.grid_sample(gt_image_gray[None], pixels_patch.view(1, -1, 1, 2), align_corners=True)
                    ref_gray_val = ref_gray_val.reshape(-1, total_patch_size)

                homo_pixels_patch = torch.cat([ori_rays.reshape(-1, 2), torch.ones((sample_num * total_patch_size, 1), device="cuda")], dim=-1)
                pixels_patch_stack = ori_pixels_patch.reshape(-1, 2)    
                pixels_patch_mask = visible_mask[0, pixels_patch_stack[:, 1].long(), pixels_patch_stack[:, 0].long()].reshape(sample_num, total_patch_size).to(torch.float32)
                pts_ref_cam = homo_pixels_patch * depth[0, pixels_patch_stack[:, 1].long(), pixels_patch_stack[:, 0].long()].unsqueeze(1)
                pts_ref_cam_homo = torch.cat([pts_ref_cam, torch.ones((pts_ref_cam.shape[0], 1), device="cuda")], dim=-1)
                
                ref_extr = viewpoint_cam.get_extrinsics()
                ref_extr_inv = torch.linalg.inv(ref_extr)
                pts_world_homo = torch.matmul(pts_ref_cam_homo, ref_extr_inv.T)
                
                src_extr = nearest_cam.get_extrinsics()
                pts_src_cam_homo = torch.matmul(pts_world_homo, src_extr.T)
            
                src_pixels_patch = pts_src_cam_homo[:, :2].clone().reshape(sample_num, total_patch_size, 2)
                src_pixels_patch[:, :, 0] = nearest_cam.Fx * src_pixels_patch[:, :, 0] / (pts_src_cam_homo[:, 2].reshape(sample_num, total_patch_size)) + nearest_cam.Cx
                src_pixels_patch[:, :, 1] = nearest_cam.Fy * src_pixels_patch[:, :, 1] / (pts_src_cam_homo[:, 2].reshape(sample_num, total_patch_size)) + nearest_cam.Cy

                src_pixels_patch[:, :, 0] = 2 * src_pixels_patch[:, :, 0] / (nearest_cam.image_width - 1) - 1.0
                src_pixels_patch[:, :, 1] = 2 * src_pixels_patch[:, :, 1] / (nearest_cam.image_height - 1) - 1.0

                _, nearest_image_gray = nearest_cam.get_image(bg, with_mask = False)
                src_gray_val = F.grid_sample(nearest_image_gray[None], src_pixels_patch.view(1, -1, 1, 2), align_corners=True)
                src_gray_val = src_gray_val.reshape(-1, total_patch_size)

                ## compute loss
                ncc, ncc_mask = lncc_weight(ref_gray_val, src_gray_val, weight=pixels_patch_mask)
                mask = ncc_mask.reshape(-1)
                ncc = ncc.reshape(-1)
                ncc = ncc[mask].squeeze()

                if mask.sum() > 0:
                    ncc_loss = 0.15 * ncc.mean()
                    loss += ncc_loss

        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * image_loss.item() + 0.6 * ema_loss_for_log
            ema_multi_view_geo_for_log = 0.4 * geo_depth_loss.item() if geo_depth_loss is not None else 0.0 + 0.6 * ema_multi_view_geo_for_log
            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "Geo": f"{ema_multi_view_geo_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, bg), bg, use_mask, dataset.data_device)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                    
            if gaussians.skybox:
                gaussians._xyz.grad[:gaussians.skybox_points, :] = 0

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                mask = (render_pkg["out_observe"] > 0) & visibility_filter
                gaussians.max_radii2D[mask] = torch.max(gaussians.max_radii2D[mask], radii[mask])
                if gaussians.skybox:
                    viewspace_point_tensor[:gaussians.skybox_points, :] = 0
                    viewspace_point_tensor_abs[:gaussians.skybox_points, :] = 0
                gaussians.add_densification_stats(viewspace_point_tensor, viewspace_point_tensor_abs, visibility_filter, render_pkg["pixels"], render_pkg["depth_z"], depth_threshold=opt.depth_threshold*scene.cameras_extent)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    if dataset.contrib_densify:
                        gaussians.densify_and_prune(opt.densify_grad_threshold, \
                                                    opt.opacity_cull_threshold, \
                                                    scene.cameras_extent, size_threshold)
                    else:
                        gaussians.densify_and_prune(0.0002, opt.opacity_cull_threshold, \
                                                        scene.cameras_extent, size_threshold)

            # reset_opacity
            if iteration < opt.densify_until_iter:
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            if iteration % 500 == 0:
                torch.cuda.empty_cache()
    
    torch.cuda.empty_cache()

def prepare_output_and_logger(args):    

    if not args.model_path:
        model_path = f'{args.source_path}/train_output/{os.path.basename(args.source_path)}_smdgs'
        label = 0
        args.model_path = f'{model_path}_{label}'
        while os.path.exists(f'{model_path}_{label}'):
            label += 1
            args.model_path = f'{model_path}_{label}'

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, bg, use_mask, data_device):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    out = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = out["render"]
                    image = torch.clamp(image, 0.0, 1.0)
                    gt_image, _ = viewpoint.get_image(bg, False)
                    gt_image = torch.clamp(gt_image.to(data_device), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    torch.set_num_threads(8)
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6007)
    parser.add_argument('--debug_from', type=int, default=-100)           
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_00, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_00, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    args.test_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    t0 = time.time()
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)
    t1 = time.time()
    print(f"Used Time: {(t1 - t0) / 60} mins.")
    # All done
    print("\nTraining complete.")
