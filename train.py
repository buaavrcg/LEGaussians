import os
import torch
import yaml
from random import randint
import torch.nn.functional as F
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from scene.index_decoder import *
from utils.general_utils import safe_state
from utils.lem_utils import *
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import Namespace
import configargparse
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

FIRST_REPORT = True

def compute_semantic_loss(language_feature_indices, gt_language_feature_indices, uncertainty, ce):
    # Lem Loss
    # Resize clip feature indices to match image size
    a, b = language_feature_indices.shape[1], language_feature_indices.shape[2]

    upsampled_gt_language_feature_indices = \
                F.interpolate(gt_language_feature_indices.unsqueeze(0).float(), size=(a, b), mode='nearest').squeeze(0).long()

    upsampled_gt_language_feature_indices = upsampled_gt_language_feature_indices.permute(1,2,0).view(-1)
    
    language_feature_indices = language_feature_indices.permute(1,2,0)
    language_feature_indices = language_feature_indices.reshape(-1, language_feature_indices.shape[-1])
    semantic_loss = ce(language_feature_indices, upsampled_gt_language_feature_indices)
    
    uncertainty = 1.0 - uncertainty.permute(1,2,0).reshape(-1)
    semantic_loss = (semantic_loss * uncertainty).mean()
    
    return semantic_loss

def compute_mlp_smooth_loss(xyzs, embedding, decoder, gs_semantic_features, uncertainty, smooth_loss_uncertainty_min):
    xyzs_pe = embedding(xyzs)
    xyzs_features = decoder(xyzs_pe)
    xyz_mlp_loss = ((xyzs_features - gs_semantic_features.detach()) ** 2).mean(dim=1)
    
    weights = (1 - uncertainty) * smooth_loss_uncertainty_min + uncertainty * 1.0
    semantic_features_smooth_loss = ((xyzs_features.detach() - gs_semantic_features) ** 2).mean(dim=1) * weights
    
    return xyz_mlp_loss.mean(), semantic_features_smooth_loss.mean()

def training(dataset, opt, pipe, 
             testing_iterations, test_set,
             saving_iterations, checkpoint_iterations,
             checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset, opt, pipe)
    gaussians = GaussianModel(dataset.sh_degree, dataset.semantic_features_dim, dataset.points_num_limit)
    scene = Scene(dataset, gaussians, test_set=test_set)
    gaussians.training_setup(opt)
    
    index_decoder = IndexDecoder(dataset.semantic_features_dim, dataset.codebook_size).to("cuda")
    decoder_optim = torch.optim.AdamW(index_decoder.parameters(), lr=opt.decoder_lr, weight_decay=1e-5)
    ce = torch.nn.CrossEntropyLoss(reduction='none')
    index_decoder.train()
    
    dataset.xyz_encoding_in_channels_xyz = 3*(2*dataset.xyz_embedding_N_freqs+1)
    xyz_embedding = Embedding(3, dataset.xyz_embedding_N_freqs).to("cuda")
    xyz_decoder = XyzMLP(D=dataset.xyz_encoding_D,
                         W=dataset.xyz_encoding_W,
                         in_channels_xyz=dataset.xyz_encoding_in_channels_xyz, 
                         out_channels_xyz=dataset.xyz_encoding_out_channels_xyz).to("cuda")
    xyz_decoder_optim = torch.optim.AdamW(xyz_decoder.parameters(), lr=opt.decoder_lr, weight_decay=1e-5)
    xyz_decoder.train()

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        index_decoder_ckpt = os.path.join(os.path.dirname(checkpoint), "index_decoder_" + os.path.basename(checkpoint))
        index_decoder.load_state_dict(torch.load(index_decoder_ckpt))

    bg_color = [1, 1, 1, 1,1,1,1,1,1,1,1,1] if dataset.white_background else [0, 0, 0, 0,0,0,0,0,0,0,0,0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress", dynamic_ncols=True)
    first_iter += 1
    
    color_map = generate_colors(dataset.codebook_size)
    
    for iteration in range(first_iter, opt.iterations + 1):        
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        semantic_features = render_pkg["semantic_features"]
        uncertainty = render_pkg["uncertainty"]
        
        # Index Features Regularization Loss
        xyzs = gaussians.get_xyz.detach()
        gs_semantic_features = gaussians.get_semantic_features
        gs_uncertainty = gaussians.get_uncertainty
        
        if iteration % 10 == 0:
            xyz_mlp_loss, smooth_loss = compute_mlp_smooth_loss(xyzs, xyz_embedding, xyz_decoder, gs_semantic_features, gs_uncertainty.squeeze().detach(), 
                                                  dataset.smooth_loss_uncertainty_min)
        else:
            xyz_mlp_loss, smooth_loss = 0.0, 0.0
        
        # Lem Loss
        norm_semantic_features = F.normalize(semantic_features, p=2, dim=0)
        language_feature_indices = index_decoder(norm_semantic_features.unsqueeze(0)).squeeze(0)
        gt_language_feature_indices = viewpoint_cam.language_feature_indices.permute(2, 0, 1)
        
        semantic_loss = compute_semantic_loss(language_feature_indices, gt_language_feature_indices, uncertainty, ce)
        
        uncertainty_loss = torch.mean(-torch.log(1-uncertainty))
        
        # Recon Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image.unsqueeze(0), gt_image.unsqueeze(0)))
        
        total_loss = dataset.reconstruction_loss_weight * loss \
                    + dataset.semantic_loss_weight * semantic_loss \
                    + dataset.uncertainty_loss_weight * uncertainty_loss \
                    + dataset.xyzmlp_loss_weight * xyz_mlp_loss \
                    + dataset.smooth_loss_weight * smooth_loss
        
        total_loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, viewpoint_cam.is_novel_view, 
                            color_map,
                            Ll1, loss, semantic_loss, l1_loss, xyz_mlp_loss, smooth_loss, uncertainty_loss,
                            iter_start.elapsed_time(iter_end), testing_iterations, 
                            scene, index_decoder, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                index_decoder.eval()
                scene.save(iteration, index_decoder, color_map)
                index_decoder.train()

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                decoder_optim.step()
                decoder_optim.zero_grad(set_to_none = True)
                xyz_decoder_optim.step()
                xyz_decoder_optim.zero_grad(set_to_none = True)
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                torch.save(index_decoder.state_dict(), scene.model_path + "/index_decoder_chkpnt" + str(iteration) + ".pth")
                torch.save(xyz_decoder.state_dict(), scene.model_path + "/xyz_decoder_chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(dataset, opt, pipe):
    if not dataset.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        dataset.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(dataset.model_path))
    os.makedirs(dataset.model_path, exist_ok = True)
    # The original log method
    with open(os.path.join(dataset.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(dataset))))
    # Save all the params
    with open(os.path.join(dataset.model_path, "cfg_args.yml"), 'w') as cfg_log_f:
        args = {
            "ModelParams" : vars(dataset),
            "PipelineParams" : vars(pipe),
            "OptimizationParams" : vars(opt),
        }
        yaml.dump(args, cfg_log_f)
    
    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(dataset.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, is_novel_view,
                    color_maps,
                    Ll1, loss, semantic_loss, l1_loss, xyz_mlp_loss, smooth_loss, uncertainty_loss,
                    elapsed, testing_iterations, 
                    scene : Scene, index_decoder, renderFunc, renderArgs):
    global FIRST_REPORT
    
    if tb_writer:
        if not is_novel_view:
            tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
            tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/semantic_loss', semantic_loss.item(), iteration)
        if smooth_loss > 0.0:
            tb_writer.add_scalar('train_loss_patches/smooth_loss', smooth_loss.item(), iteration)
        if xyz_mlp_loss > 0.0:
            tb_writer.add_scalar('train_loss_patches/xyz_mlp_loss', xyz_mlp_loss.item(), iteration)
        if uncertainty_loss > 0.0:
            tb_writer.add_scalar('train_loss_patches/uncertainty_loss', uncertainty_loss.item(), iteration)
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
                semantic_loss_test = 0.0
                uncertainty_test = 0.0
                
                # mkdir for test rendering
                if config['name'] == "test":
                    if FIRST_REPORT:
                        os.makedirs(os.path.join(scene.model_path, "renders", f"{iteration}_gt_images"), exist_ok=True)
                        os.makedirs(os.path.join(scene.model_path, "renders", f"{iteration}_gt_indices"), exist_ok=True)
                    os.makedirs(os.path.join(scene.model_path, "renders", f"{iteration}_images"), exist_ok=True)
                    os.makedirs(os.path.join(scene.model_path, "renders", f"{iteration}_indices"), exist_ok=True)
                
                for idx, viewpoint in enumerate(config['cameras']):
                    render_result = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_result["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    pca_feat_image = pca(F.normalize(render_result["semantic_features"], p=2, dim=0)) # already in [0.0, 1.0]
                    decoded_clip_feat_indices = index_decoder(F.normalize(render_result["semantic_features"], p=2, dim=0).unsqueeze(0))

                    # color_maps: (128, 3)
                    temp = 1   # ->0 = argmax, ->+inf = unifrom
                    prob_tensor_1 = torch.softmax(decoded_clip_feat_indices / temp, dim=1)  # (N, C=128, H, W)
                    feat_indices_image_1 = torch.einsum('nchw,ck->nkhw', prob_tensor_1, color_maps.to(prob_tensor_1.device))  # (N, 3, H, W)
                    
                    temp = 0.01   # ->0 = argmax, ->+inf = unifrom
                    prob_tensor_001 = torch.softmax(decoded_clip_feat_indices / temp, dim=1)  # (N, C=128, H, W)
                    feat_indices_image_001 = torch.einsum('nchw,ck->nkhw', prob_tensor_001, color_maps.to(prob_tensor_001.device))  # (N, 3, H, W)
                    
                    uncertainty = torch.clamp(render_result["uncertainty"], 0.0, 1.0)
                    
                    if tb_writer and (idx < 50):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render_uncertainty".format(viewpoint.image_name), uncertainty[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render_indices_softmax".format(viewpoint.image_name), feat_indices_image_1, global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render_indices".format(viewpoint.image_name), feat_indices_image_001, global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render_indices_feat_pca3".format(viewpoint.image_name), pca_feat_image[None], global_step=iteration)
                        if FIRST_REPORT:
                            gt_indices = F.embedding(viewpoint.language_feature_indices, color_maps.to(viewpoint.language_feature_indices.device)).squeeze().permute(2, 0, 1)
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth_indices".format(viewpoint.image_name), gt_indices[None], global_step=iteration)
                            upsampled_gt_indices = F.interpolate(gt_indices.unsqueeze(0), size=(gt_image.shape[1], gt_image.shape[2]), mode='nearest')
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth_upsampled_indices".format(viewpoint.image_name), upsampled_gt_indices, global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    semantic_loss_test += compute_semantic_loss(
                        decoded_clip_feat_indices.squeeze(0), 
                        viewpoint.language_feature_indices.to("cuda").permute(2, 0, 1), 
                        render_result["uncertainty"], 
                        torch.nn.CrossEntropyLoss(reduction='none')).double()
                    uncertainty_test += torch.mean(render_result["uncertainty"]).double()
                    torch.cuda.empty_cache()
                
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                semantic_loss_test /= len(config['cameras'])
                uncertainty_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} semantic_loss {} uncert {}"
                      .format(iteration, config['name'], l1_test, psnr_test, semantic_loss_test, uncertainty_test))
                
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - semantic_loss', semantic_loss_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - uncertainty', uncertainty_test, iteration)

        if tb_writer:
            # tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()
        
        if FIRST_REPORT:
            FIRST_REPORT = False

if __name__ == "__main__":
    # Set up command line argument parser
    parser = configargparse.ArgParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    
    parser.add('--config', required=True, is_config_file=True, help='config file path')
    parser.add('--debug_from', type=int, default=-1)
    parser.add('--detect_anomaly', action='store_true', default=False)
    parser.add("--test_iterations", nargs="+", type=int, default=[0]+[i for i in range(0, 30_001, 10_000)])
    parser.add("--test_set", nargs="+", type=str, default=[])
    parser.add("--save_iterations", nargs="+", type=int, default=[i for i in range(0, 30_001, 10_000)])
    parser.add("--quiet", action="store_true")
    parser.add("--checkpoint_iterations", nargs="+", type=int, default=[i for i in range(0, 30_001, 10_000)])
    parser.add("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), 
             args.test_iterations, args.test_set,
             args.save_iterations, args.checkpoint_iterations,
             args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
