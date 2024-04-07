import os
import sys
import torch
import torch.nn.functional as F
import torchvision

import numpy as np

from scene import Scene, GaussianModel
from scene.index_decoder import *
from gaussian_renderer import render
from scene.cameras import Camera

from utils.lem_utils import *
from utils.general_utils import safe_state

import configargparse
from arguments import ModelParams, PipelineParams, OptimizationParams

def interpolate_camera_path(positions, rotations, steps):
    """
    Interpolate a camera path given start and end positions and rotations.
    
    :param positions: List of two 3D positions (start and end), each a list or array of length 3.
    :param rotations: List of two 3x3 rotation matrices (start and end).
    :param steps: Number of steps for interpolation.
    :return: A tuple of two lists: interpolated positions and interpolated rotation matrices.
    """
    from scipy.spatial.transform import Rotation as R
    def lerp(start, end, t):
        """ Linear interpolation between start and end """
        return start + t * (end - start)

    def slerp(quat0, quat1, t):
        """ Spherical linear interpolation between two quaternions """
        # Compute the cosine of the angle between the two vectors
        dot = np.dot(quat0, quat1)

        # If the dot product is negative, slerp won't take the shorter path
        # So we negate one of the inputs
        if dot < 0.0:
            quat1 = -quat1
            dot = -dot

        # Clamp the value to be in the range of Acos
        # This may be necessary due to numerical imprecision
        dot = np.clip(dot, -1.0, 1.0)

        # Calculate coefficients
        theta = np.arccos(dot) * t
        quat2 = quat1 - quat0 * dot
        quat2 = quat2 / np.linalg.norm(quat2)

        # Interpolation
        result = quat0 * np.cos(theta) + quat2 * np.sin(theta)
        return result

    # The interpolate_camera_path function remains the same
    # interpolate_camera_path(positions, rotations, steps)

    start_position, end_position = positions
    start_rotation, end_rotation = rotations

    # Convert start and end rotations to scipy Rotation objects
    start_rotation = R.from_matrix(start_rotation)
    end_rotation = R.from_matrix(end_rotation)

    # Interpolate positions
    interpolated_positions = [lerp(np.array(start_position), np.array(end_position), i / (steps - 1)) for i in range(steps)]
    
    # Convert rotations to quaternions for slerp
    start_quat = start_rotation.as_quat()
    end_quat = end_rotation.as_quat()

    # Interpolate rotations
    interpolated_rotations_quat = [slerp(start_quat, end_quat, i / (steps - 1)) for i in range(steps)]
    interpolated_rotations = [R.from_quat(q).as_matrix() for q in interpolated_rotations_quat]

    return interpolated_positions, interpolated_rotations

# # Example usage
# positions_example = [[0, 0, 0], [10, 10, 10]]
# rotations_example = [np.eye(3), np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])]
# steps_example = 10

# interpolated_positions_example, interpolated_rotations_example = interpolate_camera_path(positions_example, rotations_example, steps_example)
# interpolated_positions_example, interpolated_rotations_example  # Display the results

def generate_path(viewpoint_stack):
    viewpoint_stack = sorted(viewpoint_stack, key=lambda x: x.image_name)

    Ts = []
    Rs = []
    for v in viewpoint_stack:
        Ts.append(v.T.tolist())
        Rs.append(v.R.tolist())

    interpolated_positions, interpolated_rotations = interpolate_camera_path(Ts, Rs, 240)
    
    path = []
    cam0 = viewpoint_stack[0]
    for i in range(len(interpolated_positions)):
        path.append(Camera(colmap_id=cam0.colmap_id, R=interpolated_rotations[i], T=interpolated_positions[i], 
                  FoVx=cam0.FoVx, FoVy=cam0.FoVy, 
                  image=cam0.original_image, gt_alpha_mask=None, language_feature_indices=None,
                  image_name=f"{cam0.image_name}_{i:04}", uid=cam0.uid, data_device="cuda"))
        
    return path
    

def rendering_mask(dataset, opt, pipe, checkpoint, codebook_pth, test_set, texts_dict, a, scale, com_type, device="cuda"):
    gaussians = GaussianModel(dataset.sh_degree, dataset.semantic_features_dim, dataset.points_num_limit)
    scene = Scene(dataset, gaussians, test_set=test_set, is_test=True)
    index_decoder = IndexDecoder(dataset.semantic_features_dim, dataset.codebook_size).to(device)
    (model_params, first_iter) = torch.load(checkpoint)
    gaussians.restore(model_params, opt)
    index_decoder_ckpt = os.path.join(os.path.dirname(checkpoint), "index_decoder_" + os.path.basename(checkpoint))
    index_decoder.load_state_dict(torch.load(index_decoder_ckpt))
    codebook = read_codebook(codebook_pth)
    clip_rele = CLIPRelevance(device=device)
    
    ouptut_dir = os.path.dirname(checkpoint)

    eval_name = f"trajectory_{com_type}_s{scale}_a{str(a).replace('.', '')}"
    pred_images_pth = f"{ouptut_dir}/{eval_name}/pred_images"
    rele_pth = f"{ouptut_dir}/{eval_name}/relevancy"
    
    os.makedirs(pred_images_pth, exist_ok=True)
    os.makedirs(rele_pth, exist_ok=True)
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    viewpoint_stack = scene.getTestCameras().copy()
    path = generate_path(viewpoint_stack)
    for cam in path:
        render_pkg = render(cam, gaussians, pipe, background)
        
        image = render_pkg["render"].detach()
        
        torchvision.utils.save_image(image, f"{pred_images_pth}/{cam.image_name}.png")
        
        os.makedirs(f"{rele_pth}/{cam.image_name}/array", exist_ok=True)
        os.makedirs(f"{rele_pth}/{cam.image_name}/images", exist_ok=True)
        
        semantic_features = render_pkg["semantic_features"].detach()
        norm_semantic_features = F.normalize(semantic_features, p=2, dim=0)
        with torch.no_grad():
            indices = index_decoder(norm_semantic_features.unsqueeze(0))

        index_tensor = torch.argmax(indices, dim=1).squeeze()
        if com_type == "argmax":
            # argmax
            clip_features = F.embedding(index_tensor, codebook[:, :512])
        elif com_type == "softmax":
            temp = 1   # ->0 = argmax, ->+inf = unifrom
            prob_tensor = torch.softmax(indices / temp, dim=1).permute(0, 2, 3, 1)  # (N, C=128, H, W)
            clip_features = (prob_tensor @ codebook[:, :512]).squeeze()
        
        seg_indices = -1 * torch.ones_like(index_tensor)
        for i in range(len(list(texts_dict.keys()))):
            text = list(texts_dict.keys())[i]
            if type(texts_dict[text]) is list:
                rele0 = clip_rele.get_relevancy(clip_features, texts_dict[text][0], scale).squeeze()[..., 0]
                rele1 = clip_rele.get_relevancy(clip_features, texts_dict[text][1], scale).squeeze()[..., 0]
                rele = torch.logical_or((rele0 >= a).float(), (rele1 >= a).float())
            else:
                rele = clip_rele.get_relevancy(clip_features, texts_dict[text], negatives=None, scale=scale).squeeze()[..., 0]
            
            msk = (rele >= a)
            
            np.save(f"{rele_pth}/{cam.image_name}/array/{text}.npy", rele.detach().cpu().numpy())
            torchvision.utils.save_image(rele, f"{rele_pth}/{cam.image_name}/images/{text}.png")
            
            seg_indices[msk] = i
        
        torch.cuda.empty_cache()

    

if __name__ == "__main__":
    # Set up command line argument parser
    parser = configargparse.ArgParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    
    parser.add('--config', required=True, is_config_file=True, help='config file path')
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--mode", type=str, default="search", choices=["search"])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--codebook", type=str, default = None)
    parser.add_argument("--test_set", nargs="+", type=str, default=[])
    parser.add_argument("--texts", nargs="+", type=str, default=[])
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--scale", type=float, default=100)
    parser.add_argument("--com_type", type=str, default="")
    args = parser.parse_args(sys.argv[1:])
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(False)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    texts_dict = {}
    for i in range(len(args.texts)):
        texts_dict[args.texts[i]] = args.texts[i]
    
    rendering_mask(lp.extract(args), op.extract(args), pp.extract(args), 
            args.start_checkpoint, args.codebook,
            args.test_set, texts_dict, args.alpha, args.scale, args.com_type)

    print("Rendering complete.")