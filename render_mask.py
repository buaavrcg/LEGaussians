import io
import os
import json
import sys
import torch
import torch.nn.functional as F
import torchvision

import matplotlib.pyplot as plt

from PIL import Image
import seaborn as sns
import numpy as np

from scene import Scene, GaussianModel
from scene.index_decoder import *
from gaussian_renderer import render

from utils.lem_utils import *
from utils.general_utils import safe_state

import configargparse
from arguments import ModelParams, PipelineParams, OptimizationParams

def draw_rele_distrib(rele, kde=True):
    rele = rele.view(-1).detach().to("cpu").numpy()
    
    plt.figure()
    if kde:
        sns.kdeplot(rele, color='blue', label='rele')
    else:
        plt.hist(rele, bins=30, color='blue', alpha=0.5, label='rele')
    plt.legend(loc='upper right')
    
    # create a file-like object from the figure, to convert to PIL
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    
    plt.close()

    return img

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
    
    eval_name = f"open_new_eval_{com_type}_s{scale}_a{str(a).replace('.', '')}"
    gt_images_pth = f"{ouptut_dir}/{eval_name}/gt_images"
    pred_images_pth = f"{ouptut_dir}/{eval_name}/pred_images"
    pred_segs_pth = f"{ouptut_dir}/{eval_name}/pred_segs"
    rele_pth = f"{ouptut_dir}/{eval_name}/relevancy"
    
    os.makedirs(gt_images_pth, exist_ok=True)
    os.makedirs(pred_images_pth, exist_ok=True)
    os.makedirs(pred_segs_pth, exist_ok=True)
    os.makedirs(rele_pth, exist_ok=True)
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    viewpoint_stack = scene.getTestCameras().copy()
    for cam in viewpoint_stack:
        render_pkg = render(cam, gaussians, pipe, background)
        
        gt_image = cam.original_image
        image = render_pkg["render"].detach()
        
        torchvision.utils.save_image(gt_image, f"{gt_images_pth}/{cam.image_name}.png")
        torchvision.utils.save_image(image, f"{pred_images_pth}/{cam.image_name}.png")
        
        os.makedirs(f"{pred_segs_pth}/{cam.image_name}", exist_ok=True)
        os.makedirs(f"{pred_segs_pth}/{cam.image_name}/distr", exist_ok=True)
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
            
            # norm
            # rele = (rele - rele.min()) / (rele.max() - rele.min()) 
            rele_distr_img = draw_rele_distrib(rele)
            
            msk = (rele >= a)
            
            np.save(f"{rele_pth}/{cam.image_name}/array/{text}.npy", rele.detach().cpu().numpy())
            torchvision.utils.save_image(rele, f"{rele_pth}/{cam.image_name}/images/{text}.png")
            torchvision.utils.save_image(msk.float(), f"{pred_segs_pth}/{cam.image_name}/{text}.png")
            rele_distr_img.save(f"{pred_segs_pth}/{cam.image_name}/distr/{text}.png")
            
            seg_indices[msk] = i

        with open(f"{pred_segs_pth}/texts_dict.json", "w") as f:
            json.dump(texts_dict, f, indent=4)
        
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

    # All done
    print("Rendering done.")