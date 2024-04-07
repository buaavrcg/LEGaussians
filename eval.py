import os
import json

import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as tf

from pathlib import Path
from PIL import Image
from tqdm import tqdm
import configargparse

# Reconstruction metric
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
from utils.image_utils import psnr

def is_pic(fname):
    if fname.split(".")[-1] in ["JPG", "jpg", "png"]:
        return True
    return False

def read_images(renders_dir, gt_dir):
    
    renders = []
    gts = []
    image_names = []
    
    for fname in sorted(os.listdir(renders_dir)):
        if not is_pic(fname):
                continue
        render = Image.open(renders_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :])
        image_names.append(fname)
    
    for fname in sorted(os.listdir(gt_dir)):
        if not is_pic(fname):
                continue
        gt = Image.open(gt_dir / fname).resize(render.size)
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :])

    return renders, gts, image_names

def read_3dovs_masks(renders_dir, gt_dir):
    
    renders = {}
    gts = {}
    names = {}
    
    with open(renders_dir / "texts_dict.json", "r") as f:
        texts_dict = json.load(f)
    
    for pic_dir in os.listdir(gt_dir):
        
        if not os.path.isdir(os.path.join(gt_dir, pic_dir)):
            continue
        
        render_masks = {}
        gt_masks = {}
        image_names = []
        
        for fname in os.listdir(gt_dir / pic_dir):
            if not is_pic(fname):
                continue
            if fname.split(".")[0] not in texts_dict.keys():
                continue
            text = fname.split(".")[0]
            render = Image.open(renders_dir / pic_dir / fname)
            gt = Image.open(gt_dir / pic_dir / fname).resize(render.size)
            render_masks[text] = tf.to_tensor(render)[:1, :, :]
            gt_masks[text] = tf.to_tensor(gt)[:1, :, :]
            image_names.append(text)
        
        renders[pic_dir] = render_masks
        gts[pic_dir] = gt_masks
        names[pic_dir] = image_names
        
    return renders, gts, names, texts_dict # renders and gts: (C=1, H, W)


def mean_iou(mask1, mask2):
    intersection = torch.logical_and(mask1, mask2).float().sum()
    union = torch.logical_or(mask1, mask2).float().sum()
    iou = intersection / (union + 1e-6)  # Adding a small value to avoid division by zero
    return iou

def accuracy(mask1, mask2):
    correct_predictions = torch.eq(mask1, mask2).float().sum()
    total_pixels = mask1.numel()
    accuracy = correct_predictions / total_pixels
    return accuracy

def precision(mask1, mask2):
    tp = torch.logical_and(mask1, mask2).float().sum()  # True positives
    fp = torch.logical_and(mask1, 1-mask2).float().sum()  # False positives
    precision_value = tp / (tp + fp + 1e-6)  # Adding a small value to avoid division by zero
    return precision_value

def recall(mask1, mask2):
    tp = torch.logical_and(mask1, mask2).float().sum()
    fn = torch.logical_and(1 - mask1, mask2).float().sum()
    recall_value = tp / (tp + fn + 1e-6)
    return recall_value


def mAP_evaluate(texts_dict, relevancy_dir, gt_dir, json_pth=None):
    threshold_values = np.arange(0.0, 1.01, 0.01)   
    picture_AP_list = [] 
    picture_AP_dic = {}
    for pic_dir in tqdm(os.listdir(gt_dir), desc="mAP evaluation progress"):
        class_AP_list = []
        class_AP_dic = {}
        if not os.path.isdir(os.path.join(gt_dir, pic_dir)):
            continue
        for fname in os.listdir(gt_dir / pic_dir):
            if not is_pic(fname):
                continue
            if fname.split(".")[0] not in texts_dict.keys():
                continue
            text = fname.split(".")[0]
            recall_list = []
            precision_list = []
            
            render = np.load(relevancy_dir / pic_dir / Path("array") / Path(str(fname).split('.')[0] + ".npy"))
            h, w = render.shape[0], render.shape[1]
            gt = np.array(Image.open(gt_dir / pic_dir / fname).resize((w, h)))
            
            render = tf.to_tensor(render)
            gt = tf.to_tensor(gt)
            for threshold in threshold_values:
                msk = (render > threshold).long()
                precision_value = precision(msk, gt)
                recall_value = recall(msk, gt)
                recall_list.append(recall_value)
                precision_list.append(precision_value)
            interpolated_recall_levels = np.arange(0.0, 1.01, 0.01)
            AP = 0
            precision_list = np.array(precision_list)
            recall_list = np.array(recall_list)
            for r in interpolated_recall_levels:
                precisions_at_recall_level = precision_list[recall_list >= r]
                if len(precisions_at_recall_level) > 0:
                    interpolated_precision = np.max(precisions_at_recall_level)
                else:
                    interpolated_precision = 0
                AP += interpolated_precision
            AP /= 100
            class_AP_list.append(AP)
            class_AP_dic[text] = AP
        picture_AP = np.mean(class_AP_list)
        picture_AP_list.append(picture_AP)
        picture_AP_dic[pic_dir] = {"pic_mAP":picture_AP, "class_AP":class_AP_dic}
    mAP = np.mean(picture_AP_list)
    if json_pth:
        with open(json_pth, "w") as f:
            json.dump({"mAP": mAP, "detail": picture_AP_dic}, f, indent=4)
    print("  mAP : {:>12.7f}".format(mAP, ".5"))

    return mAP


def rec_evaluate(renders, gts):

    ssims = []
    psnrs = []
    lpipss = []

    for idx in tqdm(range(len(renders)), desc="Reconstruction metric evaluation progress"):
        ssims.append(ssim(renders[idx], gts[idx]))
        psnrs.append(psnr(renders[idx], gts[idx]))
        lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))

    print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
    print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
    print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))

def lem_evaluate(renders, gts, json_pth=None):
    # renders and gts: dict (keys are picture name) of dict (keys are text), value is mask of (C=1, H, W)
    IoUs = {}
    accuracies = {}
    precisions = {}
    
    IoUs_list = []
    accuracies_list = []
    precisions_list = []
    
    for image_name in tqdm(renders.keys(), desc="Language embedding metric evaluation progress"):
        
        image_ious = {}
        image_accs = {}
        image_precs = {}
        
        image_ious_list = []
        image_accs_list = []
        image_precs_list = []
        
        for text in renders[image_name].keys():
            render = renders[image_name][text]
            gt = gts[image_name][text]

            image_ious[text] = mean_iou(render, gt).item()
            image_accs[text] = accuracy(render, gt).item()
            image_precs[text] = precision(render, gt).item()
            
            image_ious_list.append(image_ious[text])
            image_accs_list.append(image_accs[text])
            image_precs_list.append(image_precs[text])
        
        IoUs[image_name] = image_ious
        accuracies[image_name] = image_accs
        precisions[image_name] = image_precs
        
        IoUs_list.append(np.mean(image_ious_list))
        accuracies_list.append(np.mean(image_accs_list))
        precisions_list.append(np.mean(image_precs_list))
    
    print("  mIoU : {:>12.7f}".format(np.mean(IoUs_list), ".5"))
    print("  accuracy : {:>12.7f}".format(np.mean(accuracies_list), ".5"))
    print("  precision : {:>12.7f}".format(np.mean(precisions_list), ".5"))
    
    if json_pth:
        with open(json_pth, "w") as f:
            json.dump({"IoUs": IoUs, "accuracies": accuracies, "precisions": precisions}, f, indent=4)
    

if __name__ == "__main__":
    # device = torch.device("cuda:0")
    # torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = configargparse.ArgParser(description="Training script parameters")
    parser.add_argument('--path', '-p', type=str, default="")
    parser.add_argument("--texts", nargs="+", type=str, default=[])
    args = parser.parse_args()
    
    eval_dir = Path(args.path)
    
    # Recon
    recon_renders_dir = eval_dir / "pred_images"
    recon_gt_dir = eval_dir / "gt_images"
    renders, gts, _ = read_images(recon_renders_dir, recon_gt_dir)
    rec_evaluate(renders, gts)

    # Lem
    lem_renders_dir = eval_dir / f"pred_segs"
    lem_gt_dir = eval_dir / "segmentations"
    renders, gts, names, texts_dict = read_3dovs_masks(lem_renders_dir, lem_gt_dir)
    
    lem_evaluate(renders, gts, eval_dir / "lem_metrics.json")
    mAP_evaluate(texts_dict, eval_dir / "relevancy", eval_dir / "segmentations", eval_dir / "mAP_metrics.json")
    
    
    
