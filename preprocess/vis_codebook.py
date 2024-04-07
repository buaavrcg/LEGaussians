import io
import os
import sys
from pathlib import Path
sys.path.append('../')

import torch
import torch.nn.functional as F
import numpy as np

from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from utils.lem_utils import *
from clip.clip_utils import get_image_paths, to_pil


def generate_colors(n):
    colors = []
    for i in range(n):
        hue = i / float(n)  # 在色轮上均匀分布的色调值
        rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)  # 将 HSV 色彩空间转换为 RGB
        color = torch.tensor(rgb)  # RGB 值在 [0, 1]
        colors.append(color)
    return torch.stack(colors, dim=0)


def draw_pil_images(images, images_per_row=12):
    """Draw PIL images

    Args:
        images (list): a list of PIL images list, each image has the same size
        images_per_row (int, optional): the num of images in a row. Defaults to 12.

    Returns:
        Image: ...
    """
    width, height = images[0][0].size

    total_width = images_per_row * width
    total_height = ((len(images[0]) * len(images) - 1) // images_per_row + 1) * height

    new_image = Image.new('RGB', (total_width, total_height))

    x_offset = 0
    y_offset = 0
    for i in range(len(images[0])):
        for j in range(len(images)):
            new_image.paste(images[j][i], (x_offset, y_offset))
            x_offset += width
        if (i + 1) * len(images) % images_per_row == 0:
            x_offset = 0
            y_offset += height

    return new_image


def draw_codebook_similarity_heatmap(codebook, output_dir="."):
    codebook_norm = codebook / codebook.norm(dim=1, keepdim=True)
    S = torch.mm(codebook_norm, codebook_norm.t())
    S = torch.clamp(S, -1, 1)

    print(S.shape)
    print(S)

    plt.figure(figsize=(10, 8))
    sns.heatmap(S.cpu().numpy(), cmap="viridis")  # 选择你喜欢的colormap
    plt.title("Similarity Heatmap")
    plt.xlabel("Vector Index")
    plt.ylabel("Vector Index")
    plt.savefig(output_dir + "/similarity_heatmap.png", format='png', dpi=300, bbox_inches='tight')


def draw_indices_histogram(indices, output_dir="."):
    flatten_indices = indices.cpu().numpy().flatten()
    unique_numbers, counts = np.unique(flatten_indices, return_counts=True)
    for number, count in zip(unique_numbers, counts):
        print(f"index: {number}, times: {count}")
    print(f"The num of unique_numbers: {len(unique_numbers)}")

    plt.figure(figsize=(10, 6))
    counts, bins, patches = plt.hist(flatten_indices, bins=np.arange(flatten_indices.min(), flatten_indices.max()+2)-0.5, edgecolor='black', rwidth=0.8)

    plt.xticks(bins[:-1])
    plt.title("Histogram of Values")
    plt.xlabel("Value")
    plt.ylabel("Frequency")

    # 添加文本
    for count, bin_start, patch in zip(counts, bins, patches):
        height = patch.get_height()
        width = patch.get_x() + patch.get_width()/2
        plt.text(width, height + 5, f"{int(bin_start)}", ha='center', va='center')

    plt.savefig(output_dir + "/histogram.png", format='png', dpi=300, bbox_inches='tight')


def draw_rele_distrib(gt_rele, codebook_rele, kde=True):
    gt = gt_rele.view(-1).to("cpu").numpy()
    cb = codebook_rele.view(-1).to("cpu").numpy()
    n_cb = (cb - cb.min()) / (cb.max() - cb.min()) 

    plt.figure()
    if kde:
        sns.kdeplot(gt, color='blue', label='gt')
        sns.kdeplot(cb, color='orange', label='cb')
        sns.kdeplot(n_cb, color='red', label='n_cb')
    else:
        plt.hist(gt, bins=30, color='blue', alpha=0.5, label='gt')
        plt.hist(cb, bins=30, color='orange', alpha=0.5, label='cb')
        plt.hist(n_cb, bins=30, color='red', alpha=0.5, label='n_cb')
    plt.legend(loc='upper right')
    
    # create a file-like object from the figure, to convert to PIL
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    
    plt.close()
    
    return img


def prepare_images(image_paths, dense_clip_data, dense_clip_sums, codebook, indices, 
                  re, positive_query, scale,
                  draw_list = None, num = 10, rele_distrib = True):
    images = []
    index_images = []
    gt_reles_images = []
    codebook_reles_images = []
    gt_sims_images = []
    codebook_sims_images = []
    rele_distribs_images = []
    sim_distribs_images = []
    
    
    draw_list = draw_list if draw_list else list(range(num))
    for i in draw_list:
        image = Image.open(image_paths[i])
        
        size = (image.size[0] // 5, image.size[1] // 5)
        image = image.resize(size)
        
        index_image = F.embedding(indices[i], colormap).squeeze().permute(2, 0, 1)
        index_image = to_pil(index_image).resize(size)
        
        gt_rele = re.get_relevancy(dense_clip_data[i], positive_query, scale=scale)[..., 0]
        gt_rele_image = to_pil(gt_rele).resize(size)
        codebook_rele = re.get_relevancy(F.embedding(indices[i], codebook).squeeze(), positive_query, scale=scale)[..., 0]
        codebook_rele_image = to_pil(codebook_rele).resize(size)
        
        gt_sim = re.get_simlarity(dense_clip_data[i], positive_query)
        gt_sim_image = to_pil(gt_sim).resize(size)
        codebook_sim = re.get_simlarity(F.embedding(indices[i], codebook).squeeze(), positive_query)
        codebook_sim_image = to_pil(codebook_sim).resize(size)
        
        
        images.append(image)
        index_images.append(index_image)
        gt_reles_images.append(gt_rele_image)
        codebook_reles_images.append(codebook_rele_image)
        gt_sims_images.append(gt_sim_image)
        codebook_sims_images.append(codebook_sim_image)
        
        
        if rele_distrib:
            rele_distrib_image = draw_rele_distrib(gt_rele, codebook_rele).resize(size)
            rele_distribs_images.append(rele_distrib_image)
            sim_distrib_image = draw_rele_distrib(gt_sim, codebook_sim).resize(size)
            sim_distribs_images.append(sim_distrib_image)
    
    
    if rele_distrib:
        return [images, index_images, gt_reles_images, codebook_reles_images, rele_distribs_images, gt_sims_images, codebook_sims_images, sim_distribs_images]
    else:
        return [images, index_images, gt_reles_images, codebook_reles_images, gt_sims_images, codebook_sims_images]


def prepare_index_images(image_paths,indices, colormap,
                  draw_list = None, num = 10):
    images = []
    index_images = []
    blendeds = []
    
    draw_list = draw_list if draw_list else list(range(num))
    for i in draw_list:
        image = Image.open(image_paths[i])
        
        size = (image.size[0] // 10, image.size[1] // 10)
        image = image.resize(size)
        
        index_image = F.embedding(indices[i], colormap).squeeze().permute(2, 0, 1)
        index_image = to_pil(index_image).resize(size)
        
        blended = Image.blend(image, index_image, 0.5)
        
        images.append(image)
        index_images.append(index_image)
        blendeds.append(blended)

    return [images, index_images, blendeds]


if __name__ == "__main__":
    
    mip_scene_names = ["room", "bicycle", "kitchen", "garden", "counter", "bonsai"]
    scene_name = mip_scene_names[1]
    
    for scene_name in tqdm(["counter"]):
    
        image_path = Path(f"data/360/{scene_name}/images")
        
        params_name = "r1000_prob_random2_3000_03_00502_0520_20"
        dense_clip_pth = image_path / f"large_dense_clipfeat_cache_weight/{params_name}"
        
        codebook_params_name = "test_large_diffscl02_clip_dino05_128_896_1"
        codebook_pth = image_path / f"{codebook_params_name}_codebook.pt"
        indices_pth = image_path / f"{codebook_params_name}_encoding_indices.pt"
        
        image_paths = get_image_paths(image_path)
        dense_clip_data = torch.load(dense_clip_pth.with_suffix(".pt")).permute(0, 2, 3, 1) # [N, 66, 100, 512]
        dense_clip_sums = torch.load(dense_clip_pth.with_suffix(".sums.pt")).permute(0, 2, 3, 1) # [N, 66, 100, 1]
        codebook = read_codebook(codebook_pth)[:, :512] # [128, 512]
        indices = torch.load(indices_pth).to("cuda") # [N, 66, 100, 1]
        
        colormap = generate_colors(128).to("cuda") # [128, 3]
        output_path = Path("./runs/vis/360") / f"{scene_name}" / params_name / codebook_params_name
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)
            
        re = CLIPRelevance()
        
        positive_texts = {
            "room": ["blue grey chair", "silver gray curtain", "brown shoes", "yellow books", "deep dark green carpets", "green Yucca plant", "Family Portrait Print", "wine glasses and bottles", "yellow wood floors", "piano keyboard", "white wood door", "black loud speakers", "windows", "wood desk"],
            
            "bicycle": ["green grass", "bicycle", "tire", "bench", "Asphalt ground", "Silver Oak Tree"],
            
            "kitchen": ["LEGO Technic 856 Bulldozer", "Basket Weave Cloth", "Wood plat", "old pink striped cloth", "Red Oven Gloves", "wood chair"],
            
            "garden": ["football", "wood table", "green grass", "wood pot", "dried flowers", "elderflower", "green plant", "bricks wall", "doors", "windows", "Hexagonal Stone ground"],
            
            "counter": ["Jar of coconut oil", "fruit oranges", "onions", "plants", "blue Oven Gloves","Wood Rolling Pin", "free range eggs box", "gold Ripple Baking Pan", "Napolina Tomatoes", "knife", "black granite texture plat", "Garofalo pasta", "stale bread"],
            
            "bonsai": ["piano keyboard", "bicycle", "purple table cloth", "black stool", "plastic bonsai tree", "dark grey patterned carpet"],
        }
        
        draw_lists = {
            "bicycel": None,
            "room": [i for i in range(0, 190, 10)],
            "bonsai": None,
            "counter": None,
            "kitchen": None,
            "garden": None,
        }

        test_positive_texts = [""]
        
        images = prepare_index_images(image_paths, indices, colormap, num = len(image_paths))
        for i in range(len(images[1])):
            images[1][i].save(output_path / f"{scene_name}_index{i}.png")
