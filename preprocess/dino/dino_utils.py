import os
import cv2
import torch
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm

def get_image_paths(directory):
    image_paths = []
    valid_image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp"]

    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in valid_image_extensions):
                image_paths.append(os.path.join(root, file))

    return sorted(image_paths)

def visualize_dino(cache_file, save_path, dimension = 384):
    dinos = torch.load(cache_file)

    N, H, W = dinos.shape[0], dinos.shape[1], dinos.shape[2]

    pca = PCA(n_components=3)
    os.makedirs(save_path, exist_ok=True)

    dino_pca = dinos.detach().cpu().numpy()
    dino_pca = dino_pca.reshape([-1, dimension])

    component = pca.fit_transform(dino_pca)
    component = component.reshape([N, H, W, 3])
    component = ((component - component.min()) / (component.max() - component.min())).astype(np.float32) # normalize to [0,1]
    component *= 255.
    component = component.astype(np.uint8)
    for i in tqdm(range(N)):
        cv2.imwrite(save_path + f"{i}.png", component[i].squeeze())

if __name__ == "__main__":
    cache_file = "data/360/room/images/dino_descriptors_stride2.pt"
    save_path = "./dino_pics_room_s2/"
    visualize_dino(cache_file, save_path)