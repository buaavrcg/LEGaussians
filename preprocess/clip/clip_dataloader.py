import os
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import open_clip

from clip.clip_utils import to_tensor, get_image_paths
from clip.pyramid.mixture_embedding_dataloader import MixtureEmbeddingDataloader
from clip.dense_extractor.dense_extractor import DenseCLIPExtractor, DenseCLIPExtractorParams



class DenseMixtureDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        
        self.extractor = DenseCLIPExtractor(path, DenseCLIPExtractorParams())
        
        # normalization
        data = self.extractor.data.permute(0, 2, 3, 1)
        norm_data = torch.norm(data, p=2, dim=-1, keepdim=True)
        data_normalized = data / norm_data
        
        self.data = data_normalized
        
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index]

class PyramidMixtureDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.device = "cuda"
        model, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k')
        model.eval()
        self.model = model.to(self.device)
        self.process = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.Normalize(
                        mean=[0.48145466, 0.4578275, 0.40821073],
                        std=[0.26862954, 0.26130258, 0.27577711],
                    ),
                ]
            )
        
        self.image_paths = get_image_paths(path)
        self.image_shape = to_tensor(Image.open(self.image_paths[0])).shape[1:3]

        self.cfg = {
                "tile_size_range": [0.05, 0.5],
                "tile_size_res": 7,
                "stride_scaler": 0.5,
                "image_shape": list(self.image_shape),
                "model_name": "ViT-B/16",
            }
                
        self.mixture_path = Path(os.path.join(path, f"pyramid_{self.cfg['tile_size_range'][0]}", "cache"))

        self._load_mixture()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index]

    def _get_images(self):
        image_list = []
        for image_path in tqdm(self.image_paths):
            image_list.append(to_tensor(Image.open(image_path)))
        return torch.stack(image_list)
    
    def _load_mixture(self):
        mixture_data = None
        if self.mixture_path.with_suffix(".npy").exists():
            mixture_data = MixtureEmbeddingDataloader(
                device = self.device,
                cfg = self.cfg,
                cache_path = self.mixture_path,
                model = self.model,
                process = self.process,
            )
        else:
            image_list = self._get_images()
            mixture_data = MixtureEmbeddingDataloader(
                device = self.device,
                cfg = self.cfg,
                image_list = image_list,
                cache_path = self.mixture_path,
                model = self.model,
                process = self.process,
            )
        self.data = mixture_data().to(self.device)