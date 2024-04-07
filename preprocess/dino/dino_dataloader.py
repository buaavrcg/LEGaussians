from dataclasses import dataclass

import os
import torch
from tqdm import tqdm
from torch.utils.data import Dataset

from dino.extractor import ViTExtractor
from dino.dino_utils import get_image_paths

@dataclass
class DinoExtractorParams():
    model_type: str = 'dino_vits8'
    device: str = 'cuda'
    stride: int = 2
    load_size: int = 224
    layer: int = 11
    facet: str = 'key'
    bin: bool = False
    unflatten: bool = True

class DinoDataset(Dataset):
    def __init__(self, image_dir, extractor_params: DinoExtractorParams = DinoExtractorParams()):
        super().__init__()
        self.dino_descriptors = get_dinos(image_dir, extractor_params)
        
    def __len__(self):
        return self.dino_descriptors.shape[0]

    def __getitem__(self, index):
        return self.dino_descriptors[index]

def extract_feature_single(extractor, image_path, args):
    with torch.no_grad():
        image_batch, image_pil = extractor.preprocess(image_path, args.load_size)
        descriptors = extractor.extract_descriptors(image_batch.to(args.device), args.layer, args.facet, args.bin)
        descriptors = extractor.reshape_descriptors(descriptors)
    return descriptors

def extract_feature(extractor, image_paths, args):
    descriptors = []
    for image_path in tqdm(image_paths):
        descriptors.append(extract_feature_single(extractor, image_path, args))
    descriptors = torch.stack(descriptors, dim=0)
    return descriptors

def get_dinos(image_dir, args, half = True):
    cache_file = os.path.join(image_dir, 'dino_descriptors_stride2.pt')
    if os.path.exists(cache_file):
        print(f"[DinoExtractor] Trying to load ...")
        dinos = torch.load(cache_file)
        print(f"[DinoExtractor] Loaded ... Feature shape is {dinos.shape}")
        return dinos
    print("[DinoExtractor] Loading failed. Extracting ...")
    image_paths = get_image_paths(image_dir)
    extractor = ViTExtractor(args.model_type, args.stride, device=args.device)
    if not half:
        dinos = extract_feature(extractor, image_paths, args)
    else:
        dinos = []
        i = len(image_paths) // 2
        dinos.append(extract_feature(extractor, image_paths[:i], args).to('cpu'))
        torch.cuda.empty_cache()
        dinos.append(extract_feature(extractor, image_paths[i:], args).to('cpu'))
        torch.cuda.empty_cache()
        dinos = torch.cat(dinos, dim=0)
    print(f"[DinoExtractor] Extraction done. Feature shape is {dinos.shape}")
    print("[DinoExtractor] Saving ...")
    pt_path = os.path.join(image_dir, 'dino_descriptors_stride2.pt')
    torch.save(dinos, pt_path)
    print(f"[DinoExtractor] {pt_path} saved.")
    return dinos

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='data/imagenet')
    parser.add_argument('--model_type', type=str, default='dino_vits8')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--load_size', type=int, default=224)
    parser.add_argument('--layer', type=int, default=11)
    parser.add_argument('--facet', type=str, default='key')
    parser.add_argument('--bin', type=bool, default=False)
    args = parser.parse_args()

    descriptors = get_dinos(args.image_dir, args)
    