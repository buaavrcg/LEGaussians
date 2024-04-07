import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from sklearn.decomposition import PCA

from PIL import Image
import colorsys

import open_clip

def pca(features):
    """
    Perform PCA on the given features and return the result.
    Args:
        features: (C, H, W) torch.Tensor
    Returns:
        (3, H, W) torch.Tensor
    """
    shape = features.shape
    
    np_features = features.permute(1,2,0).reshape(-1, shape[0]).cpu().numpy()
    pca = PCA(n_components=3)
    pca.fit(np_features)

    pca_features = pca.transform(np_features)
    pca_features = (pca_features - pca_features.min()) / (pca_features.max() - pca_features.min())

    pca_features = torch.from_numpy(pca_features).reshape(shape[1], shape[2], 3).permute(2, 0, 1) 
    
    return pca_features

def index_to_rgb_images(input_tensor, color_map):
    """
    Args:
        input_tensor (torch.Tensor): (B, H, W, 1)
        color_map (torch.Tensor): (N, 3)
    Returns:
        _type_: (B, H, W, 3)
    """
    index_tensor = input_tensor[:, :, :, 0].long()
    rgb_images = color_map[index_tensor]

    return rgb_images

def generate_colors(n):
    colors = []
    for i in range(n):
        hue = i / float(n)  # 在色轮上均匀分布的色调值
        rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)  # 将 HSV 色彩空间转换为 RGB
        color = torch.tensor(rgb)  # RGB 值在 [0, 1]
        colors.append(color)
    return torch.stack(colors, dim=0)

def read_codebook(path):
    return torch.load(path)['embedding.weight']

def index_to_featrues(indies_tensor, codebook):
    """
    Args:
        input_tensor (torch.Tensor): (B, H, W, 1)
        color_map (torch.Tensor): (N, x)
    Returns:
        _type_: (B, H, W, x)
    """
    index_tensor = indies_tensor[:, :, :, 0].long()
    features = codebook[index_tensor]

    return features

class CLIPRelevance:
    def __init__(self, device="cuda"):
        self.device = device
        model, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k')
        model.eval()
        self.model = model.to(self.device)
        self.tokenizer = open_clip.get_tokenizer('ViT-B-16')
        self.process = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize((224, 224)),
                    transforms.Normalize(
                        mean=[0.48145466, 0.4578275, 0.40821073],
                        std=[0.26862954, 0.26130258, 0.27577711],
                    ),
                ]
            )

        self.negatives = ("object", "things", "stuff", "texture")
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.negatives]).to(self.device)
            self.neg_embeds = self.model.encode_text(tok_phrases)
        self.neg_embeds /= self.neg_embeds.norm(dim=-1, keepdim=True)

    def encode_text(self, texts):
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(text) for text in texts]).to(self.device)
            embeds = self.model.encode_text(tok_phrases)
        embeds /= embeds.norm(dim=-1, keepdim=True)
        return embeds
    
    def encode_image(self, image):
        with torch.no_grad():
            embeds = self.model.encode_image(self.process(image).to(self.device)[None, ...])
        embeds /= embeds.norm(dim=-1, keepdim=True)
        return embeds

    def get_relevancy(self, embed: torch.Tensor, positive: str or Image, negatives=None, scale = 100) -> torch.Tensor:
        if isinstance(positive, str):
            # pos_embeds = self.encode_text([f"a photo of a {positive}"])
            pos_embeds = self.encode_text([f"{positive}"])
        else:
            pos_embeds = self.encode_image(positive)
        
        if negatives is not None:
            with torch.no_grad():
                tok_phrases = torch.cat([self.tokenizer(v) for v in negatives.items()]).to(self.device)
                out_neg_embeds = self.model.encode_text(tok_phrases)
            out_neg_embeds /= out_neg_embeds.norm(dim=-1, keepdim=True)
            phrases_embeds = torch.cat([pos_embeds, self.neg_embeds, out_neg_embeds], dim=0)
        else:
            phrases_embeds = torch.cat([pos_embeds, self.neg_embeds], dim=0)
        
        p = phrases_embeds.to(embed.dtype)  # phrases x 512

        # output = torch.matmul(embed, p.T)  # hw x phrases
        # output = F.cosine_similarity(embed[..., None, :], p[None, None, ...], dim=-1)  # hw x phrases
        output = self._cosine_sim(embed, p)
        
        positive_vals = output[..., :1]  # hw x 1
        negative_vals = output[..., 1:]  # hw x N_phrase
        repeated_pos = positive_vals.repeat(1, 1, len(self.negatives))  # hw x N_phrase

        sims = torch.stack((repeated_pos, negative_vals), dim=-1)  # hw x N-phrase x 2
        softmax = torch.softmax(scale*sims, dim=-1)  # hw x n-phrase x 2
        best_id = softmax[..., 0].argmin(dim=2)  # hw
        return torch.gather(softmax, 2, best_id[..., None, None].expand(best_id.shape[0], best_id.shape[1], len(self.negatives), 2))[
            :, :, 0, :
        ]

    def get_simlarity(self, embed: torch.Tensor, positive: str or Image) -> torch.Tensor:
        if isinstance(positive, str):
            pos_embeds = self.encode_text([positive])
        else:
            pos_embeds = self.encode_image(positive)
        sim = F.cosine_similarity(embed, pos_embeds[None, ...], dim=-1)
        return sim
    
    def _cosine_sim(self, a, b):
        a_norm = torch.norm(a, dim=-1)[...,None]
        b_norm = torch.norm(b, dim=-1)[None,...]
        d = torch.matmul(a, b.t()) / (torch.matmul(a_norm, b_norm) + 1e-6)
        return d

class Embedding(nn.Module):
    def __init__(self, in_channels, N_freqs, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super(Embedding, self).__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels*(len(self.funcs)*N_freqs+1)

        if logscale:
            self.freq_bands = 2**torch.linspace(0, N_freqs-1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2**(N_freqs-1), N_freqs)

    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...) 
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12

        Inputs:
            x: (B, self.in_channels)

        Outputs:
            out: (B, self.out_channels)
        """
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq*x)]

        return torch.cat(out, -1)