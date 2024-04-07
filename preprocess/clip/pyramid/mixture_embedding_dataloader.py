import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .feature_dataloader import FeatureDataloader
from .patch_embedding_dataloader import PatchEmbeddingDataloader
from .image_encoder import BaseImageEncoder



class MixtureEmbeddingDataloader(FeatureDataloader):
    def __init__(
        self,
        cfg: dict,
        device: torch.device,
        model: BaseImageEncoder,
        process,
        image_list: torch.Tensor = None,
        cache_path: str = None,
    ):
        assert "tile_size_range" in cfg
        assert "tile_size_res" in cfg
        assert "stride_scaler" in cfg
        assert "image_shape" in cfg
        assert "model_name" in cfg

        self.tile_sizes = torch.linspace(*cfg["tile_size_range"], cfg["tile_size_res"]).to(device)
        self.strider_scaler_list = [self._stride_scaler(tr.item(), cfg["stride_scaler"]) for tr in self.tile_sizes]

        self.model = model
        self.process = process
        # self.embed_size = self.model.embedding_dim
        self.embed_size = 512
        self.data_dict = {}
        self.data = None
        super().__init__(cfg, device, image_list, cache_path)

    def __call__(self):
        """
            return patch level clip feature mixture.
        """
        return self.data

    def _stride_scaler(self, tile_ratio, stride_scaler):
        return np.interp(tile_ratio, [0.05, 0.15], [1.0, stride_scaler])

    def load(self):
        # don't create anything, PatchEmbeddingDataloader will create itself
        cache_info_path = self.cache_path.with_suffix(".info")
        mix_cache_path = self.cache_path.with_suffix(".npy")

        # check if cache exists
        if not cache_info_path.exists():
            raise FileNotFoundError

        # if config is different, remove all cached content
        with open(cache_info_path, "r") as f:
            cfg = json.loads(f.read())
        if cfg != self.cfg:
            for f in os.listdir(self.cache_path):
                os.remove(os.path.join(self.cache_path, f))
            raise ValueError("Config mismatch")

        # load mixture
        self.data = torch.from_numpy(np.load(mix_cache_path)).half()

    def create(self, image_list):
        os.makedirs(self.cache_path, exist_ok=True)
        for i, tr in enumerate(tqdm(self.tile_sizes, desc="Scales")):
            stride_scaler = self.strider_scaler_list[i]
            self.data_dict[i] = PatchEmbeddingDataloader(
                cfg={
                    "tile_ratio": tr.item(),
                    "stride_ratio": stride_scaler,
                    "image_shape": self.cfg["image_shape"],
                    "model_name": self.cfg["model_name"],
                },
                device=self.device,
                model=self.model,
                process=self.process,
                image_list=image_list,
                cache_path=Path(f"{self.cache_path}/level_{i}.npy"),
            )
        # create mixture
        self._create_mixture()


    def save(self):
        cache_info_path = self.cache_path.with_suffix(".info")
        with open(cache_info_path, "w") as f:
            f.write(json.dumps(self.cfg))
        # don't save PatchEmbeddingDataloader, PatchEmbeddingDataloader will save itself
        # save mixture
        np.save(self.cache_path.with_suffix(".npy"), self.data)

    def _create_mixture(self):
        mix_feat = self.data_dict[0].data.detach().clone().permute(0, 3, 1, 2).float()
        _, _, a, b = mix_feat.shape
        for i in range(1, len(self.tile_sizes) - 1):
            feat = self.data_dict[i].data.permute(0, 3, 1, 2).float()
            feat_interp = F.interpolate(feat, size=(a, b), mode="nearest")
            mix_feat += feat_interp
        self.data = (mix_feat.permute(0, 2, 3, 1) / len(self.tile_sizes)).half()