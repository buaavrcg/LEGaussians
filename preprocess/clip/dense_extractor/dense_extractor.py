import os
import json
import torch
import random

from dataclasses import dataclass
from pathlib import Path
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

import open_clip
from scipy.stats import skewnorm

to_tensor = transforms.Compose([
    transforms.ToTensor()
])

to_pil = transforms.ToPILImage()

def get_image_paths(directory):
    image_paths = []
    valid_image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp"]

    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in valid_image_extensions):
                image_paths.append(os.path.join(root, file))

    return sorted(image_paths)

@dataclass
class DenseCLIPExtractorParams():
    resolution: float = 1000
    extract_type: str = "prob_random"
    exp: float = 2
    feat_scale: float = 0.1
    patch_scale: tuple = (0.05, 0.2)
    hw_ratio: tuple = (0.5, 2.0)
    interv_n: int = 20
    epoch: int = 3000
    draw_rele: bool = False
    device: str = "cuda"
    weight_mode: str = ""

class DenseCLIPExtractor:
    def __init__(self, path: str, args: DenseCLIPExtractorParams = DenseCLIPExtractorParams()):
        self.resolution = args.resolution
        self.extract_type = args.extract_type
        self.exp = args.exp
        self.feat_scale = args.feat_scale
        self.patch_scale = args.patch_scale
        self.hw_ratio = args.hw_ratio
        self.interv_n = args.interv_n
        self.epoch = args.epoch
        self.draw_rele = args.draw_rele
        self.device = args.device
        self.weight_mode = args.weight_mode
        
        print(f"[DenseCLIPExtractor] : {self.weight_mode}")
        print(f"[DenseCLIPExtractor] : {self.patch_scale}")
        print(f"[DenseCLIPExtractor] : {self.interv_n}")

        self.path = Path(path)
        res_str = str(self.resolution).replace(".", "")
        feat_scale_str = str(self.feat_scale).replace(".", "")
        patch_scale_str = str(self.patch_scale[0]).replace(".", "") + str(self.patch_scale[1]).replace(".", "")
        hw_ratio_str = str(self.hw_ratio[0]).replace(".", "") + str(self.hw_ratio[1]).replace(".", "")
        exp_str = str(self.exp).replace(".", "")
        self.params_name = f"r{res_str}_{self.weight_mode}square_{self.extract_type}{exp_str}_{self.epoch}_{feat_scale_str}_{patch_scale_str}_{hw_ratio_str}_{self.interv_n}"
        self.cache_path = Path(os.path.join(path, "dense_clipfeat_cache_weight", 
                                            self.params_name))

        self.cfg = dict(
                resolution = self.resolution,
                extract_type = self.extract_type,
                size = self.feat_scale,
                scale = self.patch_scale,
                hw_ratio = self.hw_ratio,
                exp = self.exp,
                interv_n = self.interv_n,
                epoch = self.epoch,
                draw_rele = self.draw_rele,
                device = self.device
            )

        self.try_load()
    
    def _get_model(self):
        model, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k')
        model.eval()
        model = model.to(self.device)
        process = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.Normalize(
                        mean=[0.48145466, 0.4578275, 0.40821073],
                        std=[0.26862954, 0.26130258, 0.27577711],
                    ),
                ]
            )
        return model, process

    def _get_images(self):
        self.image_paths = get_image_paths(self.path)
        num = len(self.image_paths)
        
        # the resize method is the same as the one in gaussian splatting
        image = Image.open(self.image_paths[0])
        orig_w, orig_h = image.size
        if self.resolution in [1, 2, 4, 8]:
            resolution = round(orig_w/self.resolution), round(orig_h/self.resolution)
        else:  # should be a type that converts to float
            if self.resolution == -1:
                if orig_w > 1600:
                    global_down = orig_w / 1600
                else:
                    global_down = 1
                scale = float(global_down)
                resolution = (int(orig_w / scale), int(orig_h / scale))
            else:
                global_down = orig_w / self.resolution
                scale = float(global_down)
                resolution = (self.resolution, int(orig_h / scale))

        print(f"[DenseCLIPExtractor] Resize the images from {image.size} to {resolution}.")
        
        images = []
        for i in tqdm(range(num)):
            image = Image.open(self.image_paths[i]).resize(resolution)
            images.append(to_tensor(image))
        images = torch.stack(images, dim = 0)
        
        return resolution, images.to(self.device)
    
    def _random_extract(self):
        model, process = self._get_model()
        
        resolution, images = self._get_images()
        num = images.shape[0]

        feat_res = (int(self.feat_scale * resolution[0]),
                    int(self.feat_scale * resolution[1]))

        data = torch.zeros((num, 512, feat_res[1], feat_res[0]), dtype=torch.float32, device=self.device)
        sum_times = torch.zeros((num, 1, feat_res[1], feat_res[0]), dtype=torch.float32, device=self.device)

        for i in tqdm(range(self.epoch)):
            scale = random.uniform(self.patch_scale[0], self.patch_scale[1])
            patch_w, patch_h = int(resolution[0] * scale), int(resolution[1] * scale) # patch的宽和高
            feat_w, feat_h = int(patch_w * self.feat_scale), int(patch_h * self.feat_scale) # feature的宽和高
            xs = torch.randint(0, resolution[0] - patch_w, size=(num,)) # 图片中像素点的横坐标
            ys = torch.randint(0, resolution[1] - patch_h, size=(num,)) # 图片中像素点的纵坐标
            for j in range(num):
                cropped_images = images[j, :, ys[j]:ys[j]+patch_h, xs[j]:xs[j]+patch_w].unsqueeze(0)
                with torch.no_grad():
                    feats = model.encode_image(process(cropped_images).to(self.device))
                feats = feats[..., None, None].expand(-1, -1, feat_h, feat_w).squeeze(0)
                feat_xs = int(xs[j] * self.feat_scale) #feature map中feature的横坐标
                feat_ys = int(ys[j] * self.feat_scale) #feature map中feature的纵坐标
                data[j, :, feat_ys:feat_ys+feat_h, feat_xs:feat_xs+feat_w] += feats 
                sum_times[j, :, feat_ys:feat_ys+feat_h, feat_xs:feat_xs+feat_w] += 1
        
        self.sums = sum_times
        self.data = data / (sum_times + 1e-6)

    def _prob_extract(self):
        model, process = self._get_model()
        
        resolution, images = self._get_images()
        num = images.shape[0]

        feat_res = (int(self.feat_scale * resolution[0]), int(self.feat_scale * resolution[1]))
        
        self.data = torch.zeros((num, 512, feat_res[1], feat_res[0]), dtype=torch.float16, device=self.device)
        self.sums = torch.zeros((num, 1, feat_res[1], feat_res[0]), dtype=torch.float16, device=self.device)
        
        a, b = self.patch_scale
        interval_lengths = [(b - a) / self.interv_n] * self.interv_n

        # the mean scale of each interval
        interval_means = [a + (i + 0.5) * interval_lengths[i] for i in range(self.interv_n)]
        probabilities = [1 / (mean**2) for mean in interval_means]

        # Normlization
        total_probability = sum(probabilities)
        normalized_probabilities = [p / total_probability for p in probabilities]
        
        sample_counts = [0] * self.interv_n
        
        for i in tqdm(range(self.epoch)):
            # choose an interval according to the probability distribution
            selected_interval = random.choices(range(self.interv_n), normalized_probabilities)[0]
            # choose a scale in the selected interval
            scale_w = random.uniform(a + selected_interval * interval_lengths[0],
                                    a + (selected_interval + 1) * interval_lengths[0])
            # scale_h = scale_w * random.uniform(self.hw_ratio[0], self.hw_ratio[1])
            scale_h = random.uniform(a + selected_interval * interval_lengths[0],
                                    a + (selected_interval + 1) * interval_lengths[0])
            
            patch_w, patch_h = int(resolution[0] * scale_w), int(resolution[1] * scale_h) # patch的宽和高
            feat_w, feat_h = int(patch_w * self.feat_scale), int(patch_h * self.feat_scale) # feature的宽和高
            
            x = torch.randint(0, resolution[0] - patch_w, size=(1,))[0] # 图片中像素点的横坐标
            y = torch.randint(0, resolution[1] - patch_h, size=(1,))[0] # 图片中像素点的纵坐标
            
            cropped_images = images[:, :, y:y+patch_h, x:x+patch_w]
            with torch.no_grad():
                feats = model.encode_image(process(cropped_images).to(self.device))
            feats = feats[..., None, None].expand(-1, -1, feat_h, feat_w)
            feat_x = int(x * self.feat_scale) #feature map中feature的横坐标
            feat_y = int(y * self.feat_scale) #feature map中feature的纵坐标
            if self.weight_mode == "avg":
                self.data[:, :, feat_y:feat_y+feat_h, feat_x:feat_x+feat_w] += (feats)
                self.sums[:, :, feat_y:feat_y+feat_h, feat_x:feat_x+feat_w] += (1)
            elif self.weight_mode == "h+w":
                self.data[:, :, feat_y:feat_y+feat_h, feat_x:feat_x+feat_w] += (feats / (scale_w + scale_h))
                self.sums[:, :, feat_y:feat_y+feat_h, feat_x:feat_x+feat_w] += (1 / (scale_w + scale_h))
            sample_counts[selected_interval] += 1
            
        for i in range(self.interv_n):
            print(f"interval {i + 1}: sample times {sample_counts[i]}")
        
        # we do it out of the class to save memory
        self.data = self.data / (self.sums + 1e-6)

    def _gaussian_prob_extract(self):
        model, process = self._get_model()

        resolution, images = self._get_images()
        num = images.shape[0]

        feat_res = (int(self.feat_scale * resolution[0]),
                    int(self.feat_scale * resolution[1]))

        self.data = torch.zeros((num, 512, feat_res[1], feat_res[0]),
                                dtype=torch.float16,
                                device=self.device)
        self.sums = torch.zeros((num, 1, feat_res[1], feat_res[0]),
                                dtype=torch.float16,
                                device=self.device)

        a, b = self.patch_scale
        a = -.3
        mean = 0.2
        std_dev = .08

        sample_counts = [0] * self.interv_n
        scale_w_candidate = [i for i in skewnorm.rvs(a, loc=mean, scale=std_dev, size=self.epoch) if i > 0.01 and i < 1]

        for i in tqdm(range(len(scale_w_candidate))):
            scale_w = scale_w_candidate[i]
            scale_h = scale_w * random.uniform(self.hw_ratio[0],
                                               self.hw_ratio[1])

            patch_w, patch_h = int(resolution[0] * scale_w), int(resolution[1] * scale_h)
            feat_w, feat_h = int(patch_w * self.feat_scale), int(patch_h * self.feat_scale)

            x = torch.randint(0, resolution[0] - patch_w, size=(1, ))[0]
            y = torch.randint(0, resolution[1] - patch_h, size=(1, ))[0]

            cropped_images = images[:, :, y:y + patch_h, x:x + patch_w]
            with torch.no_grad():
                feats = model.encode_image(
                    process(cropped_images).to(self.device))
            feats = feats[..., None, None].expand(-1, -1, feat_h, feat_w)
            feat_x = int(x * self.feat_scale)
            feat_y = int(y * self.feat_scale)
            self.data[:, :, feat_y:feat_y + feat_h,
                      feat_x:feat_x + feat_w] += (feats / (scale_w ** scale_h))
            self.sums[:, :, feat_y:feat_y + feat_h,
                      feat_x:feat_x + feat_w] += (1 / (scale_w ** scale_h))

            # sample_counts[selected_interval] += 1

        # for i in range(self.interv_n):
        #     print(f"interval {i + 1}: sample times {sample_counts[i]}")

        self.data = self.data / (self.sums + 1e-6)

    def _large_prob_extract(self):
        model, process = self._get_model()
        
        resolution, images = self._get_images()
        num = images.shape[0]

        feat_res = (int(self.feat_scale * resolution[0]), int(self.feat_scale * resolution[1]))
        
        self.data = torch.zeros((num, 512, feat_res[1], feat_res[0]), dtype=torch.float16, device="cpu")
        self.sums = torch.zeros((num, 1, feat_res[1], feat_res[0]), dtype=torch.float16, device="cpu")
        
        a, b = self.patch_scale
        interval_lengths = [(b - a) / self.interv_n] * self.interv_n

        # the mean scale of each interval
        interval_means = [a + (i + 0.5) * interval_lengths[i] for i in range(self.interv_n)]
        probabilities = [1 / mean for mean in interval_means]

        # Normlization
        total_probability = sum(probabilities)
        normalized_probabilities = [p / total_probability for p in probabilities]
        
        sample_counts = [0] * self.interv_n
        
        pins = [0, num//2, num]
        
        for i in tqdm(range(len(pins)-1)):
            data = torch.zeros((pins[i+1] - pins[i], 512, feat_res[1], feat_res[0]), dtype=torch.float16, device=self.device)
            sums = torch.zeros((pins[i+1] - pins[i], 1, feat_res[1], feat_res[0]), dtype=torch.float16, device=self.device)
            for j in tqdm(range(self.epoch)):
                # choose an interval according to the probability distribution
                selected_interval = random.choices(range(self.interv_n), normalized_probabilities)[0]
                # choose a scale in the selected interval
                scale_w = random.uniform(a + selected_interval * interval_lengths[0],
                                        a + (selected_interval + 1) * interval_lengths[0])
                scale_h = scale_w * random.uniform(self.hw_ratio[0], self.hw_ratio[1])
                
                patch_w, patch_h = int(resolution[0] * scale_w), int(resolution[1] * scale_h)
                feat_w, feat_h = int(patch_w * self.feat_scale), int(patch_h * self.feat_scale)
                
                x = torch.randint(0, resolution[0] - patch_w, size=(1,))[0]
                y = torch.randint(0, resolution[1] - patch_h, size=(1,))[0]
                
                cropped_images = images[pins[i]:pins[i+1], :, y:y+patch_h, x:x+patch_w]
                with torch.no_grad():
                    feats = model.encode_image(process(cropped_images))
                
                feats = feats[..., None, None].expand(-1, -1, feat_h, feat_w)
                feat_x = int(x * self.feat_scale)
                feat_y = int(y * self.feat_scale)
                data[:, :, feat_y:feat_y+feat_h, feat_x:feat_x+feat_w] += (feats / (scale_w ** self.exp))
                sums[:, :, feat_y:feat_y+feat_h, feat_x:feat_x+feat_w] += (1 / (scale_w ** self.exp))
                
                sample_counts[selected_interval] += 1
            
            self.data[pins[i]:pins[i+1]] = data.to(self.data.device)
            self.sums[pins[i]:pins[i+1]] = sums.to(self.sums.device)
            
        for i in range(self.interv_n):
            print(f"interval {i + 1}: sample times {sample_counts[i]}")
        
        self.data = self.data / (self.sums + 1e-6)

    def get_data_sums(self):
        return self.data, self.sums
    
    def extract(self):
        if self.extract_type == "random":
            self._random_extract()
        elif self.extract_type == "prob_random":
            self._prob_extract()
        elif self.extract_type == "gaussian_prob_random":
            self._gaussian_prob_extract()

    def load(self):
        cache_info_path = self.cache_path.with_suffix(".info")
        if not cache_info_path.exists():
            raise FileNotFoundError
        # with open(cache_info_path, "r") as f:
        #     cfg = json.loads(f.read())
        # if cfg != self.cfg:
        #     raise ValueError("Config mismatch")
        self.data = torch.load(self.cache_path.with_suffix(".pt")).to(self.device)
        self.sums = torch.load(self.cache_path.with_suffix(".sums.pt")).to(self.device)

    def save(self):
        os.makedirs(self.cache_path.parent, exist_ok=True)
        cache_info_path = self.cache_path.with_suffix(".info")
        with open(cache_info_path, "w") as f:
            f.write(json.dumps(self.cfg))
        torch.save(self.data, self.cache_path.with_suffix(".pt"))
        print(f"Data saved in {self.cache_path}")
        torch.save(self.sums, self.cache_path.with_suffix(".sums.pt"))
        print(f"Sums saved in {self.cache_path}")

    def try_load(self):
        try:
            print(f"[DenseCLIPExtractor] Trying to load {self.params_name}...")
            self.load()
            print(f"[DenseCLIPExtractor] {self.params_name} loaded. Feature shape is {self.data.shape}")
        except (FileNotFoundError, ValueError):
            print("[DenseCLIPExtractor] Loading failed. Extracting ...")
            self.extract()
            print(f"[DenseCLIPExtractor] Extraction done. Feature shape is {self.data.shape}")
            print("[DenseCLIPExtractor] Saving ...")
            self.save()
            print(f"[DenseCLIPExtractor] {self.cache_path} saved.")


# Slow
class PicLevelDenseCLIPExtractor:
    def __init__(self, path: str, args: DenseCLIPExtractorParams = DenseCLIPExtractorParams()):
        self.resolution = args.resolution
        self.extract_type = args.extract_type
        self.exp = args.exp
        self.feat_scale = args.feat_scale
        self.patch_scale = args.patch_scale
        self.hw_ratio = args.hw_ratio
        self.interv_n = args.interv_n
        self.epoch = args.epoch
        self.draw_rele = args.draw_rele
        self.device = args.device

        self.path = Path(path)
        res_str = str(self.resolution).replace(".", "")
        feat_scale_str = str(self.feat_scale).replace(".", "")
        patch_scale_str = str(self.patch_scale[0]).replace(".", "") + str(self.patch_scale[1]).replace(".", "")
        hw_ratio_str = str(self.hw_ratio[0]).replace(".", "") + str(self.hw_ratio[1]).replace(".", "")
        exp_str = str(self.exp).replace(".", "")
        self.params_name = f"r{res_str}_{self.extract_type}{exp_str}_{self.epoch}_{feat_scale_str}_{patch_scale_str}_{hw_ratio_str}_{self.interv_n}"
        self.cache_path = Path(os.path.join(path, "piclevel_dense_clipfeat_cache_weight", 
                                            self.params_name))

        self.cfg = dict(
                resolution = self.resolution,
                extract_type = self.extract_type,
                size = self.feat_scale,
                scale = self.patch_scale,
                hw_ratio = self.hw_ratio,
                exp = self.exp,
                interv_n = self.interv_n,
                epoch = self.epoch,
                draw_rele = self.draw_rele,
                device = self.device
            )

        self.try_load()
    
    def _get_model(self):
        model, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k')
        model.eval()
        model = model.to(self.device)
        process = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.Normalize(
                        mean=[0.48145466, 0.4578275, 0.40821073],
                        std=[0.26862954, 0.26130258, 0.27577711],
                    ),
                ]
            )
        return model, process

    def _get_images_paths(self):
        self.image_paths = get_image_paths(self.path)
        num = len(self.image_paths)
        
        # the resize method is the same as the one in gaussian splatting
        image = Image.open(self.image_paths[0])
        orig_w, orig_h = image.size
        if self.resolution in [1, 2, 4, 8]:
            resolution = round(orig_w/self.resolution), round(orig_h/self.resolution)
        else:  # should be a type that converts to float
            if self.resolution == -1:
                if orig_w > 1600:
                    global_down = orig_w / 1600
                else:
                    global_down = 1
                scale = float(global_down)
                resolution = (int(orig_w / scale), int(orig_h / scale))
            else:
                global_down = orig_w / self.resolution
                scale = float(global_down)
                resolution = (self.resolution, int(orig_h / scale))

        print(f"[DenseCLIPExtractor] Resize the images from {image.size} to {resolution}.")
        
        return resolution

    def _prob_extract(self):
        model, process = self._get_model()
        
        resolution = self._get_images_paths()
        num = len(self.image_paths)

        feat_res = (int(self.feat_scale * resolution[0]), int(self.feat_scale * resolution[1]))
        
        a, b = self.patch_scale
        interval_lengths = [(b - a) / self.interv_n] * self.interv_n

        # the mean scale of each interval
        interval_means = [a + (i + 0.5) * interval_lengths[i] for i in range(self.interv_n)]
        probabilities = [1 / mean for mean in interval_means]

        # Normlization
        total_probability = sum(probabilities)
        normalized_probabilities = [p / total_probability for p in probabilities]
        
        sample_counts = [0] * self.interv_n
        
        self.feat_cache_info = {}
        os.makedirs(self.cache_path, exist_ok=True)
        for i in tqdm(range(num)):
            image = to_tensor(Image.open(self.image_paths[i]).resize(resolution))
            data = torch.zeros((1, 512, feat_res[1], feat_res[0]), dtype=torch.float16, device=self.device)
            sums = torch.zeros((1, 1, feat_res[1], feat_res[0]), dtype=torch.float16, device=self.device)
            
            for j in tqdm(range(self.epoch)):
                # choose an interval according to the probability distribution
                selected_interval = random.choices(range(self.interv_n), normalized_probabilities)[0]
                # choose a scale in the selected interval
                scale_w = random.uniform(a + selected_interval * interval_lengths[0],
                                        a + (selected_interval + 1) * interval_lengths[0])
                scale_h = scale_w * random.uniform(self.hw_ratio[0], self.hw_ratio[1])
                
                patch_w, patch_h = int(resolution[0] * scale_w), int(resolution[1] * scale_h) # patch的宽和高
                feat_w, feat_h = int(patch_w * self.feat_scale), int(patch_h * self.feat_scale) # feature的宽和高
                
                x = torch.randint(0, resolution[0] - patch_w, size=(1,))[0] # 图片中像素点的横坐标
                y = torch.randint(0, resolution[1] - patch_h, size=(1,))[0] # 图片中像素点的纵坐标
                
                cropped_images = image[:, y:y+patch_h, x:x+patch_w][None, ...]
                with torch.no_grad():
                    feats = model.encode_image(process(cropped_images).to(self.device))
                feats = feats[..., None, None].expand(-1, -1, feat_h, feat_w)
                feat_x = int(x * self.feat_scale) #feature map中feature的横坐标
                feat_y = int(y * self.feat_scale) #feature map中feature的纵坐标
                data[:, :, feat_y:feat_y+feat_h, feat_x:feat_x+feat_w] += (feats / (scale_w ** self.exp))
                sums[:, :, feat_y:feat_y+feat_h, feat_x:feat_x+feat_w] += (1 / (scale_w ** self.exp))
                
                sample_counts[selected_interval] += 1
            
            image_name = self.image_paths[i].split("/")[-1].split(".")[0]
            cache_path = self.cache_path / f"{image_name}.pt"
            self.feat_cache_info[i] = cache_path
            torch.save(data / sums, cache_path)
            print(f"[DenseCLIPExtractor] {cache_path}.pt saved.")
        
        self.feat_cache_info["feature shape"] = data.shape
        
        for i in range(self.interv_n):
            print(f"interval {i + 1}: sample times {sample_counts[i]}")
        
    def extract(self):
        if self.extract_type is "prob_random":
            self._prob_extract()
        else:
            raise ValueError(f"Can't handle {self.extract_type} extraction type.")

    def load(self):
        cache_info_path = self.cache_path.with_suffix(".info")
        if not cache_info_path.exists():
            raise FileNotFoundError
        with open(cache_info_path, "r") as f:
            cache_info = json.loads(f.read())
        if cache_info["cfg"] != self.cfg:
            raise ValueError("Config mismatch")
        self.feat_cache_info = cache_info["cache_paths"]

    def save(self):
        os.makedirs(self.cache_path.parent, exist_ok=True)
        cache_info_path = self.cache_path.with_suffix(".info")
        with open(cache_info_path, "w") as f:
            f.write(json.dumps({"info": self.cfg, "cache_paths": self.feat_cache_info}))

    def try_load(self):
        try:
            print(f"[DenseCLIPExtractor] Trying to load {self.params_name} info...")
            self.load()
            print(f"[DenseCLIPExtractor] {self.params_name} info loaded. Feature shape is {self.feat_cache_info['feature shape']}")
        except (FileNotFoundError, ValueError):
            print("[DenseCLIPExtractor] Loading failed. Extracting ...")
            self.extract()
            print(f"[DenseCLIPExtractor] Extraction done. Feature shape is {self.feat_cache_info['feature shape']}")
            print("[DenseCLIPExtractor] Saving info ...")
            self.save()
            print(f"[DenseCLIPExtractor] {self.cache_path} info saved.")


if __name__ == "__main__":
    path = "data/360/room/images"
    # extractor_args = DenseCLIPExtractorParams(feat_scale = 0.3)
    # extractor = PicLevelDenseCLIPExtractor(path, extractor_args)

    extractor_args = DenseCLIPExtractorParams()
    extractor = DenseCLIPExtractor(path, extractor_args)
