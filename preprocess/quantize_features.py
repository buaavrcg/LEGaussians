import os
import random
from tqdm import tqdm
from datetime import datetime   
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from quantizer import VectorQuantizer
from dino.dino_dataloader import DinoDataset
from clip.clip_dataloader import PyramidMixtureDataset, DenseMixtureDataset
from semantic_feature_dataloader import SematicFeatureDataset, PyramidSematicFeatureDataset
import sys
sys.path.append('..')
from utils.lem_utils import index_to_rgb_images, generate_colors
import configargparse

# Configuration
random.seed(0)

# Global writer initialization
writer = None

class Trainer:
    def __init__(self, args):
        self.args = args
        self.tensorboard_step = 0
        self.writer = None
        self.prefix = self.name_prefix()
        self.initialize_writer()

    def initialize_writer(self):
        writer_dir_base = os.path.join("runs", self.args.dataset, self.args.image_dir.split("/")[-2], self.prefix)
        writer_dir = writer_dir_base
        if os.path.exists(writer_dir):
            counter = 0
            while os.path.exists(writer_dir):
                counter += 1
                writer_dir = f"{writer_dir_base}_{counter}"
        self.writer = SummaryWriter(log_dir=writer_dir)

    def name_prefix(self):
        dino_w_str = str(self.args.dino_weight).replace(".", "")
        kl_beta_str = str(self.args.kl_beta).replace(".", "")
        min_p_str = str(self.args.min_p).replace(".", "")
        max_p_str = str(self.args.max_p).replace(".", "")
        # Format the current timestamp. For example: "20240331-235959"
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"{self.args.feat_type}{dino_w_str}_{self.args.weight_mode}{min_p_str}{max_p_str}_{self.args.n_e}_{self.args.e_dim}_{kl_beta_str}_{timestamp}"

    def select_dataset(self):
        dataset_cls = {
            'dino': DinoDataset,
            'pyrclip': PyramidMixtureDataset,
            'mixclip': DenseMixtureDataset,
            'clip_dino': SematicFeatureDataset,
            'pyrclip_dino': PyramidSematicFeatureDataset
        }.get(self.args.feat_type, DinoDataset)
        return dataset_cls(self.args.image_dir)

    def train(self):
        data_loader = DataLoader(self.select_dataset(), batch_size=self.args.batch_size, shuffle=self.args.shuffle)
        color_map = generate_colors(self.args.n_e)
        model, optimizer, scheduler = self.setup_training()

        model.train()
        for epoch in tqdm(range(self.args.epoch), dynamic_ncols=True):
            encoding_indices = []
            for feature in tqdm(data_loader, leave=False, dynamic_ncols=True):
                loss_cos, constrative_loss, loss_kl, encoding_indices_prob, d, z_q, perplexity, min_encodings, min_encoding_indices = model(feature)
                
                encoding_indices.append(min_encoding_indices.view(*feature.shape[:3], 1))
                
                flattened_encoding_indices = min_encoding_indices.view(-1)
                histogram = torch.histc(flattened_encoding_indices.float(), bins=args.n_e, min=0, max=args.n_e-1)
                num_elements = histogram.sum()
                frac = histogram / num_elements
                flattened_encoding_indices_prob = encoding_indices_prob.view(-1, args.n_e)
                load_balancing_loss = (frac * torch.mean(flattened_encoding_indices_prob, dim=0)).sum()
                
                loss_d = -1 * torch.log2(d.mean() if d.mean() > 0 else torch.tensor(1e-10))
                
                loss = loss_cos + args.load_balance_weight * load_balancing_loss

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            scheduler.step()

            metric_loss = 1 - torch.mean(torch.cosine_similarity(feature[...,:512].to("cuda"), z_q[...,:512], dim = -1))
            encoding_indices_tensor = torch.cat(encoding_indices, dim=0).to("cpu")

            self.write_tensorboard(metric_loss, loss, loss_cos, loss_kl, load_balancing_loss, d, loss_d, perplexity)
            if self.tensorboard_step % self.args.interv_n == 0:
                self.save_model(model, encoding_indices_tensor, color_map)
            self.tensorboard_step += 1
                

    def setup_training(self):
        concat = self.args.feat_type in ['clip_dino', 'pyrclip_dino']
        model = VectorQuantizer(self.args.n_e, self.args.e_dim, self.args.beta, self.args.device, concat=concat, dino_weight=self.args.dino_weight)
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        scheduler = LinearLR(optimizer, start_factor=1, end_factor=0.1, total_iters=1000)
        return model, optimizer, scheduler

    def save_model(self, model, encoding_indices, color_map):
        output_dir = self.args.output_codebook_dir if self.args.output_codebook_dir else self.args.image_dir
        torch.save(model.state_dict(), os.path.join(output_dir, f'{self.prefix}_codebook.pt'))
        torch.save(encoding_indices, os.path.join(output_dir, f'{self.prefix}_encoding_indices.pt'))
        for img_idx in range(0, 5):
            image = index_to_rgb_images(encoding_indices[img_idx].unsqueeze(0), color_map).permute(0, 3, 1, 2)[0]
            self.writer.add_image(f'encoding_image/pic{img_idx}', image, self.tensorboard_step)
        
    def write_tensorboard(self, metric_loss, loss, loss_cos, loss_kl, load_balancing_loss, d, loss_d, perplexity):
        # Example tensorboard writing function, extend as needed
        self.writer.add_scalar('loss/metric_loss', metric_loss.item(), self.tensorboard_step)
        self.writer.add_scalar('loss/loss', loss.item(), self.tensorboard_step)
        self.writer.add_scalar('loss/loss_cos', loss_cos.item(), self.tensorboard_step)
        self.writer.add_scalar('loss/loss_kl', loss_kl.item(), self.tensorboard_step)
        self.writer.add_scalar('loss/load_balancing_loss', load_balancing_loss.item(), self.tensorboard_step)
        self.writer.add_scalar('loss/d', d.mean().item(), self.tensorboard_step)
        self.writer.add_scalar('loss/loss_d', loss_d.item(), self.tensorboard_step)
        self.writer.add_scalar('loss/perplexity', perplexity.item(), self.tensorboard_step)

def parse_args():
    parser = configargparse.ArgParser(description="Training script parameters")
    parser.add_argument('--config', required=True, is_config_file=True, help='config file path')
    parser.add_argument('--image_dir', type=str, default=None)
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--output_codebook_dir', type=str, default=None)
    parser.add_argument('--base_codebook_path', type=str, default="")
    parser.add_argument('--feat_type', type=str, default='dino')
    parser.add_argument('--dino_weight', type=float, default=0.1)
    parser.add_argument('--load_balance_weight', type=float, default=1.0)
    parser.add_argument('--n_e', type=int, default=128)
    parser.add_argument('--e_dim', type=int, default=896) # 384, 512, 512 + 384 = 896
    parser.add_argument('--beta', type=float, default=0.)
    parser.add_argument('--kl_beta', type=float, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--shuffle', type=bool, default=False)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--interv_n', type=int, default=20)
    parser.add_argument('--min_p', type=float, default=0.0)
    parser.add_argument('--max_p', type=float, default=0.0)
    parser.add_argument('--weight_mode', type=str, default="")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    trainer = Trainer(args)
    trainer.train()
