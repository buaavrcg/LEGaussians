"""
Borrowed from https://github.com/MishaLaskin/vqvae/blob/master/models/quantizer.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.

    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, n_e, e_dim, beta, device, concat = False, dino_weight = 0):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim).to(device)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        if device != None:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.concat = concat
        self.dino_weight = dino_weight


    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete 
        one-hot vector that is the index of the closest embedding vector e_j

        z (continuous) -> z_q (discrete)

        z.shape = (batch, height, width, channel)

        quantization pipeline:

            1. get encoder input (B,H,W,C)
            2. flatten input to (B*H*W,C)

        """
        # reshape z -> (batch, height, width, channel) and flatten
        
        z_flattened = z.view(-1, self.e_dim)
        
        assert not torch.isnan(z_flattened).any()
        
        cb_normalized = self._normalize_cb(self.embedding.weight)
        d = self._d(cb_normalized, z_flattened)
        
        assert not torch.isnan(cb_normalized).any()
        assert not torch.isnan(d).any()

        # find closest encodings
        min_encoding_indices = torch.argmax(d, dim=1).unsqueeze(1)
        encoding_indices_prob = torch.softmax(d, dim=1)
        
        assert not torch.isnan(min_encoding_indices).any()
        assert not torch.isnan(encoding_indices_prob).any()
        
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).to(self.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)
        
        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, cb_normalized).view(z.shape)
        assert not torch.isnan(z_q).any()
        
        # compute loss for embedding
        e_mean = torch.mean(min_encodings, dim=0)
        loss_kl = - torch.sum(e_mean * torch.log(1 / self.n_e / (e_mean + 1e-6)))
        loss, constrative_loss = self._loss(cb_normalized, min_encoding_indices, z_q, z)
        
        # preserve gradients
        z_q = z + (z_q - z).detach()
        
        # perplexity
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-6)))

        return loss, constrative_loss, loss_kl, encoding_indices_prob, d, z_q, perplexity, min_encodings, min_encoding_indices

    def _d(self, cb, z_flattened):
        if self.concat:
            d_clip = self._cosine_sim(cb[:, :512], z_flattened[:, :512])
            d_dino = self._cosine_sim(cb[:, 512:], z_flattened[:, 512:])
            d = d_clip + self.dino_weight * d_dino
        else:
            d = self._cosine_sim(cb, z_flattened)
        return d
    
    def _loss(self, cb, min_encoding_indices, z_q, z):
        loss = 0
        constrative_loss = 0

        if self.concat:
            z_q_clip = z_q[:, :, :, :512]
            z_q_dino = z_q[:, :, :, 512:]
            loss_cos_clip = (1 - torch.mean(torch.cosine_similarity(z_q_clip, z.detach()[:, :, :, :512], dim = -1))) \
                            +  self.beta * (1 - torch.mean(torch.cosine_similarity(z_q_clip.detach(), z[:, :, :, :512], dim = -1)))
            loss_cos_dino = (1 - torch.mean(torch.cosine_similarity(z_q_dino, z.detach()[:, :, :, 512:], dim = -1))) \
                            + self.beta * (1 - torch.mean(torch.cosine_similarity(z_q_dino.detach(), z[:, :, :, 512:], dim = -1)))
            loss += loss_cos_clip + self.dino_weight * loss_cos_dino
            # constrative loss
            cb_clip = cb[...,:512]
            cb_dino = cb[...,512:]
            cb_clip_cos = torch.cosine_similarity(cb_clip.unsqueeze(0), cb_clip.unsqueeze(1), dim = -1)
            cb_dino_cos = torch.cosine_similarity(cb_dino.unsqueeze(0), cb_dino.unsqueeze(1), dim = -1)
            cb_cos = cb_clip_cos + self.dino_weight * cb_dino_cos
            # mean of (cosine simlarity of (every feature and the other features))
            cb_neg = (torch.sum(cb_cos, dim=1) - cb_cos[0][0]) / (cb_cos.shape[0] - 1)
            x = F.embedding(min_encoding_indices, cb_neg[...,None]).squeeze()
            
            zq_clip_cos = torch.cosine_similarity(z_q_clip, z.detach()[:, :, :, :512], dim = -1).view(-1)
            zq_dino_cos = torch.cosine_similarity(z_q_dino, z.detach()[:, :, :, 512:], dim = -1).view(-1)
            zq_cos = zq_clip_cos + self.dino_weight * zq_dino_cos
            
            constrative_loss += torch.mean(-1 * zq_cos + x)
            
        else:
            loss += (1 - torch.mean(torch.cosine_similarity(z_q, z.detach(), dim = -1))) \
                    + self.beta * (1 - torch.mean(torch.cosine_similarity(z_q.detach(), z, dim = -1)))
            # TODO: add constrative loss
            constrative_loss += 0
        
        return loss, constrative_loss
    
    def _mse(self, embedding, z_flattened):
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
        torch.sum(embedding**2, dim=1) - 2 * \
        torch.matmul(z_flattened, embedding.t())
        return d
    
    def _cosine_sim(self, embedding, z_flattened):
        embedding_norm = torch.norm(embedding, dim=-1)[None, :]
        z_flattened_norm = torch.norm(z_flattened, dim=-1)[:, None]

        assert not torch.isnan(embedding).any()
        assert not torch.isnan(z_flattened).any()
        assert not torch.isnan(embedding_norm).any()

        assert not torch.isnan(z_flattened_norm).any()

        d = torch.matmul(z_flattened, embedding.t()) / (torch.matmul(z_flattened_norm, embedding_norm) + 1e-6)
        assert not torch.isnan(torch.matmul(z_flattened, embedding.t())).any()
        assert not torch.isnan(torch.matmul(z_flattened_norm, embedding_norm)).any()
        assert not torch.isnan(d).any()

        return d
    
    def _normalize_cb(self, cb):
        norm_cb_clip = torch.norm(cb[...,:512], p=2, dim=-1, keepdim=True)
        norm_cb_dino = torch.norm(cb[...,512:], p=2, dim=-1, keepdim=True)
        cb_clip_normalized = cb[..., :512] / norm_cb_clip
        cb_dino_normalized = cb[..., 512:] / norm_cb_dino
        cb_normalized = torch.cat((cb_clip_normalized, cb_dino_normalized), dim=-1)
        return cb_normalized