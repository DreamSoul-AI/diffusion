import torch
import torch.nn as nn
from torch.nn import functional as F
from .net import *
from .diffusion import get_alphas_sigmas, get_index_from_list


class XZero(nn.Module):
    def __init__(self, data_shape, hidden_size, target_size):
        super().__init__()
        self.data_shape = data_shape
        self.hidden_size = hidden_size
        self.target_size = target_size
        self.timestep_embed = FourierFeatures(1, 16)
        self.class_embed = nn.Embedding(self.target_size + 1, 4)
        self.net = net(data_shape, hidden_size)

    def forward(self, x_0, t, cond):
        x_pred = self.forward_diffusion_pass(x_0, t, cond)
        output_target = x_pred

        x_noisy, noise = self.forward_diffusion_sample(x_0, t)
        loss = F.mse_loss(x_noisy, x_pred)
        return output_target, loss

    def forward_diffusion_pass(self, x_0, t, cond):
        timestep_embed = expand_to_planes(self.timestep_embed(t[:, None]), x_0.shape)
        class_embed = expand_to_planes(self.class_embed(cond + 1), x_0.shape)
        pred = self.net(torch.cat([x_0, class_embed, timestep_embed], dim=1))
        return pred

    def forward_diffusion_sample(self, x_0, t):
        """
        Takes an image and a timestep as input and
        returns the noisy version of it
        """
        alphas, sigmas = get_alphas_sigmas(t)  # sigma: noise level

        betas = sigmas
        alphas = 1. - betas
        # Pre-calculate different terms for closed form
        noise = torch.randn_like(x_0, device=x_0.device)
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
            sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )
        # mean + variance
        output = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        return output, noise


def xzero(cfg):
    data_shape = cfg['data_shape']
    hidden_size = cfg['diffusion']['hidden_size']
    target_size = cfg['target_size']
    model = XZero(data_shape, hidden_size, target_size)
    return model
