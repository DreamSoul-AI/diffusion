import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import yaml
from .model import *

# Use a low discrepancy quasi-random sequence to sample uniformly distributed
# timesteps. This considerably reduces the between-batch variance of the loss.
if 'cfg' not in globals():
    with open('config.yml', 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)


class ResidualBlock(nn.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input):
        return self.main(input) + self.skip(input)


class ResConvBlock(ResidualBlock):
    def __init__(self, c_in, c_mid, c_out, is_last=False):
        skip = None if c_in == c_out else nn.Conv2d(c_in, c_out, 1, bias=False)
        super().__init__([
            nn.Conv2d(c_in, c_mid, 3, padding=1),
            nn.Dropout2d(0.1, inplace=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_mid, c_out, 3, padding=1),
            nn.Dropout2d(0.1, inplace=True) if not is_last else nn.Identity(),
            nn.ReLU(inplace=True) if not is_last else nn.Identity(),
        ], skip)


class SelfAttention2d(nn.Module):
    def __init__(self, c_in, n_head=1, dropout_rate=0.1):
        super().__init__()
        assert c_in % n_head == 0
        self.norm = nn.GroupNorm(1, c_in)
        self.n_head = n_head
        self.qkv_proj = nn.Conv2d(c_in, c_in * 3, 1)
        self.out_proj = nn.Conv2d(c_in, c_in, 1)
        self.dropout = nn.Dropout2d(dropout_rate, inplace=True)

    def forward(self, input):
        n, c, h, w = input.shape
        qkv = self.qkv_proj(self.norm(input))
        qkv = qkv.view([n, self.n_head * 3, c // self.n_head, h * w]).transpose(2, 3)
        q, k, v = qkv.chunk(3, dim=1)
        scale = k.shape[3]**-0.25
        att = ((q * scale) @ (k.transpose(2, 3) * scale)).softmax(3)
        y = (att @ v).transpose(2, 3).contiguous().view([n, c, h, w])
        return input + self.dropout(self.out_proj(y))


class SkipBlock(nn.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input):
        return torch.cat([self.main(input), self.skip(input)], dim=1)


class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.):
        super().__init__()
        assert out_features % 2 == 0
        self.weight = nn.Parameter(torch.randn([out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)


def expand_to_planes(input, shape):
    return input[..., None, None].repeat([1, 1, shape[2], shape[3]])


class DiffusionEpsilon(nn.Module):
    def __init__(self):
        super().__init__()
        c = 64  # The base channel count

        self.timestep_embed = FourierFeatures(1, 16)
        self.class_embed = nn.Embedding(11, 4)

        self.net = nn.Sequential(   # 32x32
            ResConvBlock(3 + 16 + 4, c, c),
            ResConvBlock(c, c, c),
            SkipBlock([
                nn.AvgPool2d(2),  # 32x32 -> 16x16
                ResConvBlock(c, c * 2, c * 2),
                ResConvBlock(c * 2, c * 2, c * 2),
                SkipBlock([
                    nn.AvgPool2d(2),  # 16x16 -> 8x8
                    ResConvBlock(c * 2, c * 4, c * 4),
                    SelfAttention2d(c * 4, c * 4 // 64),
                    ResConvBlock(c * 4, c * 4, c * 4),
                    SelfAttention2d(c * 4, c * 4 // 64),
                    SkipBlock([
                        nn.AvgPool2d(2),  # 8x8 -> 4x4
                        ResConvBlock(c * 4, c * 8, c * 8),
                        SelfAttention2d(c * 8, c * 8 // 64),
                        ResConvBlock(c * 8, c * 8, c * 8),
                        SelfAttention2d(c * 8, c * 8 // 64),
                        ResConvBlock(c * 8, c * 8, c * 8),
                        SelfAttention2d(c * 8, c * 8 // 64),
                        ResConvBlock(c * 8, c * 8, c * 4),
                        SelfAttention2d(c * 4, c * 4 // 64),
                        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    ]),  # 4x4 -> 8x8
                    ResConvBlock(c * 8, c * 4, c * 4),
                    SelfAttention2d(c * 4, c * 4 // 64),
                    ResConvBlock(c * 4, c * 4, c * 2),
                    SelfAttention2d(c * 2, c * 2 // 64),
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                ]),  # 8x8 -> 16x16
                ResConvBlock(c * 4, c * 2, c * 2),
                ResConvBlock(c * 2, c * 2, c),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ]),  # 16x16 -> 32x32
            ResConvBlock(c * 2, c, c),
            ResConvBlock(c, c, 3, is_last=True),
        )

    def forward(self, x_0, t, cond):
        output = {}
        noise_pred = self.forward_diffusion_pass(x_0, t, cond)
        output['target'] = noise_pred

        x_noisy, noise = self.forward_diffusion_sample(x_0, t, cfg['device'])
        output['loss'] = F.mse_loss(noise, noise_pred)
        return output

    def forward_diffusion_pass(self, x_0, t, cond):
        timestep_embed = expand_to_planes(self.timestep_embed(t[:, None]), x_0.shape)
        class_embed = expand_to_planes(self.class_embed(cond + 1), x_0.shape)
        return self.net(torch.cat([x_0, class_embed, timestep_embed], dim=1))
    
    def forward_diffusion_sample(self, x_0, t, device="cpu"):
        """
        Takes an image and a timestep as input and
        returns the noisy version of it
        """
        alphas, sigmas = get_alphas_sigmas(t) # sigma: noise level

        betas = sigmas
        alphas = 1. - betas
        # Pre-calculate different terms for closed form
        noise = torch.randn_like(x_0)
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
        return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
        + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(cfg['device'])


def diffusionEpsilon(cfg):
    model = DiffusionEpsilon()
    model.apply(init_param)
    return model