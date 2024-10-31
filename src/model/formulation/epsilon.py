import torch
import torch.nn as nn
from torch.nn import functional as F
from .diffusion import extract
from ..backbone import expand_to_planes, FourierFeatures


class Epsilon(nn.Module):
    def __init__(self, backbone, data_shape, hidden_size, target_size):
        super().__init__()
        self.data_shape = data_shape
        self.hidden_size = hidden_size
        self.target_size = target_size
        self.timestep_embed = FourierFeatures(1, 16)
        self.class_embed = nn.Embedding(self.target_size + 1, 4)
        self.backbone = backbone
        
        # Initialize beta schedule
        betas = torch.linspace(0.0001, 0.02, 100, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
        
        # Register buffers
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        
        # reverse process
        self.register_buffer("posterior_variance", betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer("posterior_log_variance_clipped", torch.log((betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)).clamp(min=1e-20)))
        # estimate x_0 given x_t
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))
        # posterior_mean_coef1
        self.register_buffer("posterior_mean_coef1", betas * torch.sqrt(alphas_cumprod_prev) / 1.0 - alphas_cumprod)
        self.register_buffer("posterior_mean_coef2", (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas_cumprod) / (1.0 - alphas_cumprod))
        
    def q_sample(self, x_start, t, noise):
        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
        return sample
    
    def forward(self, x_0, t, cond, training=True):
        """
        Handles both training and sampling
        """
        if training:
            x_recon = self.forward_diffusion_pass(x_0, t, cond)
            output_target = x_recon
    
            # Create noisy data
            x_noisy, noise = self.forward_diffusion_sample(x_0, t)
            #loss = F.mse_loss(x_recon, noise)
            loss = F.mse_loss(x_recon, x_0)
        else:
            x_recon = self.forward_diffusion_pass(x_0, t, cond)
            output_target = x_recon
            loss = 0
        return output_target, loss
    
    def forward_diffusion_pass(self, x_0, t, cond):
        """
        Pass through the denoising model
        """
        # Ensure t has at least one dimension
        if t.dim() == 0:
            t = t.unsqueeze(0)
            
        timestep_embed = self.timestep_embed(t[:, None])

        # Expand timestep_embed and class_embed to match the spatial dimensions of x_0
        timestep_embed = expand_to_planes(self.timestep_embed(t[:, None]), x_0.shape, repeat_batch=True)
    
        # Resize class_embed to match the spatial dimensions of x_0 and timestep_embed
        class_embed = expand_to_planes(self.class_embed(cond + 1), x_0.shape, repeat_batch=False)
    
        # Concatenate along dimension 1 (channel dimension)
        pred = self.backbone(torch.cat([x_0, class_embed, timestep_embed], dim=1))
        return pred


    def forward_diffusion_sample(self, x_0, t):
        """
        Sample noisy versions of x_0
        """
        noise = torch.randn_like(x_0, device=x_0.device)
        x_noisy = self.q_sample(x_0, t, noise)
        return x_noisy, noise


def epsilon(backbone, cfg):
    data_shape = cfg['data_shape']
    hidden_size = cfg['diffusion']['hidden_size']
    target_size = cfg['target_size']
    model = Epsilon(backbone, data_shape, hidden_size, target_size)
    return model
