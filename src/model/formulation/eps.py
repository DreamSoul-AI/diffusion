import torch
import torch.nn as nn
from torch.nn import functional as F
from .diffusion import get_alphas_sigmas
from ..backbone import expand_to_planes, FourierFeatures


class Eps(nn.Module):
    def __init__(self, backbone, data_shape, hidden_size, target_size, class_dropout):
        super().__init__()
        self.data_shape = data_shape
        self.hidden_size = hidden_size
        self.target_size = target_size
        self.class_dropout = class_dropout
        self.timestep_embed = FourierFeatures(1, 16)
        self.class_embed = nn.Embedding(self.target_size + 1, 4)
        self.backbone = backbone

    def forward(self, x_0, t, cond, training=True):
        if training:
            noised_reals, targets, classes_drop = self.forward_diffusion_sample(x_0, t, cond)
            predicted_noise = self.forward_diffusion_pass(noised_reals, t, classes_drop)
            output_target = predicted_noise
            loss = F.mse_loss(output_target, targets)
        else:
            predicted_noise = self.forward_diffusion_pass(x_0, t, cond)
            output_target = predicted_noise
            loss = 0
        return output_target, loss

    def forward_diffusion_pass(self, x_0, t, cond):
        timestep_embed = expand_to_planes(self.timestep_embed(t[:, None]), x_0.shape)
        class_embed = expand_to_planes(self.class_embed(cond + 1), x_0.shape)
        pred = self.backbone(torch.cat([x_0, class_embed, timestep_embed], dim=1))
        return pred

    def forward_diffusion_sample(self, x_0, t, classes):
        # Calculate the noise schedule parameters for those timesteps
        alphas, sigmas = get_alphas_sigmas(t)

        # Combine the ground truth images and the noise
        alphas = alphas[:, None, None, None]
        sigmas = sigmas[:, None, None, None]
        noise = torch.randn_like(x_0)
        noised_reals = x_0 * alphas + noise * sigmas
        targets = noise

        # Drop out the class of the examples
        to_drop = torch.rand(classes.shape, device=classes.device).le(self.class_dropout)
        classes_drop = torch.where(to_drop, -torch.ones_like(classes), classes)
        return noised_reals, targets, classes_drop


def eps(backbone, cfg):
    data_shape = cfg['data_shape']
    hidden_size = cfg['diffusion']['hidden_size']
    target_size = cfg['target_size']
    class_dropout = cfg['diffusion']['class_dropout']
    model = Eps(backbone, data_shape, hidden_size, target_size, class_dropout)
    return model
