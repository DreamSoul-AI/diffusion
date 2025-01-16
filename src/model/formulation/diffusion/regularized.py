import torch
import torch.nn as nn
from torch.nn import functional as F
from model.backbone import FourierFeatures
from .utils import get_alphas_sigmas, expand_to_planes


class Regularized(nn.Module):
    def __init__(self, backbone, data_shape, hidden_size, target_size):
        super().__init__()
        self.data_shape = data_shape
        self.hidden_size = hidden_size
        self.target_size = target_size
        self.timestep_embed = FourierFeatures(1, 16)
        self.class_embed = nn.Embedding(self.target_size + 1, 4)
        self.backbone = backbone

    def forward(self, x_0, t, cond, training=True):
        combined_prediction = {}
        if training:
            noised_reals, v_targets, eps_targets, x0_targets, classes_drop = self.forward_diffusion_sample(x_0, t, cond)
            predicted_v = self.forward_diffusion_pass(noised_reals, t, classes_drop)
            alphas, sigmas = get_alphas_sigmas(t)
            alphas, sigmas = alphas[:, None, None, None], sigmas[:, None, None, None]
            predicted_x0 = noised_reals * alphas - predicted_v * sigmas
            predicted_eps = noised_reals * sigmas + predicted_v * alphas
            output_target = predicted_v
            # Regularize predictions
            # v_predicted = torch.tanh(v_predicted)
            # eps_predicted = torch.tanh(eps_predicted)
            # x0_predicted = torch.tanh(x0_predicted)

            # Compute individual losses
            loss_v = F.mse_loss(predicted_v, v_targets)
            loss_x0 = F.mse_loss(predicted_x0, x0_targets)
            loss_eps = F.mse_loss(predicted_eps, eps_targets)

            # # Weight losses by their magnitude
            # w_v = 1.0 / (loss_v.item() + 1e-5)
            # w_eps = 1.0 / (loss_eps.item() + 1e-5)
            # w_x0 = 1.0 / (loss_x0.item() + 1e-5)

            # Use fixed weights
            w_v, w_x0, w_eps = 1, 0.1, 0.1

            # Compute final loss
            loss = w_v * loss_v + w_x0 * loss_x0 + w_eps * loss_eps

            # Combine the three predicted components
            # combined_prediction['v_predicted'] = v_predicted
            # combined_prediction['eps_predicted'] = eps_predicted
            # combined_prediction['x0_predicted'] = x0_predicted
        else:
            # Inference mode: predict a combination of v, eps, and x_0
            # Forward diffusion sample to generate noisy inputs and targets
            noised_reals, v_targets, eps_targets, x0_targets, classes_drop = self.forward_diffusion_sample(x_0, t, cond)

            alphas, sigmas = get_alphas_sigmas(t)
            # alphas, sigmas = alphas[:, None, None, None], sigmas[:, None, None, None]

            # Model prediction
            predicted_v = self.forward_diffusion_pass(x_0, t, cond)
            output_target = predicted_v
            loss = 0

            # eps_predicted = (v_predicted * sigmas + noised_reals) / alphas
            # x0_predicted = (noised_reals - eps_predicted * sigmas) / alphas
            #
            # # Combine the three predicted components
            # # combined_prediction['v_predicted'] = v_predicted
            # # combined_prediction['eps_predicted'] = eps_predicted
            # # combined_prediction['x0_predicted'] = x0_predicted
            # loss = None  # No loss in inference mode
        return output_target, loss

    def forward_diffusion_pass(self, x_0, t, cond):
        """Pass inputs through the backbone network."""
        timestep_embed = expand_to_planes(self.timestep_embed(t[:, None]), x_0.shape)
        class_embed = expand_to_planes(self.class_embed(cond + 1), x_0.shape)
        pred = self.backbone(torch.cat([x_0, class_embed, timestep_embed], dim=1))
        return pred

    def forward_diffusion_sample(self, x_0, t, classes):
        """Generate noisy samples and targets."""
        alphas, sigmas = get_alphas_sigmas(t)
        alphas = alphas[:, None, None, None]
        sigmas = sigmas[:, None, None, None]
        noise = torch.randn_like(x_0)
        noised_reals = x_0 * alphas + noise * sigmas

        # Calculate targets
        v_targets = alphas * noise - sigmas * x_0
        eps_targets = noise
        x0_targets = x_0

        # Random class dropout
        to_drop = torch.rand(classes.shape, device=classes.device).le(0.2)
        classes_drop = torch.where(to_drop, -torch.ones_like(classes), classes)
        return noised_reals, v_targets, eps_targets, x0_targets, classes_drop

    @staticmethod
    def normalize(tensor):
        """Normalize a tensor to zero mean and unit variance."""
        mean = tensor.mean()
        std = tensor.std()
        return (tensor - mean) / (std + 1e-5)  # Add epsilon to avoid division by zero


def regularized(backbone, cfg):
    data_shape = cfg['data_shape']
    hidden_size = cfg['diffusion']['hidden_size']
    target_size = cfg['target_size']
    model = Regularized(backbone, data_shape, hidden_size, target_size)
    return model
