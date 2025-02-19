import torch
import torch.nn as nn
from torch.nn import functional as F
from model.backbone import FourierFeatures
from ..utils import get_alphas_sigmas, expand_to_planes


class Base(nn.Module):
    def __init__(self, backbone, target_size, class_dropout):
        super().__init__()
        self.target_size = target_size
        self.class_dropout = class_dropout
        self.timestep_embed = FourierFeatures(1, 16)
        self.class_embed = nn.Embedding(self.target_size + 1, 4)
        self.backbone = backbone

    def forward_diffusion_pass(self, z, t, cond):
        timestep_embed = expand_to_planes(self.timestep_embed(t[:, None]), z.shape)
        class_embed = expand_to_planes(self.class_embed(cond + 1), z.shape)
        pred = self.backbone(torch.cat([z, class_embed, timestep_embed], dim=1))
        return pred

    def make_noise(self, x_0):
        noise = torch.randn_like(x_0)
        return noise

    def make_noised_reals(self, x_0, noise, t):
        # Calculate the noise schedule parameters for those timesteps
        alphas, sigmas = get_alphas_sigmas(t)
        # Combine the ground truth images and the noise
        alphas = alphas.view(alphas.size(0), *[1 for _ in range(len(x_0.shape[1:]))])
        sigmas = sigmas.view(alphas.size(0), *[1 for _ in range(len(x_0.shape[1:]))])
        noised_reals = x_0 * alphas + noise * sigmas
        return noised_reals

    def make_targets(self, x_0, noise, t):
        raise NotImplementedError

    def make_classes_drop(self, classes):
        # Drop out the class of the examples
        to_drop = torch.rand(classes.shape, device=classes.device).le(self.class_dropout)
        classes_drop = torch.where(to_drop, -torch.ones_like(classes), classes)
        return classes_drop

    def forward(self, z, t, cond, training=True):
        if training:
            noise = self.make_noise(z)
            noised_reals = self.make_noised_reals(z, noise, t)
            targets = self.make_targets(z, noise, t)
            classes_drop = self.make_classes_drop(cond)
            predicted = self.forward_diffusion_pass(noised_reals, t, classes_drop)
            loss = F.mse_loss(predicted, targets)
        else:
            predicted = self.forward_diffusion_pass(z, t, cond)
            loss = 0
        return predicted, loss


class Eps(Base):
    def __init__(self, backbone, target_size, class_dropout):
        super().__init__(backbone, target_size, class_dropout)

    def make_targets(self, x_0, noise, t):
        targets = noise
        return targets


class X(Base):
    def __init__(self, backbone, target_size, class_dropout):
        super().__init__(backbone, target_size, class_dropout)

    def make_targets(self, x_0, noise, t):
        targets = x_0
        return targets


class V(Base):
    def __init__(self, backbone, target_size, class_dropout):
        super().__init__(backbone, target_size, class_dropout)

    def make_targets(self, x_0, noise, t):
        alphas, sigmas = get_alphas_sigmas(t)
        alphas = alphas.view(alphas.size(0), *[1 for _ in range(len(x_0.shape[1:]))])
        sigmas = sigmas.view(alphas.size(0), *[1 for _ in range(len(x_0.shape[1:]))])
        targets = noise * alphas - x_0 * sigmas
        return targets


class Regularized(Base):
    def __init__(self, backbone, target_size, class_dropout, regularization):
        super().__init__(backbone, target_size, class_dropout)
        self.lambda_v = regularization['v']
        self.lambda_x0 = regularization['x0']
        self.lambda_eps = regularization['eps']

    def make_targets(self, x_0, noise, t):
        alphas, sigmas = get_alphas_sigmas(t)
        alphas = alphas.view(alphas.size(0), *[1 for _ in range(len(x_0.shape[1:]))])
        sigmas = sigmas.view(alphas.size(0), *[1 for _ in range(len(x_0.shape[1:]))])
        v_targets = alphas * noise - sigmas * x_0
        eps_targets = noise
        x0_targets = x_0
        return v_targets, eps_targets, x0_targets

    def forward(self, z, t, cond, training=True):
        if training:
            alphas, sigmas = get_alphas_sigmas(t)
            alphas = alphas.view(alphas.size(0), *[1 for _ in range(len(z.shape[1:]))])
            sigmas = sigmas.view(alphas.size(0), *[1 for _ in range(len(z.shape[1:]))])

            noise = self.make_noise(z)
            noised_reals = self.make_noised_reals(z, noise, t)
            classes_drop = self.make_classes_drop(cond)
            predicted_v = self.forward_diffusion_pass(noised_reals, t, classes_drop)
            predicted_x0 = noised_reals * alphas - predicted_v * sigmas
            predicted_eps = noised_reals * sigmas + predicted_v * alphas
            predicted = predicted_v
            v_targets, eps_targets, x0_targets = self.make_targets(z, noise, t)

            # Compute individual losses
            loss_v = F.mse_loss(predicted_v, v_targets)
            loss_x0 = F.mse_loss(predicted_x0, x0_targets)
            loss_eps = F.mse_loss(predicted_eps, eps_targets)
            # Compute final loss
            loss = self.lambda_v * loss_v + self.lambda_x0 * loss_x0 + self.lambda_eps * loss_eps
        else:
            predicted = self.forward_diffusion_pass(z, t, cond)
            loss = 0
        return predicted, loss


def eps(backbone, cfg):
    target_size = cfg['target_size']
    class_dropout = cfg['diffusion']['class_dropout']
    model = Eps(backbone, target_size, class_dropout)
    return model


def x(backbone, cfg):
    target_size = cfg['target_size']
    class_dropout = cfg['diffusion']['class_dropout']
    model = X(backbone, target_size, class_dropout)
    return model


def v(backbone, cfg):
    target_size = cfg['target_size']
    class_dropout = cfg['diffusion']['class_dropout']
    model = V(backbone, target_size, class_dropout)
    return model


def regularized(backbone, cfg):
    target_size = cfg['target_size']
    class_dropout = cfg['diffusion']['class_dropout']
    regularization = cfg['diffusion']['regularization']
    model = Regularized(backbone, target_size, class_dropout, regularization)
    return model
