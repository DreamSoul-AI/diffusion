from torch.nn import functional as F
from .base import Base
from ..utils import get_alphas_sigmas


class Regularized(Base):
    def __init__(self, backbone, target_size, class_dropout, regularization):
        super().__init__(backbone, target_size, class_dropout)
        self.lambda_v = regularization['v']
        self.lambda_x0 = regularization['x0']
        self.lambda_eps = regularization['eps']
        # self.target_size = target_size
        # self.class_dropout = class_dropout
        # self.timestep_embed = FourierFeatures(1, 16)
        # self.class_embed = nn.Embedding(self.target_size + 1, 4)
        # self.backbone = backbone

    # def forward_diffusion_pass(self, x_0, t, cond):
    #     """Pass inputs through the backbone network."""
    #     timestep_embed = expand_to_planes(self.timestep_embed(t[:, None]), x_0.shape)
    #     class_embed = expand_to_planes(self.class_embed(cond + 1), x_0.shape)
    #     pred = self.backbone(torch.cat([x_0, class_embed, timestep_embed], dim=1))
    #     return pred

    def make_targets(self, x_0, noise, t):
        alphas, sigmas = get_alphas_sigmas(t)
        alphas = alphas.view(alphas.size(0), *[1 for _ in range(len(x_0.shape[1:]))])
        sigmas = sigmas.view(alphas.size(0), *[1 for _ in range(len(x_0.shape[1:]))])
        v_targets = alphas * noise - sigmas * x_0
        eps_targets = noise
        x0_targets = x_0
        return v_targets, eps_targets, x0_targets

    # def forward_diffusion_sample(self, x_0, t, classes):
    #     """Generate noisy samples and targets."""
    #     alphas, sigmas = get_alphas_sigmas(t)
    #     alphas = alphas[:, None, None, None]
    #     sigmas = sigmas[:, None, None, None]
    #     noise = torch.randn_like(x_0)
    #     noised_reals = x_0 * alphas + noise * sigmas
    #
    #     # Calculate targets
    #     v_targets = alphas * noise - sigmas * x_0
    #     eps_targets = noise
    #     x0_targets = x_0
    #
    #     # Random class dropout
    #     to_drop = torch.rand(classes.shape, device=classes.device).le(self.class_dropout)
    #     classes_drop = torch.where(to_drop, -torch.ones_like(classes), classes)
    #     return noised_reals, v_targets, eps_targets, x0_targets, classes_drop

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

            #
            # noised_reals, v_targets, eps_targets, x0_targets, classes_drop = self.forward_diffusion_sample(x_0, t, cond)
            # predicted_v = self.forward_diffusion_pass(noised_reals, t, classes_drop)
            # alphas, sigmas = get_alphas_sigmas(t)
            # alphas, sigmas = alphas[:, None, None, None], sigmas[:, None, None, None]
            # predicted_x0 = noised_reals * alphas - predicted_v * sigmas
            # predicted_eps = noised_reals * sigmas + predicted_v * alphas
            # output_target = predicted_v

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


def regularized(backbone, cfg):
    target_size = cfg['target_size']
    class_dropout = cfg['diffusion']['class_dropout']
    regularization = cfg['diffusion']['regularization']
    model = Regularized(backbone, target_size, class_dropout, regularization)
    return model
