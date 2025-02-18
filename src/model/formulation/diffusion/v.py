from .base import Base
from ..utils import get_alphas_sigmas

class V(Base):
    def __init__(self, backbone, target_size, class_dropout):
        super().__init__(backbone, target_size, class_dropout)
        # self.target_size = target_size
        # self.class_dropout = class_dropout
        # self.timestep_embed = FourierFeatures(1, 16)
        # self.class_embed = nn.Embedding(self.target_size + 1, 4)
        # self.backbone = backbone

    def make_targets(self, x_0, noise, t):
        alphas, sigmas = get_alphas_sigmas(t)
        alphas = alphas.view(alphas.size(0), *[1 for _ in range(len(x_0.shape[1:]))])
        sigmas = sigmas.view(alphas.size(0), *[1 for _ in range(len(x_0.shape[1:]))])
        targets = noise * alphas - x_0 * sigmas
        return targets

    # def forward_diffusion_pass(self, x_0, t, cond):
    #     timestep_embed = expand_to_planes(self.timestep_embed(t[:, None]), x_0.shape)
    #     class_embed = expand_to_planes(self.class_embed(cond + 1), x_0.shape)
    #     pred = self.backbone(torch.cat([x_0, class_embed, timestep_embed], dim=1))
    #     return pred
    #
    # def forward_diffusion_sample(self, x_0, t, classes):
    #     # Calculate the noise schedule parameters for those timesteps
    #     alphas, sigmas = get_alphas_sigmas(t)
    #
    #     # Combine the ground truth images and the noise
    #     alphas = alphas[:, None, None, None]
    #     sigmas = sigmas[:, None, None, None]
    #     noise = torch.randn_like(x_0)
    #     noised_reals = x_0 * alphas + noise * sigmas
    #     targets = noise * alphas - x_0 * sigmas
    #
    #     # Drop out the class of the examples
    #     to_drop = torch.rand(classes.shape, device=classes.device).le(self.class_dropout)
    #     classes_drop = torch.where(to_drop, -torch.ones_like(classes), classes)
    #     return noised_reals, targets, classes_drop
    #
    # def forward(self, x_0, t, cond, training=True):
    #     if training:
    #         noised_reals, targets, classes_drop = self.forward_diffusion_sample(x_0, t, cond)
    #         predicted_v = self.forward_diffusion_pass(noised_reals, t, classes_drop)
    #         output_target = predicted_v
    #         loss = F.mse_loss(output_target, targets)
    #     else:
    #         predicted_v = self.forward_diffusion_pass(x_0, t, cond)
    #         output_target = predicted_v
    #         loss = 0
    #     return output_target, loss


def v(backbone, cfg):
    target_size = cfg['target_size']
    class_dropout = cfg['diffusion']['class_dropout']
    model = V(backbone, target_size, class_dropout)
    return model
