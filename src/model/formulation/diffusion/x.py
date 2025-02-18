from .base import Base


class X(Base):
    def __init__(self, backbone, target_size, class_dropout):
        super().__init__(backbone, target_size, class_dropout)
        # self.target_size = target_size
        # self.class_dropout = class_dropout
        # self.timestep_embed = FourierFeatures(1, 16)
        # self.class_embed = nn.Embedding(self.target_size + 1, 4)
        # self.backbone = backbone

    # def forward_diffusion_pass(self, x_0, t, cond):
    #     timestep_embed = expand_to_planes(self.timestep_embed(t[:, None]), x_0.shape)
    #     class_embed = expand_to_planes(self.class_embed(cond + 1), x_0.shape)
    #     pred = self.backbone(torch.cat([x_0, class_embed, timestep_embed], dim=1))
    #     return pred
    #
    # def make_noised_reals(self, x_0, t):
    #     # Calculate the noise schedule parameters for those timesteps
    #     alphas, sigmas = get_alphas_sigmas(t)
    #     # Combine the ground truth images and the noise
    #     alphas = alphas.view(alphas.size(0), *[1 for _ in range(len(x_0.shape[1:]))])
    #     sigmas = sigmas.view(alphas.size(0), *[1 for _ in range(len(x_0.shape[1:]))])
    #     noise = torch.randn_like(x_0)
    #     noised_reals = x_0 * alphas + noise * sigmas
    #     return noised_reals

    def make_targets(self, x_0, noise, t):
        targets = x_0
        return targets
    #
    # def make_classes_drop(self, classes):
    #     # Drop out the class of the examples
    #     to_drop = torch.rand(classes.shape, device=classes.device).le(self.class_dropout)
    #     classes_drop = torch.where(to_drop, -torch.ones_like(classes), classes)
    #     return classes_drop


    # def forward_diffusion_sample(self, x_0, t, classes):
    #     # Calculate the noise schedule parameters for those timesteps
    #     alphas, sigmas = get_alphas_sigmas(t)
    #
    #     # Combine the ground truth images and the noise
    #     alphas = alphas[:, None, None, None] # TODO: make this more general
    #     sigmas = sigmas[:, None, None, None]
    #     noise = torch.randn_like(x_0)
    #     noised_reals = x_0 * alphas + noise * sigmas
    #     targets = x_0  # Update targets to be x_0 instead of noise
    #
    #
    #     return noised_reals, targets, classes_drop

    # def forward(self, x_0, t, cond, training=True):
    #     if training:
    #         noised_reals = self.make_noised_reals(x_0, t)
    #         targets = self.make_targets(x_0, t)
    #         classes_drop = self.make_classes_drop(cond)
    #         # noised_reals, targets, classes_drop = self.forward_diffusion_sample(x_0, t, cond)
    #         predicted = self.forward_diffusion_pass(noised_reals, t, classes_drop)
    #         loss = F.mse_loss(predicted, targets)
    #     else:
    #         predicted = self.forward_diffusion_pass(x_0, t, cond)
    #         loss = 0
    #     return predicted, loss


def x(backbone, cfg):
    target_size = cfg['target_size']
    class_dropout = cfg['diffusion']['class_dropout']
    model = X(backbone, target_size, class_dropout)
    return model
