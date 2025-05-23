import math
from model.model import *
from model.backbone import TimeEmbedding, ConditionEmbedding, expand_to_planes


class Base(nn.Module):
    def __init__(self, backbone, target_size, class_dropout, timestep_embedding_size, timestep_embedding_mode,
                 cond_embedding_size):
        super().__init__()
        self.backbone = backbone
        self.target_size = target_size
        self.class_dropout = class_dropout
        self.time_embedding = TimeEmbedding(timestep_embedding_size, timestep_embedding_mode) # TODO: rename timestep to time
        self.cond_embedding = ConditionEmbedding(self.target_size + 1, cond_embedding_size)

    @property
    def is_time(self):
        return self.time_embedding.is_time

    @property
    def is_cond(self):
        return self.cond_embedding.is_cond

    def forward_diffusion_pass(self, z, t, cond=None):
        time_embedding = self.time_embedding(t[:, None])
        if self.is_cond and cond is not None:
            cond_embedding = self.cond_embedding(cond + 1)
        else:
            cond_embedding = None
        pred = self.backbone(z, time_embedding, cond_embedding)
        # time_embedding = self.time_embedding(t)
        # cond_embedding = self.cond_embedding(cond)
        # pred = self.backbone(z, time_embedding, cond_embedding)
        # pred = z + pred # add residual
        return pred

    def make_noised_reals(self, x_0, noise, t):
        raise NotImplementedError

    def make_targets(self, x_0, noise, t):
        raise NotImplementedError

    def make_cond(self, classes):
        if self.is_cond:
            # Drop out the class of the examples
            to_drop = torch.rand(classes.shape, device=classes.device).le(self.class_dropout)
            cond = torch.where(to_drop, -torch.ones_like(classes), classes)
        else:
            cond = None
        return cond

    def forward(self, z, t, cond, training=True):
        if training:
            x_0 = z
            noise = self.make_noise(x_0)
            noised_reals = self.make_noised_reals(x_0, noise, t)
            targets = self.make_targets(x_0, noise, t)
            cond = self.make_cond(cond)
            predicted = self.forward_diffusion_pass(noised_reals, t, cond)
            loss = F.mse_loss(predicted, targets)
        else:
            predicted = self.forward_diffusion_pass(z, t, cond)
            loss = 0
        return predicted, loss


class OptimalTransport(Base):
    def __init__(self, backbone, target_size, class_dropout, timestep_embedding_size, timestep_embedding_mode,
                 cond_embedding_size):
        super().__init__(backbone, target_size, class_dropout, timestep_embedding_size, timestep_embedding_mode,
                         cond_embedding_size)

    def make_noise(self, x):
        noise = torch.randn_like(x)
        return noise

    def make_noised_reals(self, x_0, noise, t):
        # psi_t
        """ Conditional Flow
        """
        # return (1 - (1 - self.sig_min) * t) * noise + t * x_0
        t = t.view(t.size(0), *[1 for _ in range(len(x_0.shape[1:]))])
        noised_reals = (1 - t) * noise + t * x_0
        return noised_reals

    def make_targets(self, x_0, noise, t):
        targets = x_0 - noise
        return targets


class VariancePreserve(Base):

    def __init__(self, backbone, target_size, class_dropout, timestep_embedding_size, timestep_embedding_mode,
                 cond_embedding_size):
        super().__init__(backbone, target_size, class_dropout, timestep_embedding_size, timestep_embedding_mode,
                         cond_embedding_size)

    def alpha_t(self, t):
        return torch.cos(t * math.pi / 2)

    def sigma_t(self, t):
        return torch.sin(t * math.pi / 2)

    def make_noise(self, x):
        noise = torch.randn_like(x)
        return noise

    def make_noised_reals(self, x_0, noise, t):
        t = t.view(t.size(0), 1)
        noised_reals = self.alpha_t(t) * x_0 + self.sigma_t(t) * noise
        return noised_reals

    def make_targets(self, x_0, noise, t):
        t = t.view(t.size(0), 1)  # TODO: make this more adaptable
        targets = math.pi / 2 * (self.alpha_t(t) * noise - self.sigma_t(t) * x_0)
        return targets


class VarianceExplode(Base):

    def __init__(self, backbone, target_size, class_dropout, timestep_embedding_size, timestep_embedding_mode,
                 cond_embedding_size):
        super().__init__(backbone, target_size, class_dropout, timestep_embedding_size, timestep_embedding_mode,
                         cond_embedding_size)
        self.sigma_min = 0.01
        self.sigma_max = 2.
        self.eps = 1e-5

    def sigma_t(self, t):
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t

    def dsigma_dt(self, t):
        return self.sigma_t(t) * torch.log(torch.tensor(self.sigma_max / self.sigma_min))

    def u_t(self, t: torch.Tensor, x: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
        return -(self.dsigma_dt(1. - t) / self.sigma_t(1. - t)) * (x - x_1)

    def make_noise(self, x):
        noise = torch.randn_like(x)
        return noise

    def make_noised_reals(self, x_0, noise, t):
        t = t.view(t.size(0), 1)
        noised_reals = x_0 + self.sigma_t(1. - t) * noise
        return noised_reals

    def make_targets(self, x_0, noise, t):
        t = t.view(t.size(0), 1)
        noised_reals = self.make_noised_reals(x_0, noise, t)
        targets = -(self.dsigma_dt(1. - t) / self.sigma_t(1. - t)) * (noised_reals - x_0)
        return targets


def ot(backbone, cfg):
    target_size = cfg['target_size']
    class_dropout = cfg['flow']['class_dropout']
    timestep_embedding_size = cfg['timestep_embedding_size']
    timestep_embedding_mode = cfg['timestep_embedding_mode']
    cond_embedding_size = cfg['cond_embedding_size']
    model = OptimalTransport(backbone, target_size, class_dropout, timestep_embedding_size, timestep_embedding_mode,
                             cond_embedding_size)
    return model


def vp(backbone, cfg):
    target_size = cfg['target_size']
    class_dropout = cfg['flow']['class_dropout']
    timestep_embedding_size = cfg['timestep_embedding_size']
    timestep_embedding_mode = cfg['timestep_embedding_mode']
    cond_embedding_size = cfg['cond_embedding_size']
    model = VariancePreserve(backbone, target_size, class_dropout, timestep_embedding_size, timestep_embedding_mode,
                             cond_embedding_size)
    return model


def ve(backbone, cfg):
    target_size = cfg['target_size']
    class_dropout = cfg['flow']['class_dropout']
    timestep_embedding_size = cfg['timestep_embedding_size']
    timestep_embedding_mode = cfg['timestep_embedding_mode']
    cond_embedding_size = cfg['cond_embedding_size']
    model = VarianceExplode(backbone, target_size, class_dropout, timestep_embedding_size, timestep_embedding_mode,
                            cond_embedding_size)
    return model
