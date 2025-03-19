# from functools import partial
# from zuko.utils import odeint
from model.model import *
from model.backbone import TimeEmbedding


class Base(nn.Module):
    def __init__(self, backbone, target_size, class_dropout, timestep_embedding_size, timestep_embedding_mode,
                 cond_embedding_size):
        super().__init__()
        self.backbone = backbone
        self.target_size = target_size
        self.class_dropout = class_dropout
        self.timestep_embedding = TimeEmbedding(timestep_embedding_size, timestep_embedding_mode)
        if cond_embedding_size > 0:
            self.cond_embedding = nn.Embedding(self.target_size + 1, cond_embedding_size)
            self.is_cond = True
        else:
            self.cond_embedding = None
            self.is_cond = False

    def forward_diffusion_pass(self, z, t, cond=None):
        timestep_embedding = self.timestep_embedding(t)
        if self.cond_embedding is not None and cond is not None:
            cond_embedding = self.cond_embedding(cond + 1)
        else:
            cond_embedding = None
        pred = self.backbone(z, timestep_embedding, cond_embedding)
        return pred

    def make_noise(self, x_1):
        noise = torch.randn_like(x_1)
        return noise

    def make_noised_reals(self, x_1, noise, t):
        raise NotImplementedError

    def make_targets(self, x_1, noise, t):
        raise NotImplementedError

    def make_cond(self, classes):
        if self.is_cond:
            # Drop out the class of the examples
            to_drop = torch.rand(classes.shape, device=classes.device).le(self.class_dropout)
            cond = torch.where(to_drop, -torch.ones_like(classes), classes)
        else:
            cond = None
        return cond

    # def ode_wrapper(self, t, x, cond=None):
    #     # t = t * torch.ones(len(x), device=x.device)
    #     # x = x.new_full((len(t),), t.item(), device=x.device)
    #     t = t.expand(x.size(0))
    #     return self.forward_diffusion_pass(x, t, cond)

    # # def decode_t0_t1(self, x_0, t0, t1):  # TODO: merge with decode
    # #     return odeint(self.wrapper, x_0, t0, t1, self.parameters())
    #
    # # def encode(self, x_1, t0=1., t1=0.):  # TODO: not used, add t0, t1 option
    # #     return odeint(self.ode_wrapper, x_1, t0, t1, self.parameters())
    # #
    # # def decode(self, x_0, t0=0., t1=1.):
    # #     return odeint(self.ode_wrapper, x_0, t0, t1, self.parameters())
    #
    # def encode(self, x_1, t0=1., t1=0., cond=None):  # TODO: not used, add t0, t1 option
    #     ode_wrapper = partial(self.ode_wrapper, cond=cond)
    #     return odeint(ode_wrapper, x_1, t0, t1, self.parameters())
    #
    # def decode(self, noise, t0=0., t1=1., cond=None):
    #     ode_wrapper = partial(self.ode_wrapper, cond=cond)
    #     return odeint(ode_wrapper, noise, t0, t1, self.parameters())

    # def forward_diffusion_sample(self, x_1, t, classes):  # TODO: inverse x_0 and x_1 from diffusion
    #     t = t[:, None, None, None]
    #     x_0 = torch.randn_like(x_1) # TODO: x_0 is noise
    #     noised_reals = self.psi_t(x_0, x_1, t, self.sig_min)
    #     targets = x_1 - (1 - self.sig_min) * x_0
    #
    #     # Drop out the class of the examples
    #     to_drop = torch.rand(classes.shape, device=classes.device).le(self.class_dropout)
    #     classes_drop = torch.where(to_drop, -torch.ones_like(classes), classes)
    #     return noised_reals, targets, classes_drop

    # def forward(self, z, t, cond, training=True):
    #     if training:
    #         noise = self.make_noise(z)
    #         noised_reals = self.make_noised_reals(z, noise, t)
    #         targets = self.make_targets(z, noise, t)
    #         classes_drop = self.make_classes_drop(cond)
    #
    #         noised_reals, targets, classes_drop = self.forward_diffusion_sample(x_0, t, cond)
    #         predicted = self.forward_diffusion_pass(noised_reals, t, classes_drop)
    #         loss = F.mse_loss(predicted, targets)
    #     else:
    #         predicted = self.forward_diffusion_pass(z, t, cond)
    #         loss = 0
    #     return predicted, loss

    def forward(self, z, t, cond, training=True):
        if training:
            noise = self.make_noise(z)
            noised_reals = self.make_noised_reals(z, noise, t)
            targets = self.make_targets(z, noise, t)
            cond = self.make_cond(cond)
            predicted = self.forward_diffusion_pass(noised_reals, t, cond)
            loss = F.mse_loss(predicted, targets)
        else:
            predicted = self.forward_diffusion_pass(z, t, cond)
            loss = 0
        return predicted, loss


class OptimalTransport(Base):
    def __init__(self, backbone, target_size, class_dropout, timestep_embedding_size, timestep_embedding_mode,
                 cond_embedding_size, sig_min=1e-3):
        super().__init__(backbone, target_size, class_dropout, timestep_embedding_size, timestep_embedding_mode,
                         cond_embedding_size)
        self.sig_min = sig_min

    def make_noised_reals(self, x_1, noise, t):
        # psi_t
        """ Conditional Flow
        """
        t = t.view(t.size(0), *[1 for _ in range(len(x_1.shape[1:]))])
        # return (1 - (1 - self.sig_min) * t) * noise + t * x_0
        return (1 - (1 - self.sig_min) * t) * noise + t * x_1

    def make_targets(self, x_1, noise, t):
        targets = x_1 - (1 - self.sig_min) * noise  # TODO: still got question
        return targets


def ot(backbone, cfg):
    target_size = cfg['target_size']
    class_dropout = cfg['flow']['class_dropout']
    timestep_embedding_size = cfg['timestep_embedding_size']
    timestep_embedding_mode = cfg['timestep_embedding_mode']
    cond_embedding_size = cfg['cond_embedding_size']
    sig_min = cfg['flow']['sig_min'] # TODO: remove this
    model = OptimalTransport(backbone, target_size, class_dropout, timestep_embedding_size, timestep_embedding_mode,
                             cond_embedding_size, sig_min)
    return model


class VPDiffusionFlowMatching(Base):

    def __init__(self) -> None:
        super().__init__()
        self.beta_min = 0.1
        self.beta_max = 20.0
        self.eps = 1e-5

    def T(self, s: torch.Tensor) -> torch.Tensor:
        return self.beta_min * s + 0.5 * (s ** 2) * (self.beta_max - self.beta_min)

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        return torch.exp(-0.5 * self.T(t))

    def mu_t(self, t: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
        return self.alpha(1. - t) * x_1

    def sigma_t(self, t: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(1. - self.alpha(1. - t) ** 2)

    def u_t(self, t: torch.Tensor, x: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
        num = torch.exp(-self.T(1. - t)) * x - torch.exp(-0.5 * self.T(1. - t)) * x_1
        denum = 1. - torch.exp(- self.T(1. - t))
        return - 0.5 * self.beta(1. - t) * (num / denum)

    def loss(self, v_t: nn.Module, x_1: torch.Tensor) -> torch.Tensor:
        """ Compute loss
        """
        # t ~ Unif([0, 1])
        t = (torch.rand(1, device=x_1.device) + torch.arange(len(x_1), device=x_1.device) / len(x_1)) % (1 - self.eps)
        t = t[:, None].expand(x_1.shape)
        # x ~ p_t(x|x_1)
        x = self.mu_t(t, x_1) + self.sigma_t(t, x_1) * torch.randn_like(x_1)

        return torch.mean((v_t(t[:, 0], x) - self.u_t(t, x, x_1)) ** 2)


class VEDiffusionFlowMatching:

    def __init__(self) -> None:
        super().__init__()
        self.sigma_min = 0.01
        self.sigma_max = 2.
        self.eps = 1e-5

    def sigma_t(self, t: torch.Tensor) -> torch.Tensor:
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t

    def dsigma_dt(self, t: torch.Tensor) -> torch.Tensor:
        return self.sigma_t(t) * torch.log(torch.tensor(self.sigma_max / self.sigma_min))

    def u_t(self, t: torch.Tensor, x: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
        return -(self.dsigma_dt(1. - t) / self.sigma_t(1. - t)) * (x - x_1)

    def loss(self, v_t: nn.Module, x_1: torch.Tensor) -> torch.Tensor:
        """ Compute loss
        """
        # t ~ Unif([0, 1])
        t = (torch.rand(1, device=x_1.device) + torch.arange(len(x_1), device=x_1.device) / len(x_1)) % (1 - self.eps)
        t = t[:, None].expand(x_1.shape)
        # x ~ p_t(x|x_1)
        x = x_1 + self.sigma_t(1. - t) * torch.randn_like(x_1)

        return torch.mean((v_t(t[:, 0], x) - self.u_t(t, x, x_1)) ** 2)
