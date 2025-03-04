import math
from model.model import *
from zuko.utils import odeint


# TODO: seems the same as diffusion, let ot inherit this
class Flow(nn.Module):

    def __init__(self, core):
        super().__init__()
        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)
        self.core = core

    # def psi_t(self, x, x_1, t, sig_min=1e-3):
    #     """ Conditional Flow
    #     """
    #     return (1 - (1 - sig_min) * t) * x + t * x_1

    def forward(self, input):
        x_1 = input['data']
        cond = input['target']
        if 'training' in input:
            training = input['training']
        else:
            training = self.training
        if training:
            t = self.rng.draw(x_1.shape[0])[:, 0].to(x_1.device)
        else:
            t = input['t']
        output = {}
        output['data'], output['loss'] = self.core(x_1, t, cond, training)
        # x_0 = torch.randn_like(x_1)
        # v_psi = v_t(t[:, 0], self.psi_t(x_0, x_1, t))
        # d_psi = x_1 - (1 - self.sig_min) * x_0
        # loss = torch.mean((v_psi - d_psi) ** 2)  # TODO: improve mse loss
        return output

    # def loss(self, v_t, x_1):
    #     """ Compute loss
    #     """
    #     # t ~ Unif([0, 1])
    #     # TODO: why add? different samples for each data points, similar to torch.quasirandom.SobolEngine
    #     t = (torch.rand(1, device=x_1.device) + torch.arange(len(x_1), device=x_1.device) / len(x_1)) % (1 - self.eps)
    #     t = t[:, None].expand(x_1.shape)
    #     # x ~ p_t(x_0)
    #     x_0 = torch.randn_like(x_1)
    #     v_psi = v_t(t[:, 0], self.psi_t(x_0, x_1, t))
    #     d_psi = x_1 - (1 - self.sig_min) * x_0
    #     loss = torch.mean((v_psi - d_psi) ** 2)  # TODO: improve mse loss
    #     return loss


def flow(core, cfg):
    model = Flow(core)
    # model.apply(init_param)
    return model
