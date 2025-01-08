import math
from model.model import *


class Diffusion(nn.Module):
    def __init__(self, core):
        super().__init__()
        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)
        self.core = core

    def forward(self, input):
        x_0 = input['data']
        cond = input['target']
        if 'training' in input:
            training = input['training']
        else:
            training = self.training
        if training:
            t = self.rng.draw(x_0.shape[0])[:, 0].to(x_0.device)
        else:
            t = input['t']
        output = {}
        output['data'], output['loss'] = self.core(x_0, t, cond, training)
        return output


def extract(a, t, x_shape):
    # retreive the data from the buffer according to the timestep and reshape to the shape wanted
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def get_index_from_list(vals, t, x_shape):
    """
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    # Scale t to the appropriate range
    t_scaled = t * (vals.size(0) - 1)
    # Convert to integer indices
    t_int = t_scaled.long()
    out = vals.gather(-1, t_int)
    out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    return out


def get_alphas_sigmas(t):
    """Returns the scaling factors for the clean image (alpha) and for the
    noise (sigma), given a timestep."""
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)


def diffusion(core, cfg):
    model = Diffusion(core)
    # model.apply(init_param)
    return model
