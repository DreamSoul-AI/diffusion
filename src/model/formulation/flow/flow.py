import math
from model.model import *


# TODO: seems the same as diffusion, let ot inherit this
class Flow(nn.Module):

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


def flow(core, cfg):
    model = Flow(core)
    # model.apply(init_param)
    return model
