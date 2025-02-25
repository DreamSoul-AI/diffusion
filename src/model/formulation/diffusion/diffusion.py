import torch
import torch.nn as nn


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




def diffusion(core, cfg):
    model = Diffusion(core)
    return model
