from model.model import *


class Base(nn.Module):

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
        output['data'], loss, loss_v, loss_x0, loss_x1 = self.core(x_0, t, cond, training)
        output['loss'] = loss
        output['loss_v'] = loss_v
        output['loss_x0'] = loss_x0
        output['loss_x1'] = loss_x1
        return output


def base(core, cfg):
    model = Base(core)
    return model
