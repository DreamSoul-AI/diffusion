import kornia.augmentation as K
from model.model import *


class Base(nn.Module):

    def __init__(self, core, data_name):
        super().__init__()

        self.augmentation = {}
        if data_name in ['MNIST', 'FashionMNIST']:
            self.augmentation['train'] = nn.Sequential(
                K.Resize(32),
                K.Normalize(mean=0.5, std=0.5)
            )
            self.augmentation['test'] = nn.Sequential(
                K.Resize(32),
                K.Normalize(mean=0.5, std=0.5)
            )
        elif data_name in ['CIFAR10', 'CIFAR100']:
            self.augmentation['train'] = nn.Sequential(
                K.RandomHorizontalFlip(p=0.5),
                K.RandomCrop((32, 32), padding=4, padding_mode='reflect'),
                K.Normalize(mean=0.5, std=0.5)
            )
            self.augmentation['test'] = K.Normalize(mean=0.5, std=0.5)
        elif data_name in ['SVHN']:
            self.augmentation['train'] = nn.Sequential(
                K.RandomCrop((32, 32), padding=4, padding_mode='reflect'),
                K.Normalize(mean=0.5, std=0.5)
            )
            self.augmentation['test'] = K.Normalize(mean=0.5, std=0.5)
        else:
            raise ValueError('Not valid data_name')
        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)
        self.core = core

    def forward(self, input):
        x_0 = input['data']
        cond = input['target']

        if self.training:
            x_0 = self.augmentation['train'](x_0)
        else:
            x_0 = self.augmentation['test'](x_0)

        if 'training' in input:
            training = input['training']
        else:
            training = self.training

        if training:
            t = self.rng.draw(x_0.shape[0])[:, 0].to(x_0.device)
        else:
            t = input['t']

        output = {}
        output['data'], loss, loss_v, loss_x0, loss_x1, loss_consistency = self.core(x_0, t, cond, training)
        output['loss'] = loss
        output['loss_v'] = loss_v
        output['loss_x0'] = loss_x0
        output['loss_x1'] = loss_x1
        output['loss_consistency'] = loss_consistency
        return output


def base(core, cfg):
    model = Base(core, cfg['data_name'])
    return model
