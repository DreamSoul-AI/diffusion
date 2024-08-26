from config import cfg
from .Diffusion_Epsilon import *
from .Diffusion_V import *
from .Diffusion_Xprev import *
from .Diffusion_Xzero import *

def make_model(cfg):
    #model = eval('model.{}(cfg)'.format(cfg['model_name']))
    if cfg['model_name'] == 'diffusionEpsilon':
        model = diffusionEpsilon(cfg)
    return model