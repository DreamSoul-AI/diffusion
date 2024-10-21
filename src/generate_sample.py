import argparse
import matplotlib.pyplot as plt
import os
import torch
import torch.backends.cudnn as cudnn
from config import cfg, process_args
from model import make_model
from module import process_control, makedir_exist_ok
from module.sampler import ddim_sample_loop_V
from torchvision.utils import save_image

cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
args = vars(parser.parse_args())
process_args(args)


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    return model


def generate_sample():
    cfg['seed'] = int(cfg['tag'].split('_')[0])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    cfg['run_mode'] = 'test'
    cfg['path'] = os.path.join('output', 'exp')
    cfg['tag_path'] = os.path.join(cfg['path'], cfg['tag'])
    cfg['checkpoint_path'] = os.path.join(cfg['tag_path'], 'checkpoint')
    cfg['best_path'] = os.path.join(cfg['tag_path'], 'best')
    cfg['logger_path'] = os.path.join('output', 'logger', 'test', 'runs', cfg['tag'])
    cfg['result_path'] = os.path.join('output', 'result', cfg['tag'])
    cfg['sample_path'] = os.path.join('output', 'sample')

    # 256个 1 - 10 的数字    
    if(cfg['data_name'] == 'MNIST'):
        noise = torch.randn(cfg['generate']['batch_size'], 1, 32, 32).to(cfg['device'])
        classes = torch.arange(cfg['generate']['batch_size'], device=cfg['device']) % 10
    elif(cfg['data_name'] == 'CIFAR10'):
        noise = torch.randn(cfg['generate']['batch_size'], 3, 32, 32).to(cfg['device'])
        classes = torch.arange(cfg['generate']['batch_size'], device=cfg['device']) % 10
    else:
        raise ValueError('Not valid data name')
    
    model = make_model(cfg['model'])
    model = load_checkpoint(model, os.path.join(cfg['checkpoint_path'], 'model'), cfg['device'])
    model.to(cfg['device'])
    model.train(False)
    steps = cfg['generate']['steps']
    guidance_scale = cfg['generate']['guidance_scale']
    # The amount of noise to add each timestep when sampling
    # controls the scale of the variance (0 is DDIM, and 1 is one type of DDPM)
    # 0 = no noise (DDIM)
    # 1 = full noise (DDPM)
    eta = 1. if not cfg['generate']['use_ddim'] else 0.
    sample = ddim_sample_loop_V(model, noise, steps, eta, classes, guidance_scale)
    makedir_exist_ok(os.path.join(cfg['sample_path']))
    save_image(sample, os.path.join(cfg['sample_path'], '{}.{}'.format(cfg['tag'], cfg['generate']['img_fmt'])))
    
    from torchvision import datasets, transforms, utils
    from torchvision.transforms import functional as TF
    grid = utils.make_grid(sample, 10).cpu()
    filename = os.path.join(cfg['sample_path'], '{}.{}'.format(cfg['tag'], cfg['generate']['img_fmt']).replace('.png', '_grid.png'))
    TF.to_pil_image(grid.add(1).div(2).clamp(0, 1)).save(filename)
    return sample


def runExperiment():
    generate_sample()
    return


def main():
    seeds = list(range(cfg['init_seed'], cfg['init_seed'] + cfg['num_experiments']))
    for i in range(cfg['num_experiments']):
        tag_list = [str(seeds[i]), cfg['control_name']]
        cfg['tag'] = '_'.join([x for x in tag_list if x])
        process_control()
        print('Experiment: {}'.format(cfg['tag']))
        runExperiment()
    return


if __name__ == "__main__":
    main()
