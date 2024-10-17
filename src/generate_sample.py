import argparse
import matplotlib.pyplot as plt
import os
import torch
import torch.backends.cudnn as cudnn
from config import cfg, process_args
from model import make_model
from module import process_control, makedir_exist_ok
from module.sampler import ddim_sample_loop_V
from torchvision.utils import make_grid, save_image

cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
args = vars(parser.parse_args())
process_args(args)


def save_grid_image(tensor, path, nrow=16):
    # Make a grid of images
    grid = make_grid(tensor, nrow=nrow)
    # Convert grid to numpy array
    array = grid.cpu().numpy()
    # If the grid has a single channel, keep it as a single channel
    if array.shape[0] == 1:
        array = array.squeeze(0)
        plt.imsave(path, array, cmap='gray')
    else:
        # Transpose the array to (H, W, C)
        array = array.transpose(1, 2, 0)
        # Normalize the array to [0, 1]
        array = (array - array.min()) / (array.max() - array.min())
        # Save the image
        plt.imsave(path, array)
    return


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

    # classes = torch.arange(10, device=cfg['device']).repeat_interleave(10, 0)
    # 256个 1 - 10 的数字
    noise = torch.randn(cfg['generate']['batch_size'], 1, 32, 32).to(cfg['device'])
    classes = torch.arange(cfg['generate']['batch_size'], device=cfg['device']) % 10

    model = make_model(cfg['model'])
    model = load_checkpoint(model, os.path.join(cfg['checkpoint_path'], 'model'), cfg['device'])
    model.to(cfg['device'])
    steps = cfg['generate']['steps']
    guidance_scale = cfg['generate']['guidance_scale']
    eta = 1. if not cfg['generate']['use_ddim'] else 0.
    # The amount of noise to add each timestep when sampling
    # controls the scale of the variance (0 is DDIM, and 1 is one type of DDPM)
    # 0 = no noise (DDIM)
    # 1 = full noise (DDPM)
    sample = ddim_sample_loop_V(model, noise, steps, eta, classes, guidance_scale)
    # save_grid_image(sample, "output/sample.png")
    makedir_exist_ok(os.path.join(cfg['sample_path']))
    save_image(sample, os.path.join(cfg['sample_path'], '{}.{}'.format(cfg['tag'], cfg['generate']['img_fmt'])))
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
