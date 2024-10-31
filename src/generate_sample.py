import argparse
import os
import torch
import torch.backends.cudnn as cudnn
from config import cfg, process_args
from model import make_model
from dataset import make_dataset, process_dataset
from module import resume, process_control, makedir_exist_ok, ddim_sample_loop_V
from torchvision.utils import save_image

cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
args = vars(parser.parse_args())
process_args(args)


def main():
    seeds = list(range(cfg['init_seed'], cfg['init_seed'] + cfg['num_experiments']))
    for i in range(cfg['num_experiments']):
        tag_list = [str(seeds[i]), cfg['control_name']]
        cfg['tag'] = '_'.join([x for x in tag_list if x])
        process_control()
        print('Experiment: {}'.format(cfg['tag']))
        runExperiment()
    return


def runExperiment():
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
    dataset = make_dataset(cfg['data_name'])
    dataset = process_dataset(dataset)
    model = make_model(cfg['model'])
    result = resume(cfg['best_path'])
    model = model.to(cfg['device'])
    model.load_state_dict(result['model'])
    generate(model)
    return


def generate(model):
    if cfg['data_name'] in ['MNIST', 'CIFAR10']:
        size = (cfg['generate']['batch_size'] * cfg['model']['target_size'], *cfg['model']['data_shape'])
        noise = torch.randn(size).to(cfg['device'])
        classes = torch.arange(cfg['model']['target_size'], device=cfg['device']).repeat(
            cfg['generate']['batch_size'])
    else:
        raise ValueError('Not valid data name')

    model.train(False)
    num_steps = cfg['generate']['num_steps']
    guidance_scale = cfg['generate']['guidance_scale']
    # The amount of noise to add each timestep when sampling
    # controls the scale of the variance (0 is DDIM, and 1 is one type of DDPM)
    # 0 = no noise (DDIM)
    # 1 = full noise (DDPM)
    eta = 1. if not cfg['generate']['use_ddim'] else 0.
    if cfg['model']['formulation_mode'] == "v":
        sample = ddim_sample_loop_V(model, noise, num_steps, eta, classes, guidance_scale)
    elif cfg['model']['formulation_mode'] == "epsilon":
        sample = ddim_sample_loop_Epsilon(model, noise, steps, eta, classes, guidance_scale)
    else:
        raise ValueError('Not valid formulation mode name')

    makedir_exist_ok(os.path.join(cfg['sample_path']))
    save_image(sample, os.path.join(cfg['sample_path'], '{}.{}'.format(cfg['tag'], cfg['generate']['img_fmt'])),
               normalize=True, value_range=(-1, 1), nrow=cfg['model']['target_size'])
    return sample


if __name__ == "__main__":
    main()
