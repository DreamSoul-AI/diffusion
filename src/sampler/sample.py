import argparse
import os
import torch
import torch.backends.cudnn as cudnn
from torch import optim, nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import datasets, transforms, utils
from torchvision.transforms import functional as TF
from tqdm.notebook import tqdm, trange
from config import cfg, process_args
from metric import make_logger
from model import make_model, make_optimizer, make_scheduler, get_alphas_sigmas
from module import save, check, resume, to_device, process_control

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
        print('sampling...: {}'.format(cfg['tag']))
        runExperiment()
    return


@torch.no_grad()
@torch.random.fork_rng()
def runExperiment():
    tqdm.write('\nSampling...')
    torch.manual_seed(cfg['seed'])

    noise = torch.randn([100, 3, 32, 32], device=cfg['device'])
    classes = torch.arange(10, device=cfg['device']).repeat_interleave(10, 0)

    model = make_model(cfg['model'])
    cfg['sample_path'] = os.path.join('output', 'sample', cfg['tag'])
    steps = cfg['steps']
    sample_fn = make_sample_fn(model, noise, steps, eta, classes, cfg['guidance_scale'])
    sample = sample_fn(model, noise, steps, eta, classes, cfg['guidance_scale'])
    save(sample, cfg['sample_path'])
    return

def make_sample_fn(model, x, steps, eta, classes, guidance_scale=1.):
    if cfg['model'] == 'diffusionV':
        sample_fn = (p_sample_loop_V(model, x, steps, eta, classes, guidance_scale=1.) if not cfg['use_ddim'] 
                     else ddim_sample_loop_V(model, x, steps, eta, classes, guidance_scale=1.))
    elif cfg['model'] == 'diffusionEpsilon':
        sample_fn = (p_sample_loop_Epsilon(model, x, steps, eta, classes, guidance_scale=1.) if not cfg['use_ddim'] 
                     else ddim_sample_loop_Epsilon(model, x, steps, eta, classes, guidance_scale=1.))
    elif cfg['model'] == 'diffusionX':
        sample_fn = (p_sample_loop_X(model, x, steps, eta, classes, guidance_scale=1.) if not cfg['use_ddim'] 
                     else ddim_sample_loop_X(model, x, steps, eta, classes, guidance_scale=1.))
    else:
        raise ValueError('Not valid sample function')
    return sample_fn

@torch.no_grad()
def ddim_sample_loop_V(model, x, steps, eta, classes, guidance_scale=1.):
    """Draws samples from a model given starting noise."""
    ts = x.new_ones([x.shape[0]])

    # Create the noise schedule
    t = torch.linspace(1, 0, steps + 1)[:-1]
    alphas, sigmas = get_alphas_sigmas(t)

    # The sampling loop
    for i in trange(steps):

        # Get the model output (v, the predicted velocity)
        with torch.cuda.amp.autocast():
            x_in = torch.cat([x, x])
            ts_in = torch.cat([ts, ts])
            classes_in = torch.cat([-torch.ones_like(classes), classes])
            v_uncond, v_cond = model(x_in, ts_in * t[i], classes_in).float().chunk(2)
        v = v_uncond + guidance_scale * (v_cond - v_uncond)

        # Predict the noise and the denoised image
        pred = x * alphas[i] - v * sigmas[i]
        eps = x * sigmas[i] + v * alphas[i]

        # If we are not on the last timestep, compute the noisy image for the
        # next timestep.
        if i < steps - 1:
            # If eta > 0, adjust the scaling factor for the predicted noise
            # downward according to the amount of additional noise to add
            ddim_sigma = eta * (sigmas[i + 1]**2 / sigmas[i]**2).sqrt() * \
                (1 - alphas[i]**2 / alphas[i + 1]**2).sqrt()
            adjusted_sigma = (sigmas[i + 1]**2 - ddim_sigma**2).sqrt()

            # Recombine the predicted noise and predicted denoised image in the
            # correct proportions for the next step
            x = pred * alphas[i + 1] + eps * adjusted_sigma

            # Add the correct amount of fresh noise
            if eta:
                x += torch.randn_like(x) * ddim_sigma

    # If we are on the last timestep, output the denoised image
    return pred

@torch.no_grad()
def p_sample_loop_V():
    return

@torch.no_grad()
def p_sample_loop_Epsilon():
    return

@torch.no_grad()
def ddim_sample_loop_Epsilon():
    return

@torch.no_grad()
def p_sample_loop_X():
    return

@torch.no_grad()
def ddim_sample_loop_X():
    return


if __name__ == "__main__":
    main()
