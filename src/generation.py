from sampler import *
from model import *
from config import cfg

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
        runExperiment()
    return


@torch.no_grad()
@torch.random.fork_rng()
def runExperiment():
    tqdm.write('\nSampling...')
    torch.manual_seed(cfg['seed'])

    noise = torch.randn([100, 3, 32, 32], device=cfg['device'])
    sample = make_sample(noise)

    grid = utils.make_grid(sample, 10).cpu()
    filename = f'demo_{cfg['tag']}.png'
    TF.to_pil_image(grid.add(1).div(2).clamp(0, 1)).save(filename)
    display.display(display.Image(filename))
    tqdm.write('')

    save(sample, cfg['sample_path'])
    return

if __name__ == "__main__":
    main()