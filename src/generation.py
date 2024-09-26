from sampler import *
from model import *
from config import cfg
import os
from PIL import Image

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

    noise = torch.randn([100, 1, 32, 32],
                        device=cfg['device'])  # note: the channel here has to be the same as dataset trained
    sample = make_sample(noise)

    output_dir = './generations/'
    # Loop through the batch of images
    for i in range(sample.size(0)):
        # Get the single image tensor
        image_tensor = sample[i]  # Shape: (channels, 32, 32)

        # Squeeze the tensor to remove the channel dimension
        image_tensor = image_tensor.squeeze(0)  # Shape: (32, 32)

        # Convert to a PIL Image
        image = Image.fromarray(image_tensor.cpu().numpy(), mode='L')  # 'L' mode for grayscale

        # Save the image
        image.save(os.path.join(output_dir, f'image_{i}.jpg'))
    return


if __name__ == "__main__":
    main()
