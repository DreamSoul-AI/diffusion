import argparse
import os
import torch
import torch.backends.cudnn as cudnn
from config import cfg, process_args
from dataset import make_dataset, make_data_loader, process_dataset
from metric import make_logger
from model import make_model, Sampler
from module import save, resume, to_device, process_control, ddim_sample_loop_V

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
    dataset = make_dataset(cfg['data_name'])
    dataset = process_dataset(dataset)
    model = make_model(cfg['model'])
    result = resume(cfg['best_path'])
    cfg['step'] = result['cfg']['step']
    model = model.to(cfg['device'])
    model.load_state_dict(result['model'])
    data_loader = make_data_loader(dataset, cfg[cfg['tag']]['optimizer']['batch_size'])
    test_logger = make_logger(cfg['logger_path'], data_name=cfg['data_name'], run_mode=cfg['run_mode'])
    test(data_loader['test'], model, test_logger)
    result = resume(cfg['checkpoint_path'])
    result = {'cfg': cfg, 'logger': {'train': result['logger'],
                                     'test': test_logger.state_dict()}}
    save(result, cfg['result_path'])
    return


def test(data_loader, model, logger):
    with torch.no_grad():
        sampler = Sampler(cfg['generate']['num_steps'], cfg['generate']['guidance_scale'], cfg['generate']['eta'])
        model.train(False)
        for i, input in enumerate(data_loader):
            input_size = input['data'].size(0)
            input = to_device(input, cfg['device'])
            noise = torch.randn(input['data'].size()).to(cfg['device'])
            if sampler.guidance_scale > 1:
                classes = input['target']
            else:
                classes = -input['data'].new_ones((input_size,), dtype=torch.long)
            samples = sampler.sample(noise, model, classes)
            input = {'data': sampler.apply_normalize(input['data'], -1, 1)}
            output = {'data': samples}
            evaluation = logger.evaluate('test', 'batch', input, output)
            logger.append(evaluation, 'test', input_size)
            logger.add('test', input, output)
        evaluation = logger.evaluate('test', 'full')
        logger.append(evaluation, 'test', input_size)
        info = {'info': ['Model: {}'.format(cfg['tag']),
                         'Test Epoch: {}({:.0f}%)'.format(cfg['step'] // cfg['eval_period'], 100.)]}
        logger.append(info, 'test')
        print(logger.write('test'))
        logger.save(True)
    return


if __name__ == "__main__":
    main()
