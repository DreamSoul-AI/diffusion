import argparse
import datetime
import os
import shutil
import time
import torch
import torch.backends.cudnn as cudnn
from config import cfg, process_args
from dataset import make_dataset, make_data_loader, process_dataset
from metric import make_logger
from model import make_model, make_optimizer, make_scheduler
from module import check, resume, to_device, process_control
from torch.utils import data
from torchvision import datasets, transforms, utils
from torchvision.transforms import functional as TF
from tqdm.notebook import tqdm, trange

cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
args = vars(parser.parse_args())
process_args(args)

rng = torch.quasirandom.SobolEngine(1, scramble=True)
scaler = torch.cuda.amp.GradScaler()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    cfg['path'] = os.path.join('output', 'exp')
    cfg['tag_path'] = os.path.join(cfg['path'], cfg['tag'])
    cfg['checkpoint_path'] = os.path.join(cfg['tag_path'], 'checkpoint')
    cfg['best_path'] = os.path.join(cfg['tag_path'], 'best')
    cfg['logger_path'] = os.path.join('output', 'logger', 'train', 'runs', cfg['tag'])
    dataset = make_dataset(cfg['data_name'])
    dataset = process_dataset(dataset)
    model = make_model(cfg['model'])
    result = resume(cfg['checkpoint_path'], resume_mode=cfg['resume_mode'])
    if result is None:
        cfg['step'] = 0
        model = model.to(cfg['device'])
        optimizer = make_optimizer(model.parameters(), cfg[cfg['tag']]['optimizer'])
        scheduler = make_scheduler(optimizer, cfg[cfg['tag']]['optimizer'])
        logger = make_logger(cfg['logger_path'], data_name=cfg['data_name'])
    else:
        cfg['step'] = result['cfg']['step']
        model = model.to(cfg['device'])
        optimizer = make_optimizer(model.parameters(), cfg[cfg['tag']]['optimizer'])
        scheduler = make_scheduler(optimizer, cfg[cfg['tag']]['optimizer'])
        logger = make_logger(cfg['logger_path'], data_name=cfg['data_name'])
        model.load_state_dict(result['model'])
        optimizer.load_state_dict(result['optimizer'])
        scheduler.load_state_dict(result['scheduler'])
        logger.load_state_dict(result['logger'])
        logger.reset()
    #data_loader = make_data_loader(dataset, cfg[cfg['tag']]['optimizer']['batch_size'], cfg['num_steps'],
                                   #cfg['step'], cfg['step_period'], cfg['pin_memory'], cfg['num_workers'],
                                   #cfg['collate_mode'], cfg['seed'])
    #data_iterator = enumerate(data_loader['train'])
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    train_set = datasets.CIFAR10('data', train=True, download=True, transform=tf)
    train_dl = data.DataLoader(train_set, 100, shuffle=True,
                           num_workers=2, persistent_workers=True, pin_memory=True)
    val_set = datasets.CIFAR10('data', train=False, download=True, transform=tf)
    val_dl = data.DataLoader(val_set, 100,
                            num_workers=2, persistent_workers=True, pin_memory=True)
    data_iterator = enumerate(tqdm(train_dl))
    while cfg['step'] < cfg['num_steps']:
        train(data_iterator, model, optimizer, scheduler, logger)
        #test(data_loader['test'], model, logger)
        result = {'cfg': cfg, 'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                  'logger': logger.state_dict()}
        check(result, cfg['checkpoint_path'])
        if logger.compare('test'):
            shutil.copytree(cfg['checkpoint_path'], cfg['best_path'], dirs_exist_ok=True)
        logger.reset()
    return


def train(data_loader, model, optimizer, scheduler, logger):
    #model.train(True)
    model.to(device)
    start_time = time.time()
    with logger.profiler:
        for i, (reals, classes) in data_loader:
            optimizer.zero_grad()
            if i % cfg['step_period'] == 0 and cfg['profile']:
                logger.profiler.step()
            #input_size = input['data'].size(0)
            reals = reals.to(device)
            input = reals
            classes = classes.to(device)
            t = rng.draw(reals.shape[0])[:, 0].to(device)
            output = model(reals, t, classes)
            loss = output['loss']
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if i % 50 == 0:
                tqdm.write(f'Iteration: {i}, loss: {loss.item():g}')

            if (i + 1) % cfg['step_period'] == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            evaluation = logger.evaluate('train', 'batch', input, output)
            logger.append(evaluation, 'train', n=input_size)
            idx = cfg['step'] % cfg['eval_period']
            if idx % max(int(cfg['eval_period'] * cfg['log_interval']), 1) == 0 and (i + 1) % cfg['step_period'] == 0:
                step_time = (time.time() - start_time) / (idx + 1)
                lr = optimizer.param_groups[0]['lr']
                epoch_finished_time = datetime.timedelta(
                    seconds=round((cfg['eval_period'] - (idx + 1)) * step_time))
                exp_finished_time = datetime.timedelta(
                    seconds=round((cfg['num_steps'] - (cfg['step'] + 1)) * step_time))
                info = {'info': ['Model: {}'.format(cfg['tag']),
                                 'Train Epoch: {}({:.0f}%)'.format((cfg['step'] // cfg['eval_period']) + 1,
                                                                   100. * idx / cfg['eval_period']),
                                 'Learning rate: {:.6f}'.format(lr),
                                 'Epoch Finished Time: {}'.format(epoch_finished_time),
                                 'Experiment Finished Time: {}'.format(exp_finished_time)]}
                logger.append(info, 'train')
                print(logger.write('train'))
            if (i + 1) % cfg['step_period'] == 0:
                cfg['step'] += 1
            if (idx + 1) % cfg['eval_period'] == 0 and (i + 1) % cfg['step_period'] == 0:
                break
    return


def test(data_loader, model, logger):
    with torch.no_grad():
        model.train(False)
        for i, input in enumerate(data_loader):
            input_size = input['data'].size(0)
            input = to_device(input, cfg['device'])
            t = rng.draw(input['data'].shape[0])[:, 0].to(cfg['device'])
            output = model(input['data'], t, input['target'])
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
