import torch
import torch.nn.functional as F
from collections import defaultdict
from torchmetrics.image.fid import FrechetInceptionDistance


def make_metric(split, **kwargs):
    data_name = kwargs['data_name']
    run_mode = kwargs['run_mode']
    metric_name = {k: [] for k in split}
    if data_name in ['MNIST', 'FashionMNIST', 'SVHN', 'CIFAR10', 'CIFAR100']:
        best = float('inf')
        best_direction = 'down'
        best_metric_name = 'Loss'
        for k in metric_name:
            if run_mode == 'train':
                metric_name[k].extend(['Loss', 'Loss_v', 'Loss_x0', 'Loss_x1'])
            else:
                metric_name[k].extend(['FID'])
    elif data_name in ['TwoMoons']:
        best = float('inf')
        best_direction = 'down'
        best_metric_name = 'Loss'
        for k in metric_name:
            metric_name[k].extend(['Loss', 'Loss_v', 'Loss_x0', 'Loss_x1'])

    else:
        raise ValueError('Not valid data name')
    metric = Metric(metric_name, best, best_direction, best_metric_name)
    return metric


def Accuracy(output, target, topk=1):
    with torch.no_grad():
        if target.dtype != torch.int64:
            target = (target.topk(1, -1, True, True)[1]).view(-1)
        batch_size = torch.numel(target)
        pred_k = output.topk(topk, -1, True, True)[1]
        correct_k = pred_k.eq(target.unsqueeze(-1).expand_as(pred_k)).float().sum()
        acc = (correct_k * (100.0 / batch_size)).item()
    return acc


def MSE(output, target):
    with torch.no_grad():
        mse = F.mse_loss(output, target).item()
    return mse


class RMSE:
    def __init__(self):
        self.reset()

    def reset(self):
        self.se = 0
        self.count = 0
        return

    def add(self, input, output):
        with torch.no_grad():
            self.se += F.mse_loss(output['target'], input['target'], reduction='sum')
            self.count += output['target'].numel()
        return

    def __call__(self, input, output):
        with torch.no_grad():
            rmse = ((self.se / self.count) ** 0.5).item()
        self.reset()
        return rmse


class FID:
    def __init__(self):
        self.fid = FrechetInceptionDistance(normalize=True)
        self.reset()

    def reset(self):
        self.fid.reset()
        return

    def add(self, input, output):
        with torch.no_grad():
            if input['data'].size(1) == 1:
                input_data = input['data'].expand(-1, 3, -1, -1)
                output_data = output['data'].expand(-1, 3, -1, -1)
            else:
                input_data = input['data']
                output_data = output['data']
            if input_data.device != self.fid.device:
                self.fid.to(input_data.device)
            self.fid.update(input_data, real=True)
            self.fid.update(output_data, real=False)
        return

    def __call__(self, input, output):
        with torch.no_grad():
            fid = float(self.fid.compute())
        self.reset()
        return fid


class Metric:
    def __init__(self, metric_name, best, best_direction, best_metric_name):
        self.metric_name = metric_name
        self.best, self.best_direction, self.best_metric_name = best, best_direction, best_metric_name
        self.metric = self.make_metric(metric_name)

    def make_metric(self, metric_name):
        metric = defaultdict(dict)
        for split in metric_name:
            for m in metric_name[split]:
                if m == 'Loss':
                    metric[split][m] = {'mode': 'batch', 'metric': (lambda input, output: output['loss'].item())}
                elif m == 'Loss_v':
                    metric[split][m] = {'mode': 'batch', 'metric': (lambda input, output: output['loss_v'].item())}
                elif m == 'Loss_x0':
                    metric[split][m] = {'mode': 'batch', 'metric': (lambda input, output: output['loss_x0'].item())}
                elif m == 'Loss_x1':
                    metric[split][m] = {'mode': 'batch', 'metric': (lambda input, output: output['loss_x1'].item())}
                elif m == 'Accuracy':
                    metric[split][m] = {'mode': 'batch',
                                        'metric': (lambda input, output: Accuracy(output['target'], input['target']))}
                elif m == 'MSE':
                    metric[split][m] = {'mode': 'batch',
                                        'metric': (lambda input, output: MSE(output['target'], input['target']))}
                elif m == 'RMSE':
                    metric[split][m] = {'mode': 'full', 'metric': RMSE()}
                elif m == 'FID':
                    metric[split][m] = {'mode': 'full', 'metric': FID()}
                else:
                    raise ValueError('Not valid metric name')
        return metric

    def add(self, split, input, output):
        for metric_name in self.metric_name[split]:
            if self.metric[split][metric_name]['mode'] == 'full':
                self.metric[split][metric_name]['metric'].add(input, output)
        return

    def evaluate(self, split, mode, input, output, metric_name):
        evaluation = {}
        for metric_name_i in metric_name[split]:
            if self.metric[split][metric_name_i]['mode'] == mode:
                evaluation[metric_name_i] = self.metric[split][metric_name_i]['metric'](input, output)
        return evaluation

    def compare(self, val, if_update):
        if self.best_direction == 'down':
            compared = self.best > val
        elif self.best_direction == 'up':
            compared = self.best < val
        else:
            raise ValueError('Not valid best direction')
        if compared and if_update:
            self.best = val
        return compared

    def load_state_dict(self, state_dict):
        self.best = state_dict['best']
        self.best_metric_name = state_dict['best_metric_name']
        self.best_direction = state_dict['best_direction']
        return

    def state_dict(self):
        return {'best': self.best, 'best_metric_name': self.best_metric_name, 'best_direction': self.best_direction}
