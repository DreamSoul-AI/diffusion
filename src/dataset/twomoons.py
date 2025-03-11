import codecs
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from module import check_exists, makedir_exist_ok, save, load
from .utils import download_url, extract_file, make_classes_counts


class TwoMoons(Dataset):
    data_name = 'Moon'

    def __init__(self, root, split, transform=None):
        self.root = os.path.expanduser(root)
        self.split = split
        self.transform = transform
        if not check_exists(self.processed_folder):
            self.process()
        self.id, self.data, self.target = load(os.path.join(self.processed_folder, self.split))
        self.other = {}
        self.classes_counts = make_classes_counts(self.target)
        self.data_size, self.target_size, self.classes_to_labels = load(os.path.join(self.processed_folder, 'meta'))

    def __getitem__(self, index):
        id, data, target = torch.tensor(self.id[index]), torch.tensor(self.data[index]), torch.tensor(
            self.target[index])
        input = {'id': id, 'data': data, 'target': target}
        other = {k: torch.tensor(self.other[k][index]) for k in self.other}
        input = {**input, **other}
        if self.transform is not None:
            input = self.transform(input)
        return input

    def __len__(self):
        return len(self.data)

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')

    def process(self):
        if not check_exists(self.raw_folder):
            self.download()
        train_set, test_set, meta = self.make_data()
        save(train_set, os.path.join(self.processed_folder, 'train'))
        save(test_set, os.path.join(self.processed_folder, 'test'))
        save(meta, os.path.join(self.processed_folder, 'meta'))
        return

    def download(self):
        makedir_exist_ok(self.raw_folder)
        return

    def __repr__(self):
        fmt_str = 'Dataset {}\nSize: {}\nRoot: {}\nSplit: {}\nTransforms: {}'.format(
            self.__class__.__name__, self.__len__(), self.root, self.split, self.transform.__repr__())
        return fmt_str

    def make_data(self):
        from sklearn.datasets import make_moons
        num_samples = 256
        noise = 0.05
        X, y = make_moons(n_samples=num_samples, noise=noise, random_state=0)
        perm = np.random.permutation(len(X))
        X, y = X[perm], y[perm]
        split_idx = int(X.shape[0] * 0.8)
        train_data, test_data = X[:split_idx].astype(np.float32), X[split_idx:].astype(np.float32)
        train_target, test_target = y[:split_idx].astype(np.int64), y[split_idx:].astype(np.int64)
        train_id, test_id = np.arange(len(train_data)).astype(np.int64), np.arange(len(test_data)).astype(np.int64)
        data_size = (2,)
        classes = list(map(str, list(range(10))))
        classes_to_labels = {classes[i]: i for i in range(len(classes))}
        target_size = len(classes)
        return (train_id, train_data, train_target), (test_id, test_data, test_target), (
            data_size, target_size, classes_to_labels)

