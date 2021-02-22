# coding:utf-8

import os.path as osp

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from config import config


class ModelNet40(Dataset):
    def __init__(self, transform=None, train=True, coord_label='n'):
        self.root_dir = config.root_dir

        self.train_files = self.get_files(osp.join(self.root_dir, 'train_files.txt'))
        self.eval_files = self.get_files(osp.join(self.root_dir, 'test_files.txt'))

        self.classes = self.get_files(osp.join(self.root_dir, 'shape_names.txt'))
        self.num_classes = len(self.classes)
        self.class_to_id = dict(zip(self.classes, range(self.num_classes)))

        self.train = train
        self.num_point = config.num_point

        self.coord_idx = {'x': 0, 'y': 1, 'z': 2, 'n': -1}
        self.coord_label = coord_label

        if self.train:
            self.train_data, self.train_label = self.get_data_and_label(self.train_files, self.coord_label)
        else:
            self.eval_data, self.eval_label = self.get_data_and_label(self.eval_files, self.coord_label)

        self.transform = transform

    def __getitem__(self, item):

        if self.train:
            train_data, train_label = self.train_data[item], self.train_label[item]
            if self.transform and np.random.random() > 0.5:
                train_data = self.transform(train_data)
            return torch.from_numpy(train_data), torch.tensor(train_label, dtype=torch.long)
        else:
            eval_data, eval_label = self.eval_data[item], self.eval_label[item]
            if self.transform:
                eval_data = self.transform(eval_data)
            return torch.from_numpy(eval_data), torch.tensor(eval_label, dtype=torch.long)

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.eval_data)

    def get_files(self, fp):
        print(fp)
        with open(fp, 'r') as f:
            files = f.readlines()
        return [f.rstrip() for f in files]

    def get_data_and_label(self, files, coord_label):
        all_data, all_label = [], []
        idx = self.coord_idx[coord_label]
        for fn in files:
            print('---------' + str(fn) + '---------')
            current_data, current_label = self.load_h5(osp.join(self.root_dir, fn))
            current_data = current_data[:, 0:self.num_point, :]
            if idx >= 0:
                current_data[:, :, idx] = 0.  # 指定轴的数据归0，相当于将点全部压到一个平面
            current_label = np.squeeze(current_label)
            all_data.append(current_data)
            all_label.append(current_label)

        return np.concatenate(all_data), np.concatenate(all_label)

    def load_h5(self, fp):
        f = h5py.File(fp)
        data = f['data'][:]
        label = f['label'][:]
        return (data, label)

    def load_h5_data_label_seg(self, fp):
        f = h5py.File(fp)
        data = f['data'][:]
        label = f['label'][:]
        seg = f['pid'][:]
        return (data, label, seg)
