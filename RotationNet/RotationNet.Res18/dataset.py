# coding:utf-8

import os.path as osp

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from config import config

from PIL import Image
import cv2
W = H = 224


class ModelNet40(Dataset):
    def __init__(self, point_transform=None, view_transform=None, train=True):

        self.train = train

        # ------------ point cloud read ------------ #

        self.point_dir = config.point_dir
        self.point_train_files = self.get_files(osp.join(self.point_dir, 'train_files.txt'))
        self.point_eval_files = self.get_files(osp.join(self.point_dir, 'test_files.txt'))

        self.num_point = config.num_point
        self.classes = self.get_files(osp.join(self.point_dir, 'shape_names.txt'))
        self.num_classes = len(self.classes)
        self.class_to_id = dict(zip(self.classes, range(self.num_classes)))

        if self.train:
            self.point_train_data, self.point_train_label = self.get_point_and_label(self.point_train_files)
        else:
            self.point_eval_data, self.point_eval_label = self.get_point_and_label(self.point_eval_files)

        self.point_transform = point_transform

        # ------------ multi-view projection pngs read ------------ #

        self.view_dir = config.view_dir

        self.view_train_files, self.view_train_label = self.get_views_and_label(
            osp.join(self.view_dir, 'train_lists.txt'))
        self.view_val_files, self.view_val_label = self.get_views_and_label(osp.join(self.view_dir, 'val_lists.txt'))
        self.view_eval_files, self.view_eval_label = self.get_views_and_label(osp.join(self.view_dir, 'test_lists.txt'))
        self.nview = config.nview
        if self.train:
            self.view_train_files.extend(self.view_val_files)
            self.view_train_label.extend(self.view_val_label)
            # self.view_train_data = self.get_pngs(self.view_train_files)
            # self.view_val_data = self.get_pngs(self.view_val_files)
            # self.view_train_data = np.concatenate([self.view_train_data, self.view_val_data])
        else:
            self.view_eval_data = self.get_pngs(self.view_eval_data)

        self.view_transform = view_transform

    def __getitem__(self, item):
        if self.train:
            point_train_data, point_train_label = self.point_train_data[item], self.point_train_label[item]
            if self.point_transform:
                point_train_data = self.point_transform(point_train_data)

            if self.train:
                view_train_file, view_train_label = self.view_train_files[item], self.view_train_label[item]
                view_train_data = self.get_png(view_train_file, self.view_transform)

            return torch.from_numpy(point_train_data), torch.tensor(point_train_label, dtype=torch.long), \
                   view_train_data, torch.tensor(view_train_label, dtype=torch.long)

        else:
            point_eval_data, point_eval_label = self.point_eval_data[item], self.point_eval_label[item]
            if self.point_transform:
                point_eval_data = self.point_transform(point_eval_data)

            view_eval_data, view_eval_label = self.view_eval_data[item], self.view_eval_label[item]
            if self.view_transform:
                view_eval_data = self.view_transform(view_eval_data)

            return torch.from_numpy(point_eval_data), torch.tensor(point_eval_label, dtype=torch.long), \
                   view_eval_data, torch.tensor(view_eval_label, dtype=torch.long)

    def __len__(self):
        if self.train:
            return len(self.point_train_data)
        else:
            return len(self.point_eval_data)

    def get_files(self, fp):
        print(fp)
        with open(fp, 'r') as f:
            files = f.readlines()
        return [f.rstrip() for f in files]

    def get_point_and_label(self, files):
        all_data, all_label = [], []
        for fn in files:
            print('---------' + str(fn) + '---------')
            current_data, current_label = self.load_h5(osp.join(self.point_dir, fn))
            current_data = current_data[:, 0:self.num_point, :]
            current_label = np.squeeze(current_label)
            all_data.append(current_data)
            all_label.append(current_label)

        return np.concatenate(all_data), np.concatenate(all_label)

    def get_views_and_label(self, fp):
        lines = np.loadtxt(fp, dtype=str).tolist()
        all_views, all_labels = zip(*[(l[0], int(l[1])) for l in lines])

        return list(all_views), list(all_labels)

    def get_pngs(self, files):
        all_pngs = []
        for file in files:
            print(file)
            with open(file, 'r') as f:
                pngs = f.readlines()
                pngs_array = []
                for i in range(2, len(pngs)):
                    im = cv2.imread(pngs[i].rstrip('\n'))
                    im = cv2.resize(im, (H, W))
                    # im = im.transpose(2, 0, 1)
                    pngs_array.append(im)
                pngs_array = np.array(pngs_array)
            all_pngs.append(pngs_array)
        return np.stack(all_pngs)

    def get_png(self, file, transform=None):
        pngs_tensor = []
        with open(file, 'r') as f:
            pngs = f.readlines()
            for i in range(2, len(pngs)):
                im = Image.open(pngs[i].rstrip('\n')).convert('RGB')
                if transform is not None:
                    im = transform(im)
                pngs_tensor.append(im)
        return torch.stack(pngs_tensor)

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


if __name__ == '__main__':
    dataset = ModelNet40()
