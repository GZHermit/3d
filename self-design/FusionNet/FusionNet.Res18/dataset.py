# coding:utf-8

import os.path as osp

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from config import config

import json

from PIL import Image
import cv2

W = H = 224


class ModelNet40(Dataset):
    def __init__(self, point_transform=None, view_transform=None, train=True):

        self.train = train
        self.point_dir = config.point_dir
        self.view_dir = config.view_dir
        self.nview = config.nview

        # ------------ point cloud read ------------ #

        self.point_train_files = self.get_files(osp.join(self.point_dir, 'train_files.txt'))
        self.point_eval_files = self.get_files(osp.join(self.point_dir, 'test_files.txt'))

        self.num_point = config.num_point
        self.classes = self.get_files(osp.join(self.point_dir, 'shape_names.txt'))
        self.num_classes = len(self.classes)
        self.class_to_id = dict(zip(self.classes, range(self.num_classes)))

        if self.train:
            self.point_train_data, self.point_train_label, self.view_train_files = self.get_point_and_label(
                self.point_train_files)
        else:
            self.point_eval_data, self.point_eval_label, self.view_eval_files = self.get_point_and_label(
                self.point_eval_files)

        self.point_transform = point_transform

        # ------------ multi-view projection pngs read ------------ #

        self.view_eval_files, self.view_eval_label = self.get_views_and_label(osp.join(self.view_dir, 'test_lists.txt'))

        self.view_transform = view_transform

    def __getitem__(self, item):
        if self.train:
            point_train_data, point_train_label = self.point_train_data[item], self.point_train_label[item]
            if self.point_transform:
                point_train_data = self.point_transform(point_train_data)

            view_train_file = self.view_train_files[item]
            view_train_data, view_train_label = self.get_png(view_train_file, self.view_transform)
            assert view_train_label == point_train_label

            return torch.from_numpy(point_train_data), torch.tensor(point_train_label, dtype=torch.long), \
                   view_train_data, torch.tensor(view_train_label, dtype=torch.long)

        else:
            # view_eval_file = self.view_eval_files[item]
            # view_eval_data, view_eval_label = self.get_png(view_eval_file, self.view_transform)
            #
            # point_eval_label = self.point_eval_label[item]
            # assert view_eval_label == point_eval_label
            #
            # return view_eval_data, torch.tensor(view_eval_label, dtype=torch.long)

            # ------------------------- #
            view_eval_file, view_eval_label = self.view_eval_files[item], self.view_eval_label[item]
            view_eval_data, label2 = self.get_png(view_eval_file, self.view_transform)

            assert view_eval_label == label2

            return view_eval_data, torch.tensor(view_eval_label, dtype=torch.long)

    def __len__(self):
        if self.train:
            return len(self.view_train_files)
        else:
            return len(self.view_eval_files)

    def get_files(self, fp):
        print(fp)
        with open(fp, 'r') as f:
            files = f.readlines()
        return [f.rstrip() for f in files]

    def get_point_and_label(self, files):
        all_point_data, all_point_label = [], []
        view_files = []
        for i, fn in enumerate(files):
            json_fn = fn.rstrip(str(i) + '.h5') + '_{}_id2file.json'.format(i)

            current_point_data, current_point_label = self.load_h5(osp.join(self.point_dir, fn))

            current_point_data = current_point_data[:, 0:self.num_point, :]
            current_point_label = np.squeeze(current_point_label)
            current_view_files = self.load_json(osp.join(self.point_dir, json_fn))
            view_files.extend(current_view_files)

            all_point_data.append(current_point_data)
            all_point_label.append(current_point_label)

        return np.concatenate(all_point_data), np.concatenate(all_point_label), view_files

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
            label = int(pngs[0].rstrip('\n'))
            for i in range(2, len(pngs)):
                im = Image.open(pngs[i].rstrip('\n')).convert('RGB')
                if transform is not None:
                    im = transform(im)
                pngs_tensor.append(im)
        return torch.stack(pngs_tensor), label

    def load_json(self, fp):
        content = json.load(open(fp, 'r'))
        flag = 'train' if self.train else 'test'
        content = [config.view_dir + '/list/' + flag + '/' + f.rstrip('ply') + 'off.txt' for f in content]
        return content

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
    import torchvision.transforms as transforms

    point_transform = None
    view_transform = transforms.Compose([
        transforms.CenterCrop(500),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = ModelNet40(point_transform, view_transform, train=False)

    for data in dataset:
        p, pl, v, vl = data
        print(p.shape)
        print(pl)
        print(v.shape)
        print(vl)
        exit()
