# coding: utf-8

import argparse
import os

import torch

from dataset import ModelNet40
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import config
from network import FusionNet
import torchvision.transforms as transforms

import numpy as np


def start(args):
    point_transform = None

    view_transform = transforms.Compose([
        transforms.CenterCrop(500),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = ModelNet40(point_transform, view_transform, train=False)

    model = FusionNet()

    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)

    if config.cuda and torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            print("we will use {} GPUs!".format(torch.cuda.device_count()))
            model = torch.nn.DataParallel(model)
            config.batch_size *= torch.cuda.device_count()
        model.cuda()

    model.eval()

    max_correct, mac_accuracy = 0, 0.
    grid_points = torch.tensor(np.random.uniform(-0.3, 0.3, (1, 2025, 2)), dtype=torch.float32)

    saved_models = os.listdir(config.save_path)

    for sm in saved_models:

        saved_epoch = int(sm.rstrip('.pth').split('_')[-1])
        if saved_epoch < args.epoch: continue

        saved_model = os.path.join(config.save_path, sm)
        model.load_state_dict(torch.load(saved_model))

        correct = torch.tensor(0)
        with tqdm(dataloader) as pbar:
            for i, data in enumerate(pbar):
                views, label = data
                if config.cuda:
                    views, label = views.cuda(), label.cuda()
                    grid_points = grid_points.cuda()
                pred, _ = model(views, grid_points)
                result = pred.max(1)[1]
                correct += result.eq(label).cpu().sum().item()
                pbar.update(i)

        print('{}, total:{}, correct:{}, accuracy: {} '.format(sm.strip('.pth'), config.eval_totality, correct,
                                                               float(correct) / config.eval_totality))
        mac_accuracy = max(mac_accuracy, float(correct) / config.eval_totality)
        max_correct = max(max_correct, correct)

    print("max_acc:{}, max_correct:{}".format(mac_accuracy, max_correct))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', type=int, default=0, help='from the specified epoch to train/val')

    args = parser.parse_args()

    start(args)
