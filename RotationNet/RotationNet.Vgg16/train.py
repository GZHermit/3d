# coding: utf-8
import math
import os
import argparse

import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import utils
from config import config
from dataset import ModelNet40
from network import FusionNet

import numpy as np


def lr_ladder_decay(optimizer, trained_epoch):
    if trained_epoch % 20 == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = max(param_group['lr'] * 0.5, config.lr_decay)


def lr_exponential_decay(optimizer, global_step, decay_rate, decay_step):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * math.pow(decay_rate, math.floor(global_step / float(decay_step)))
        param_group['lr'] = max(param_group['lr'], config.lr_decay)


def start(args):

    # point_transform = transforms.Compose([RandomIndexInBatch(), RotatePointCloud(), JitterPointCloud()])
    point_transform = None
    view_transform = transforms.Compose([
        transforms.CenterCrop(500),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = ModelNet40(point_transform, view_transform)

    model = FusionNet()

    if config.cuda and torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            print("we will use {} GPUs!".format(torch.cuda.device_count()))
            model = torch.nn.DataParallel(model)
            config.batch_size *= torch.cuda.device_count()
        model.cuda()

    model.train()

    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, drop_last=True)

    weight, bias = [], []
    for name, param in model.named_parameters():
        if 'bias' in name:
            bias.append(param)
        elif 'weight' in name:
            weight.append(param)

    optimizer = optim.Adam([{'params': weight, 'weight_decay': config.weight_decay},
                            {'params': bias, 'weight_decay': 0.}],
                           lr=config.learning_rate)

    utils.check_filepath(config.save_path, clean=args.clean)
    print("save path:{}".format(config.save_path))
    saved_models = os.listdir(config.save_path)
    last_epoch = 1

    if len(saved_models) > 0:
        saved_epochs = [int(model.rstrip('.pth').split('_')[-1]) for model in saved_models]
        last_epoch = max(saved_epochs)
        saved_model = os.path.join(config.save_path, 'cls_model_{0:03d}.pth'.format(last_epoch))
        print("load weight: {}".format(saved_model))
        model.load_state_dict(torch.load(saved_model))

    last_epoch = min(last_epoch, args.start_epoch)

    # grid_points = torch.tensor(np.random.uniform(-1., 1., (config.batch_size, 2025, 2)), dtype=torch.float32)
    grid_points = torch.tensor(np.random.uniform(-0.3, 0.3, (config.batch_size, 2025, 2)), dtype=torch.float32)
    for epoch in range(last_epoch, config.max_epoch + 1):

        total_train_loss = 0.
        total_classify_loss = 0.
        total_reconstruct_loss = 0.
        total_correct = 0

        lr_ladder_decay(optimizer, epoch)

        for i, data in enumerate(dataloader):
            point_cloud, label, view_data, _ = data
            if config.cuda:
                point_cloud, label, view_data = point_cloud.cuda(), label.cuda(), view_data.cuda()
                grid_points = grid_points.cuda()

            optimizer.zero_grad()
            pred, generated_pl = model(view_data, grid_points)
            train_loss, classify_loss, reconstruct_loss = FusionNet.get_loss(pred, label, point_cloud, generated_pl)  # because the dataparallel
            train_loss.backward()
            optimizer.step()

            # lr_exponential_decay(optimizer, epoch * config.train_totality + i*config.batch_size, config.decay_rate, config.decay_step)

            result = pred.max(1)[1]
            correct = result.eq(label).cpu().sum()
            total_correct += correct.item()
            total_train_loss += train_loss.item()
            total_classify_loss += classify_loss.item()
            total_reconstruct_loss += reconstruct_loss.item()
            if i % 16 == 0:
                print('[%d: %d/%d] lr: %f train loss: %f  classify_loss: %f reconstruct_loss: %f accuracy: %f ' % (
                    epoch, i * config.batch_size, config.train_totality, optimizer.param_groups[0]['lr'],
                    train_loss.item(),classify_loss.item(), reconstruct_loss.item(),
                    correct.item() / float(config.batch_size)))
        print('[- %d -] train loss: %f classify_loss: %f reconstruct_loss: %f accuracy: %.4f ' % (
            epoch, total_train_loss / len(dataloader), total_classify_loss / len(dataloader), total_reconstruct_loss / len(dataloader), float(total_correct) / config.train_totality))
        if epoch % 2 == 0:
            torch.save(model.state_dict(), '%s/cls_model_%03d.pth' % (config.save_path, epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--clean', type=bool, default=True, help='the flag of cleaning the save files or not')
    parser.add_argument('--start_epoch', type=int, default=250, help='from the specified epoch to train/val')

    args = parser.parse_args()

    start(args)
