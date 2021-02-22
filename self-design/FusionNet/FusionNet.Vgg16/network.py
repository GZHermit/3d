# coding:utf-8

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import config

import os


def square_distance(src, dst):
    """
    Description:
        just the simple Euclidean distance fomula，(x-y)^2,
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1).contiguous())
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def conv_bn(inp, oup, kernel, stride=1, padding=1, activation='relu'):
    seq = nn.Sequential(
        nn.Conv2d(inp, oup, kernel, stride, padding),
        nn.BatchNorm2d(oup)
    )
    if activation == 'relu':
        seq.add_module('2', nn.LeakyReLU())
    return seq


def fc_bn(inp, oup):
    return nn.Sequential(
        nn.Linear(inp, oup),
        nn.BatchNorm1d(oup),
        nn.LeakyReLU()
    )


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.nview = config.nview

        # ------------ network ------------ #

        self.conv1_1 = conv_bn(3, 64, [3, 3])
        self.conv1_2 = conv_bn(64, 64, [3, 3])

        self.conv2_1 = conv_bn(64, 128, [3, 3])
        self.conv2_2 = conv_bn(128, 128, [3, 3])

        self.conv3_1 = conv_bn(128, 256, [3, 3])
        self.conv3_2 = conv_bn(256, 256, [3, 3])
        self.conv3_3 = conv_bn(256, 256, [3, 3])

        self.conv4_1 = conv_bn(256, 512, [3, 3])
        self.conv4_2 = conv_bn(512, 512, [3, 3])
        self.conv4_3 = conv_bn(512, 512, [3, 3])

        self.conv5_1 = conv_bn(512, 512, [3, 3])
        self.conv5_2 = conv_bn(512, 512, [3, 3])
        self.conv5_3 = conv_bn(512, 512, [3, 3])

        self.fc6 = fc_bn(25088, 4096)
        self.fc7 = fc_bn(4096, 4096)
        self.fc8 = nn.Linear(4096, 512)

    def forward(self, x):
        """
            :param x: B * V * C * H * W, the different views with the same point cloud model
            :return: the global feature, the size is B * F
        """
        B, V, C, H, W = x.size()
        assert C == 3

        x = x.view(B * V, C, H, W)
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = F.max_pool2d(x, [2, 2], [2, 2])

        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = F.max_pool2d(x, [2, 2], [2, 2])

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = F.max_pool2d(x, [2, 2], [2, 2])

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = F.max_pool2d(x, [2, 2], [2, 2])

        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = F.max_pool2d(x, [2, 2], [2, 2])

        x = x.view((B, self.nview, -1))
        x = torch.max(x, dim=1)[0]

        x = self.fc6(x)
        x = self.fc7(x)
        x = self.fc8(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.side = 45
        self.m = self.side ** 2  # mostly close to 2048
        self.batch_size = config.batch_size

        # first folding layer
        self.conv1_1 = conv_bn(config.code_length + 2, 512, [1, 1], padding=0)
        self.conv1_2 = conv_bn(512, 512, [1, 1], padding=0)
        self.conv1_3 = conv_bn(512, 3, [1, 1], padding=0, activation=None)

        # second folding layer
        self.conv2_1 = conv_bn(config.code_length + 3, 512, [1, 1], padding=0)
        self.conv2_2 = conv_bn(512, 512, [1, 1], padding=0)
        self.conv2_3 = conv_bn(512, 3, [1, 1], padding=0, activation=None)

    def forward(self, x, grid_points):
        """
            :param x: B * F , F means the length of the global feature
            :return: generated point cloud, the size is (B, m, 3)
        """
        B, F = x.size()
        x = x.view((B, 1, F))
        xs = x.expand(-1, self.m, -1)

        x = torch.cat((grid_points, xs), 2)
        x = x.unsqueeze(3).permute(0, 2, 1, 3).contiguous()
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.conv1_3(x)

        x = x.squeeze(3).permute(0, 2, 1).contiguous()
        x = torch.cat((x, xs), 2)
        x = x.unsqueeze(3).permute(0, 2, 1, 3).contiguous()

        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_3(x)
        x = x.squeeze(3).permute(0, 2, 1).contiguous()
        return x


class FusionNet(nn.Module):
    # ResNet18/Vgg16 + FoldingNet
    def __init__(self):
        """
            input: B x N x 3
            output: B x 40
        """
        super(FusionNet, self).__init__()

        self.num_point = config.num_point
        self.num_classes = config.num_classes

        # --- encoder --- #
        self.encoder_net = Encoder()

        # --- decoder --- #
        self.decoder_net = Decoder()

        # --- classifier --- #
        self.fc1 = fc_bn(512, 512)
        self.fc2 = fc_bn(512, 256)
        self.fc3 = nn.Linear(256, self.num_classes)

        self.initialize_weights()

    def forward(self, pngs, grid):

        codeword = self.encoder_net(pngs)
        generated_pl = self.decoder_net(codeword, grid)
        x = self.fc1(codeword)
        x = self.fc2(x)
        x = self.fc3(x)

        return x, generated_pl

    @staticmethod
    def get_loss(cls_input, target, original_pl, generated_pl):

        """
            cls_input: B * Num_cls,
            target: B,
            original_pl: B * Num_point * 3
            generated_pl: B * Num_point * 3
        """
        classify_loss = nn.CrossEntropyLoss()(cls_input, target)

        o2g = square_distance(original_pl, generated_pl)
        o2g = torch.mean(torch.min(o2g, dim=2)[0], dim=1, keepdim=True)

        g2o = square_distance(generated_pl, original_pl)
        g2o = torch.mean(torch.min(g2o, dim=2)[0], dim=1, keepdim=True)

        reconstruct_loss = torch.mean(torch.max(torch.cat((o2g, g2o), dim=1), dim=1)[0])

        lambd = 1.
        loss = classify_loss + lambd * reconstruct_loss

        return loss, classify_loss, reconstruct_loss

    def initialize_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                # n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

        path = os.path.join('pretrained', 'vgg16.pth')
        pretrain = torch.load(path)
        rawnet = self.state_dict()

        for rkey in rawnet.keys():
            for pkey in pretrain.keys():
                if rawnet[rkey].shape == pretrain[pkey].shape and 'weight' in rkey and '0' in rkey:
                    rawnet[rkey] = pretrain[pkey]
                    rkey, pkey = rkey.rstrip('weight') + 'bias', pkey.rstrip('weight') + 'bias'
                    rawnet[rkey] = pretrain[pkey]
                    break
        self.load_state_dict(rawnet)


if __name__ == '__main__':
    # net = Decoder()
    # cls_input = torch.rand(2, 40)
    # target = (torch.rand(2) * 40).to(torch.int64)
    # generated_pl = torch.rand(2, 128, 3)
    # codeword = torch.rand(2, 512)
    # grid_points = torch.rand(2, 2025, 2)
    # res = net(codeword, grid_points)
    #

    # loss = FusionNet.get_loss(cls_input, target, generated_pl, original_pl)
    opl = torch.rand(2, 2048, 3)
    gpl = torch.rand(2, 2025, 3)
    a, b, c = FusionNet.get_loss(None, None, opl, gpl)