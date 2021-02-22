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


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        self.inplanes = 64
        self.num_classes = config.num_classes

        block = BasicBlock
        layers = [2, 2, 2, 2]

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)  # nn.AdaptiveAvgPool2d(7)

        self.fc = nn.Linear(512 * block.expansion, self.num_classes)

    def forward(self, x):
        # Swap batch and views dims
        x = x.transpose(0, 1)

        # View pool
        view_pool = []

        for v in x:
            v = self.conv1(v)
            v = self.bn1(v)
            v = self.relu(v)
            v = self.maxpool(v)

            v = self.layer1(v)
            v = self.layer2(v)
            v = self.layer3(v)
            v = self.layer4(v)

            v = self.avgpool(v)
            v = v.view(v.size(0), -1)
            view_pool.append(v)

        pooled_view = view_pool[0]
        for i in range(1, len(view_pool)):
            pooled_view = torch.max(pooled_view, view_pool[i])

        return pooled_view

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


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

        assert grid_points.shape[0] == xs.shape[0]

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
        self.encoder = Encoder()

        # --- decoder --- #
        self.decoder = Decoder()

        # --- classifier --- #

        self.initialize_weights()

    def forward(self, pngs, grid):

        codeword = self.encoder(pngs)
        x = self.encoder.fc(codeword)
        generated_pl = self.decoder(codeword, grid)
        return x, generated_pl

    @staticmethod
    def get_loss(pred, target, original_pl, generated_pl):

        """
            cls_input: B * Num_cls,
            target: B,
            original_pl: B * Num_point * 3
            generated_pl: B * Num_point * 3
        """
        classify_loss = nn.CrossEntropyLoss()(pred, target)

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

        path = os.path.join('pretrained', 'cls_model_066.pth')

        if os.path.exists(path):
            pretrain = torch.load(path)
            print("initialize weight in path:", path)
            rawnet = self.state_dict()
            for rkey in rawnet.keys():
                for pkey in pretrain.keys():
                    if pkey.lstrip('module') == rkey.lstrip('encoder') and rawnet[rkey].shape == pretrain[pkey].shape:
                        rawnet[rkey] = pretrain[pkey]
                        break
            self.load_state_dict(rawnet)
        else:
            print('randomly initialize the weight of the model!')


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
    # opl = torch.rand(2, 2048, 3)
    # gpl = torch.rand(2, 2025, 3)
    # a, b, c = FusionNet.get_loss(None, None, opl, gpl)

    views = torch.rand(2, 12, 3, 224, 224)
    net = FusionNet()
    grid_points = torch.tensor(np.random.uniform(-0.3, 0.3, (config.batch_size, 2025, 2)), dtype=torch.float32)
    ov, op = net(views, grid_points)
    # print(ov.shape, op.shape)
