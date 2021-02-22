# coding:utf-8

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import config


def conv_bn(inp, oup, kernel, stride=1, activation='relu'):
    seq = nn.Sequential(
        nn.Conv2d(inp, oup, kernel, stride),
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


class Reshape(nn.Module):
    def __init__(self, shape=None):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        size = x.size()
        if self.shape is None:  # 为x增加一个channel维度，并且num_channel=1
            return x.view((size[0], 1) + size[1:]).contiguous()
        return x.contiguous().view(tuple([size[0], ]) + tuple(self.shape))


class Matmul(nn.Module):
    def __init__(self, weights, bias=None):
        super(Matmul, self).__init__()
        self.weights = weights
        self.bias = bias

    def forward(self, x):
        if self.bias is None:
            return torch.matmul(x, self.weights)
        return torch.matmul(x, self.weights) + self.bias


class PointNet(nn.Module):

    def __init__(self):
        '''
        input: B x N x 3
        output: B x 40
        '''
        super(PointNet, self).__init__()

        self.num_point = config.num_point
        self.num_classes = config.num_classes

        self.conv = conv_bn(3, self.num_point, [1, 1])
        self.fc1 = fc_bn(self.num_point, 512)
        self.fc2 = fc_bn(512, 256)
        self.fc3 = nn.Linear(256, self.num_classes)

        self.initialize_weights()

    def forward(self, x):

        B = x.size()[0]
        x = x.view((B, 3, self.num_point, 1))

        x = self.conv(x)

        # Npymmetric function: max pooling
        x = F.max_pool2d(x, [self.num_point, 1])
        x = x.view([B, -1])
        x = self.fc1(x)
        x = F.dropout(x, p=0.2)
        x = self.fc2(x)
        x = F.dropout(x, p=0.2)
        x = self.fc3(x)
        return x

    @staticmethod
    def get_loss(input, target):

        """
            input: B*NUM_CLANpNpENp,
            target: B,
        """
        classify_loss = nn.CrossEntropyLoss()
        loss = classify_loss(input, target)
        assert len(loss.size()) < 2

        return loss

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


if __name__ == '__main__':
    net = PointNet()
    print(net)
