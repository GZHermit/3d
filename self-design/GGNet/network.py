# coding:utf-8

import math

import numpy as np
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


class Correlation(nn.Module):
    def __init__(self, channel):
        super(Correlation, self).__init__()

        self.corr_bypass = nn.Sequential(nn.Conv1d(channel, channel, kernel_size=3, padding=1),
                                         nn.BatchNorm1d(channel),
                                         nn.ReLU())
        self.extra_1 = nn.Sequential(nn.Conv1d(config.num_point, channel, kernel_size=3, padding=1),
                                     nn.BatchNorm1d(channel),
                                     nn.ReLU())
        self.extra_2 = nn.Sequential(nn.Conv1d(2 * channel, channel, kernel_size=3, padding=1),
                                     nn.BatchNorm1d(channel),
                                     nn.ReLU())

    def forward(self, x):

        x = x.squeeze(3)
        batch_size, channel_size, point_size = x.size()
        bypass = self.corr_bypass(x)

        res = x.new_tensor(torch.eye(point_size).repeat(batch_size, 1, 1))
        for i in range(point_size - 1):
            res[:, i, i + 1:] = F.cosine_similarity(x[:, :, i].unsqueeze(2), x[:, :, i + 1:])
            res[:, i + 1:, i] = res[:, i, i + 1:]
        x = self.extra_1(res)
        x = torch.cat([x, bypass], dim=1)
        x = self.extra_2(x)

        x = x.unsqueeze(3)
        return x


class GGNet(nn.Module):

    def __init__(self):
        '''
        input: B x N x 3
        output: B x 40
        '''
        super(GGNet, self).__init__()

        self.num_point = config.num_point
        self.num_classes = config.num_classes
        self.K = config.K

        self.conv1 = conv_bn(1, 64, [1, 3])
        self.conv2 = conv_bn(64, 128, [1, 1])
        self.conv3 = conv_bn(128, 256, [1, 1])
        self.conv4 = conv_bn(256, 256, [1, 1])
        self.conv5 = conv_bn(256, 512, [1, 1])

        self.correlate = Correlation(512)

        self.fc1 = fc_bn(512, 512)
        self.fc2 = fc_bn(512, 256)
        self.fc3 = nn.Linear(256, self.num_classes)

        # self.I = nn.Parameter(torch.tensor(np.eye(config.K), dtype=torch.float, requires_grad=False), requires_grad=False)

        self.initialize_weights()

    def forward(self, x):

        B = x.size()[0]
        x = x.view((B, 1, self.num_point, 3))
        # x = x.view((B, 1, 33, 3))

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.correlate(x)
        # Npymmetric function: max pooling
        x = F.max_pool2d(x, [self.num_point, 1])
        x = x.view([B, -1])
        x = self.fc1(x)
        x = F.dropout(x, p=0.3)
        x = self.fc2(x)
        x = F.dropout(x, p=0.3)
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
            elif isinstance(m, nn.Conv1d):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                # n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


if __name__ == '__main__':

    net = GGNet()
    x = torch.rand(2, config.num_point, 3)
    net(x)
    # a = torch.tensor(np.array(range(0, 10))).unsqueeze(0)
    # print(a.shape)
    # x_pad = []
    # for i in range(10):
    #     p1d = (10 - i - 1, i)
    #     x_pad.append(F.pad(a, p1d, 'constant', 0))
    # b = torch.stack(x_pad, dim=1)
    # print(b.shape)
    # res = F.cosine_similarity(a.unsqueeze(2), b, dim=1)
    # print(res)
