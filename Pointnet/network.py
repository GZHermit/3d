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


class Input_transform_net(nn.Module):
    """ Input (XYZ) Transform Net, input is BxNx3 gray image
        Return:
            Transformation matrix of size 3xK """

    def __init__(self, K=3):
        super(Input_transform_net, self).__init__()
        self.num_point = config.num_point
        self.transform_xyz_weights = nn.Parameter(torch.zeros(256, 3 * K))
        self.transform_xyz_bias = nn.Parameter(torch.zeros(3 * K) + torch.FloatTensor(np.eye(3).flatten()))

        self.num_features = 0
        self.net = nn.Sequential(
            Reshape(),
            conv_bn(1, 64, [1, 3]),
            conv_bn(64, 128, [1, 1]),
            conv_bn(128, config.num_point, [1, 1]),
            nn.MaxPool2d([self.num_point, 1]),
            Reshape([-1, ]),
            fc_bn(self.num_point, 512),
            fc_bn(512, 256),
            Matmul(self.transform_xyz_weights, self.transform_xyz_bias),
            Reshape([3, K])
        )

    def forward(self, x):
        return self.net(x)


class Feature_transform_net(nn.Module):
    """ Feature Transform Net, input is Bx64xN*1
        Return:
            Transformation matrix of size KxK """

    def __init__(self, K=64):
        super(Feature_transform_net, self).__init__()
        self.num_point = config.num_point

        self.transform_feat_weights = nn.Parameter(torch.zeros(256, K * K))
        self.transform_feat_bias = nn.Parameter(torch.zeros(K * K) + torch.FloatTensor(np.eye(K).flatten()))

        self.num_features = 0
        self.net = nn.Sequential(
            conv_bn(64, 64, [1, 1]),
            conv_bn(64, 128, [1, 1]),
            conv_bn(128, self.num_point, [1, 1]),
            nn.MaxPool2d([self.num_point, 1]),
            Reshape([-1, ]),
            fc_bn(self.num_point, 512),
            fc_bn(512, 256),
            Matmul(self.transform_feat_weights, self.transform_feat_bias),
            Reshape([K, K])
        )

    def forward(self, x):
        return self.net(x)


class PointNet(nn.Module):

    def __init__(self):
        '''
        input: B x N x 3
        output: B x 40
        '''
        super(PointNet, self).__init__()

        self.num_point = config.num_point
        self.num_classes = config.num_classes
        self.K = config.K

        self.input_transform_net = Input_transform_net()
        self.feat_transform_net = Feature_transform_net(self.K)
        self.conv1 = conv_bn(1, 64, [1, 3])
        self.conv2 = conv_bn(64, self.K, [1, 1])
        self.conv3 = conv_bn(self.K, 64, [1, 1])
        self.conv4 = conv_bn(64, 128, [1, 1])
        self.conv5 = conv_bn(128, self.num_point, [1, 1])
        self.fc1 = fc_bn(self.num_point, 512)
        self.fc2 = fc_bn(512, 256)
        self.fc3 = nn.Linear(256, self.num_classes)

        # self.I = nn.Parameter(torch.tensor(np.eye(config.K), dtype=torch.float, requires_grad=False), requires_grad=False)

        self.initialize_weights()

    def forward(self, x):
        print(x.shape)
        B = x.size()[0]

        input_transform = self.input_transform_net(x)
        x = torch.matmul(x, input_transform)
        print(x.shape)
        x = x.view((B, 1, self.num_point, 3))

        x = self.conv1(x)
        x = self.conv2(x)

        feat_transform = self.feat_transform_net(x)
        config.end_point['transform'] = feat_transform

        x = x.view((B, self.K, self.num_point))
        x = torch.matmul(feat_transform, x)
        x = x.view((B, self.K, self.num_point, 1))

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

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

        # Enforce the transformation as orthogonal matrix

        transform = config.end_point['transform']  # BxKxK
        mat_diff = torch.matmul(transform, transform.transpose(2, 1).contiguous())
        I = mat_diff.new_tensor(torch.eye(config.K))
        mat_diff.sub_(I)

        mat_diff_loss = torch.mean(torch.sum(mat_diff ** 2)) / 2.
        reg_weight = 0.001

        # decay_loss = torch.sum(mat_diff ** 2) * config.weight_decay * 0.5

        return loss.to(mat_diff_loss.device) + reg_weight * mat_diff_loss

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
    data = torch.rand(2, config.num_point ,3)
    net = PointNet()
    out = net(data)
    print(net)
