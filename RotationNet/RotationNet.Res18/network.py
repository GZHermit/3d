# coding:utf-8

import math
import os

import torch
import torch.nn as nn

from config import config
import torch.utils.model_zoo as model_zoo

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


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


class RotationNet(nn.Module):

    def __init__(self):
        super(MVCNN, self).__init__()
        self.inplanes = 64
        self.num_classes = (config.num_classes + 1) * config.nview

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

        self.initialize_weights()

    def forward(self, x):

        # Swap batch and views dims
        # View pool

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(v)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        x = x.view()
        return v

    @staticmethod
    def get_loss(cls_input, target):

        """
            input: B * NUM_CLASS,
            target: B,
        """
        classify_loss = nn.CrossEntropyLoss()(cls_input, target)
        return classify_loss

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

        pretrain = os.path.join('pretrained', 'resnet18.pth')
        if os.path.exists(pretrain):
            model_dict, pretrain = torch.load(pretrain), self.state_dict()
            pretrained_dict = {k: v for k, v in pretrain.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
        else:
            print('randomly initialize the weight of the model!')

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


if __name__ == '__main__':
    # net = MVCNN()
    # inp = torch.rand(2, 12, 3, 224, 224)
    # outp = net(inp)

    # print(net.state_dict())
    # cls_input = torch.rand(2, 40)
    # target = (torch.rand(2) * 40).to(torch.int64)
    # generated_pl = torch.rand(2, 128, 3)
    # original_pl = torch.rand(2, 100, 3)
    #
    # loss = FusionNet.get_loss(cls_input, target, generated_pl, original_pl)
    inp = torch.rand(48, 3, 224, 224)
    net = MVCNN()
    nview = 12
    nsamp = int(inp.size(0) / nview)

    output = net(inp)
    print(output.shape)
    num_classes = int(output.size(1) / nview) - 1
    print(num_classes)
    output = output.view(-1, num_classes + 1)
    print(output.shape)

    # compute scores and decide target labels
    output_ = torch.nn.functional.log_softmax(output)
    output_ = output_[:, :-1] - torch.t(output_[:, -1].repeat(1, output_.size(1) - 1).view(output_.size(1) - 1, -1))
    print(output.shape)
    output_ = output_.view(-1, nview * nview, num_classes)
    print(output_.shape)
    output_ = output_.data.cpu().numpy()
    output_ = output_.transpose(1, 2, 0)
    print(output_.shape)