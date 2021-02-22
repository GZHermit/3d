# coding:utf-8

from config import config
from network import conv_bn, fc_bn

import torch
import torch.nn as nn
import torch.nn.functional as F

class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()

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
        self.fc7 = nn.Linear(4096, 512)

    def forward(self, x):
        """
            :param x: B * V * W * H * C, the different views with the same point cloud model
            :return: the global feature, the size is B * F
        """
        B, V, W, H, C = x.size()
        if C != 3:
            x = x.permute(0, 1, 4, 2, 3).contiguous()
            _, _, W, H, C = x.size()
        x = x.view((B * self.nview, C, W, H))

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
        return x


if __name__ == '__main__':
    # path = 'Vgg16.pth'
    # net = Vgg16()
    # a = torch.load(path)
    # b = net.state_dict()
    #
    # for bkey in b.keys():
    #     for akey in a.keys():
    #         if a[akey].shape == b[bkey].shape and 'weight' in akey:
    #             print(akey, bkey)
    #             a[bkey] = a.pop(akey)
    #             akey = akey.rstrip('weight')+'bias'
    #             bkey = bkey.rstrip('weight')+'bias'
    #             print(akey, bkey)
    #             a[bkey] = a.pop(akey)
    #             break
    # torch.save(a, 'vgg16.pth')
    # net.load_state_dict(torch.load(path))

    path = 'pretrained/vgg16.pth'
    a = torch.load(path)
    print(a.keys())