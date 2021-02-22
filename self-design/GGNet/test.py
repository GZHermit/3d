import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# import numpy as np
#
# a = np.array([[1, 2, 3]])
# a = np.repeat(a, repeats=3, axis=0)
# a = np.repeat(np.expand_dims(a, axis=0), repeats=2, axis=0)
# a[0, 0, 0] = 8
# print(a.shape)
# b = np.zeros([2, 3], dtype=np.int32)
# c = a[b]
# print(b)
# print(c)
# print(c.shape)

class Correlation(nn.Module):
    def __init__(self, _final_channle, _encode_len):
        super(Correlation, self).__init__()

        _chennel = _final_channle

        self.corr_bypass = nn.Sequential(nn.Conv1d(_chennel, _chennel, kernel_size=3, padding=1),
                                         nn.BatchNorm1d(_chennel),
                                         nn.ReLU())
        corr_len = 2 * _encode_len - 1
        self.extra_1 = nn.Sequential(nn.Conv1d(corr_len, _chennel, kernel_size=3, padding=1),
                                     nn.BatchNorm1d(_chennel),
                                     nn.ReLU())
        self.extra_2 = nn.Sequential(nn.Conv1d(2 * _chennel, _chennel, kernel_size=3, padding=1),
                                     nn.BatchNorm1d(_chennel),
                                     nn.ReLU())
        # self._init_weights()

    def forward(self, x):

        b, c, p = x.size()
        res = x.new_tensor(torch.eye(p).repeat(b, 1, 1))
        print(res.shape)
        for i in range(p - 1):
            print(x[:, :, i].shape, x[:, :, i + 1:].shape)
            res[:, i, i + 1:] = F.cosine_similarity(x[:, :, i].unsqueeze(2), x[:, :, i + 1:])
            res[:, i+1:, i] = res[:, i, i + 1:]
        print(res)

    def forward_b(self, x):
        batch_size, channel_size, point_size = x.size()
        bypass = self.corr_bypass(x)
        x_pad = []
        for i in range(point_size):
            p1d = (point_size - i - 1, i)

            x_pad.append(F.pad(x, p1d, 'constant', 0))
        x_pad = torch.stack(x_pad, dim=2)
        x = F.cosine_similarity(x.unsqueeze(3), x_pad, dim=1)

        print(x)
        print(x.shape)
        x = x.permute(0, 2, 1).contiguous()
        print(x.shape)
        x = self.extra_1(x)
        print(x.shape)
        x = torch.cat([x, bypass], dim=1)
        print(x.shape)
        x = self.extra_2(x)

        return x


# x = torch.tensor([1, 2, 3], dtype=torch.float32).repeat(2, 4, 1)
# x = x + torch.tensor([[[0], [1], [2], [3]], [[0], [1], [2], [3]]], dtype=torch.float32)
# print(x.shape)
# net = Correlation(4, 3)
# y = net(x)


import torch
import torch.nn as nn


nn.Conv2d()