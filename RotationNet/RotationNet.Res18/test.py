# coding:utf-8


import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import os


class Args:
    case = 1


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
    #
    # path = 'pretrained/vgg16.pth'
    # a = torch.load(path)
    # print(a.keys())
    from config import config
    args = Args()
    print(args.case)
    vcand_fn = ''
    if args.case == '1':
        vcand_fn = 'vcand_case1.npy'
        config.nview = 12
    elif args.case == '2':
        vcand_fn = 'vcand_case2.npy'
        config.nview = 80
    elif args.case == '3':
        vcand_fn = 'vcand_case3.npy'
        config.nview = 160

    vcand = np.load(os.path.join('pretrained', vcand_fn))
    print(vcand)
    print(vcand.shape)
