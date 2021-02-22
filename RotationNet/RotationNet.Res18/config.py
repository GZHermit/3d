# coding:utf-8

import os
import os.path as osp


class Config:

    # max_acc: 0.8853322528363047, max_correct: 2185

    # -------------------- data config --------------------#

    root_dir = ''

    point_dir = os.path.join(root_dir, 'ModelNet40')
    view_dir = os.path.join(root_dir, 'view')

    save_path = '/home/guozihao/Workspace/Save/Experiment/FusionNet.VGG16'

    train_totality = 9840
    eval_totality = 2468
    num_classes = 40
    num_point = 2048

    # -------------------- model config --------------------#

    cuda = True
    nview = 12
    learning_rate = 1e-3
    momentum = 0.9
    batch_size = 2
    decay_rate = 0.7
    decay_step = 200000
    weight_decay = 1e-4
    lr_decay = 1e-5
    max_epoch = 200

    code_length = 512


config = Config()

if __name__ == '__main__':
    pass
