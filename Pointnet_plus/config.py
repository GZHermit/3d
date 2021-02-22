# coding:utf-8

import os
import os.path as osp


class Config:
    # -------------------- data config --------------------#

    root_dir = '/unsullied/sharefs/_research_detection/GeneralDetection/ModelNet40/ModelNet40'
    save_path = '/home/guozihao/Workspace/Save/Experiment/PointNet_plus'
    train_totality = 9840
    eval_totality = 2468
    num_classes = 40
    num_point = 2048

    # -------------------- model config --------------------#
    K = 64
    cuda = True
    learning_rate = 1e-3
    batch_size = 16
    decay_rate = 0.7
    decay_step = 200000
    weight_decay = 1e-4
    lr_decay = 1e-5
    max_epoch = 100
    end_point = {}


config = Config()

if __name__ == '__main__':
    import pcap
