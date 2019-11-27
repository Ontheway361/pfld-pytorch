#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import argparse
import os.path as osp


data_dir = '/Users/relu/data/benchmark_images/faceu/face_landmark/WFLW'   # mini-mac


def training_args():
    parser = argparse.ArgumentParser(description='Trainging Practical Facial Landmark Detector')

    # env
    parser.add_argument('--use_gpu',  type=bool, default=False)
    parser.add_argument('--gpu_ids',  type=list, default=[0, 1])
    parser.add_argument('--workers',  type=int,  default=8)

    # --dataset
    parser.add_argument('--train_file',      type=str, default=osp.join(data_dir, 'wflw_train/wflw_train.txt'))
    parser.add_argument('--eval_file',       type=str, default=osp.join(data_dir, 'wflw_eval/wflw_eval.txt'))
    parser.add_argument('--train_batchsize', type=int, default=256)   # default=256
    parser.add_argument('--val_batchsize',   type=int, default=8)

    ##  -- optimizer
    parser.add_argument('--base_lr',       type=float, default=0.0001)
    parser.add_argument('--weight_decay',  type=float, default=1e-6)

    # -- lr
    parser.add_argument("--lr_patience", type=int, default=40)

    # -- epoch
    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('--end_epoch',   type=int, default=1000)
    parser.add_argument('--print_freq',  type=int, default=100)

    # -- snapshot
    parser.add_argument('--resume',  type=str, default='')
    parser.add_argument('--snapshot',type=str, default='checkpoint/snapshot/')


    args = parser.parse_args()
    return args
