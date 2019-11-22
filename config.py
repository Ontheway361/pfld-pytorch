#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import os
import argparse


def training_args():
    parser = argparse.ArgumentParser(description='Trainging Practical Facial Landmark Detector')

    # env
    parser.add_argument('--use_gpu',      type=bool, default=False)
    parser.add_argument('--gpu_ids',      type=list, default=[0, 1])
    parser.add_argument('--workers',      type=int,  default=8)
    parser.add_argument('--devices_id',   default='0',     type=str)
    parser.add_argument('--test_initial', default='false', type=str2bool)

    # training
    ##  -- optimizer
    parser.add_argument('--base_lr',       type=float, default=0.0001)
    parser.add_argument('--weight-decay',  type=float, default=1e-6)

    # -- lr
    parser.add_argument("--lr_patience", default=40, type=int)

    # -- epoch
    parser.add_argument('--start_epoch', default=1,    type=int)
    parser.add_argument('--end_epoch',   default=1000, type=int)

    # -- snapshot、tensorboard log and checkpoint
    parser.add_argument(
        '--snapshot',
        default='./checkpoint/snapshot/',
        type=str,
        metavar='PATH')
    parser.add_argument(
        '--log_file', default="./checkpoint/train.logs", type=str)
    parser.add_argument(
        '--tensorboard', default="./checkpoint/tensorboard", type=str)
    parser.add_argument(
        '--resume', default='', type=str, metavar='PATH')  # TBD

    # --dataset
    parser.add_argument(
        '--dataroot',
        default='./data/train_data/list.txt',
        type=str,
        metavar='PATH')
    parser.add_argument(
        '--val_dataroot',
        default='./data/test_data/list.txt',
        type=str,
        metavar='PATH')
    parser.add_argument('--train_batchsize', default=256, type=int)
    parser.add_argument('--val_batchsize',   default=8, type=int)
    args = parser.parse_args()
    return args
