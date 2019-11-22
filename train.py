#!/usr/bin/env python3
#-*- coding:utf-8 -*-
'''
Created on 2019/11/21
author: relu
'''

import os
import time
import torch
import logging
import argparse
import numpy as np

import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tensorboardX import SummaryWriter

from dataset.datasets import WLFWDatasets
from models.pfld import PFLDInference, AuxiliaryNet
from pfld.loss import PFLDLoss
from pfld.utils import AverageMeter


def print_args(args):
    for arg in vars(args):
        s = arg + ': ' + str(getattr(args, arg))
        logging.info(s)


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    logging.info('Save checkpoint to {0:}'.format(filename))


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')


def train(train_loader, plfd_backbone, auxiliarynet, criterion, optimizer, epoch):

    losses = AverageMeter()

    for img, landmark_gt, attribute_gt, euler_angle_gt in train_loader:
        img.requires_grad = False
        img = img.cuda(non_blocking=True)

        attribute_gt.requires_grad = False
        attribute_gt = attribute_gt.cuda(non_blocking=True)

        landmark_gt.requires_grad = False
        landmark_gt = landmark_gt.cuda(non_blocking=True)

        euler_angle_gt.requires_grad = False
        euler_angle_gt = euler_angle_gt.cuda(non_blocking=True)

        plfd_backbone = plfd_backbone.cuda()
        auxiliarynet = auxiliarynet.cuda()

        features, landmarks = plfd_backbone(img)
        angle = auxiliarynet(features)
        weighted_loss, loss = criterion(attribute_gt, landmark_gt, euler_angle_gt,
                                    angle, landmarks, args.train_batchsize)

        optimizer.zero_grad()
        weighted_loss.backward()
        optimizer.step()

        losses.update(loss.item())
    return weighted_loss, loss


def validate(wlfw_val_dataloader, plfd_backbone, auxiliarynet, criterion, epoch):

    plfd_backbone.eval()
    auxiliarynet.eval()

    losses = []
    with torch.no_grad():
        for img, landmark_gt, attribute_gt, euler_angle_gt in wlfw_val_dataloader:
            img.requires_grad = False
            img = img.cuda(non_blocking=True)

            attribute_gt.requires_grad = False
            attribute_gt = attribute_gt.cuda(non_blocking=True)

            landmark_gt.requires_grad = False
            landmark_gt = landmark_gt.cuda(non_blocking=True)

            euler_angle_gt.requires_grad = False
            euler_angle_gt = euler_angle_gt.cuda(non_blocking=True)

            plfd_backbone = plfd_backbone.cuda()
            auxiliarynet = auxiliarynet.cuda()

            _, landmark = plfd_backbone(img)

            loss = torch.mean(
                torch.sum((landmark_gt - landmark)**2,axis=1))
            losses.append(loss.cpu().numpy())

    return np.mean(losses)


def main(args):
    # Step 1: parse args config
    logging.basicConfig(
        format=
        '[%(asctime)s] [p%(process)s] [%(pathname)s:%(lineno)d] [%(levelname)s] %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(args.log_file, mode='w'),
            logging.StreamHandler()
        ])
    print_args(args)

    # Step 2: model, criterion, optimizer, scheduler
    plfd_backbone = PFLDInference().cuda()
    auxiliarynet  = AuxiliaryNet().cuda()
    criterion     = PFLDLoss()
    optimizer = torch.optim.Adam(
        [{
            'params': plfd_backbone.parameters()
        }, {
            'params': auxiliarynet.parameters()
        }],
        lr=args.base_lr,
        weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=args.lr_patience, verbose=True)

    # step 3: data
    # argumetion
    transform = transforms.Compose([transforms.ToTensor()])
    wlfwdataset = WLFWDatasets(args.dataroot, transform)
    dataloader = DataLoader(
        wlfwdataset,
        batch_size=args.train_batchsize,
        shuffle=True,
        num_workers=args.workers,
        drop_last=False)

    wlfw_val_dataset = WLFWDatasets(args.val_dataroot, transform)
    wlfw_val_dataloader = DataLoader(
        wlfw_val_dataset,
        batch_size=args.val_batchsize,
        shuffle=False,
        num_workers=args.workers)

    # step 4: run
    writer = SummaryWriter(args.tensorboard)
    for epoch in range(args.start_epoch, args.end_epoch + 1):
        weighted_train_loss, train_loss = train(dataloader, plfd_backbone, auxiliarynet,
                                      criterion, optimizer, epoch)
        filename = os.path.join(
            str(args.snapshot), "checkpoint_epoch_" + str(epoch) + '.pth.tar')
        save_checkpoint({
            'epoch': epoch,
            'plfd_backbone': plfd_backbone.state_dict(),
            'auxiliarynet': auxiliarynet.state_dict()
        }, filename)

        val_loss = validate(wlfw_val_dataloader, plfd_backbone, auxiliarynet,
                            criterion, epoch)

        scheduler.step(val_loss)
        writer.add_scalar('data/weighted_loss', weighted_train_loss, epoch)
        writer.add_scalars('data/loss', {'val loss': val_loss, 'train loss': train_loss}, epoch)
    writer.close()



if __name__ == "__main__":
    args = parse_args()
    main(args)
