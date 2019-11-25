#!/usr/bin/env python3
#-*- coding:utf-8 -*-
'''
Created on 2019/11/25
author: relu
'''

import os
import sys
import time
import torch
import numpy as np
import torchvision

from torch.utils.data import DataLoader
from dataset.datasets import WLFWDatasets

from config import training_args
from pfld import PFLDbackbone, AuxiliaryNet, PFLDLoss

from IPython import embed


class PFLDTrain(object):

    def __init__(self, args = args):

        self.args   = args
        self.model  = dict()
        self.data   = dict()
        self.device = args.use_gpu and torch.cuda.is_available()

    @staticmethod
    def _report_settings():
        ''' Report the settings '''

        str = '-' * 16
        print('%sEnvironment Versions%s' % (str, str))
        print("- Python: {}".format(sys.version.strip().split('|')[0]))
        print("- PyTorch: {}".format(torch.__version__))
        print("- TorchVison: {}".format(torchvision.__version__))
        print("- device: {}".format(self.device))
        print('-'*52)


    def _model_loader(self):

        self.model['backbone']  = PFLDbackbone().cuda(0) if self.device else PFLDbackbone()
        self.model['auxilnet']  = AuxiliaryNet().cuda(0) if self.device else AuxiliaryNet()
        self.model['criterion'] = PFLDLoss()
        self.model['optimizer'] = torch.optim.Adam(
                                      [{'params': self.model['backbone'].parameters()},
                                       {'params': self.model['auxilnet'].parameters()}],
                                      lr=args.base_lr,
                                      weight_decay=args.weight_decay)
        self.model['scheduler'] = torch.optim.lr_scheduler.ReduceLROnPlateau(
                                      self.model['optimizer'], mode='min', patience=args.lr_patience, \
                                      verbose=True)

        if self.device and len(self.args.gpu_ids) > 1:
            self.model['backbone'] = torch.nn.DataParallel(self.model['backbone'], device_ids=self.args.gpu_ids)
            self.model['auxilnet'] = torch.nn.DataParallel(self.model['auxilnet'], device_ids=self.args.gpu_ids)
            self.backends.cudnn.benchmark = True
            print('Parallel mode was going ...')
        elif self.device:
            print('Single-gpu mode was going ...')
        else:
            print('CPU mode was going ...')

        if self.args.resume is not None or len(self.args.resume) < 2:
            checkpoint = torch.load(self.args.result, map_location=lambda storage, loc: storage)
            self.args.start_epoch = checkpoint['epoch']
            self.model['backbone'].load_state_dict(checkpoint['backbone'])
            self.model['auxilnet'].load_state_dict(checkpoint['auxilnet'])
            print('Resuming the train process at %3d epoches ...' % self.args.start_epoch)
        print('Model loading was finished ...')


    def _data_loader(self):

        transform = torchvision.transforms.Compose([transforms.ToTensor()])
        self.data['train_loader'] = DataLoader(WLFWDatasets(self.args.dataroot, transform),
                                        batch_size=self.args.train_batchsize,
                                        shuffle=True,
                                        num_workers=self.args.workers,
                                        drop_last=False)
        self.data['eval_loader']  = DataLoader(WLFWDatasets(args.val_dataroot, transform),
                                        batch_size=self.args.val_batchsize,
                                        shuffle=False,
                                        num_workers=self.args.workers)
        print('Data loading was finished ...')


    def _model_train(self):

        self.model['backbone'].train()
        self.model['auxilnet'].train()

        for img, landmark_gt, attribute_gt, euler_angle_gt in self.data['train_loader']:

            img.requires_grad            = False
            attribute_gt.requires_grad   = False
            landmark_gt.requires_grad    = False
            euler_angle_gt.requires_grad = False

            if self.device:
                img            = img.cuda(non_blocking=True)
                attribute_gt   = attribute_gt.cuda(non_blocking=True)
                landmark_gt    = landmark_gt.cuda(non_blocking=True)
                euler_angle_gt = euler_angle_gt.cuda(non_blocking=True)

            features, landmarks = self.model['backbone'](img)
            angle = self.model['auxilnet'](features)
            weighted_loss, loss = self.model['criterion'](attribute_gt, landmark_gt, euler_angle_gt,
                                        angle, landmarks, self.args.train_batchsize)

            self.model['optimizer'].zero_grad()
            weighted_loss.backward()
            self.model['optimizer'].step()

        return weighted_loss, loss


    def _model_eval(self):


        self.model['backbone'].eval()
        self.model['auxilnet'].eval()

        losses = []
        with torch.no_grad():
            for img, landmark_gt, attribute_gt, euler_angle_gt in self.data['eval_loader']:

                img.requires_grad            = False
                attribute_gt.requires_grad   = False
                landmark_gt.requires_grad    = False
                euler_angle_gt.requires_grad = False

                if self.device:
                    img            = img.cuda(non_blocking=True)
                    attribute_gt   = attribute_gt.cuda(non_blocking=True)
                    landmark_gt    = landmark_gt.cuda(non_blocking=True)
                    euler_angle_gt = euler_angle_gt.cuda(non_blocking=True)


                _, landmark = self.model['backbone'](img)

                loss = torch.mean(torch.sum((landmark_gt - landmark)**2,axis=1))
                losses.append(loss.cpu().numpy())

        return np.mean(losses)


    def _main_loop(self):

        for epoch in range(self.args.start_epoch, self.args.end_epoch + 1):

            weighted_train_loss, train_loss = self._model_train()
            filename = os.path.join(str(self.args.snapshot), "checkpoint_epoch_" + str(epoch) + '.pth.tar')
            torch.save({
                'epoch'   : epoch,
                'backbone': self.model['backbone'].state_dict(),
                'auxilnet': self.model['auxilnet'].state_dict()
            }, filename)

            val_loss = self._model_eval()

            self.model['scheduler'].step()


    def train_runner(self):

        self._report_settings()

        self._model_loader()

        self._data_loader()

        self._main_loop()


if __name__ == "__main__":

    pfld_trainer = PFLDTrain(training_args())
    pfld_trainer.train_runner()
