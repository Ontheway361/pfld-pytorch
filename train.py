#!/usr/bin/env python3
#-*- coding:utf-8 -*-

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

    def __init__(self, args):

        self.args   = args
        self.model  = dict()
        self.data   = dict()
        self.device = args.use_gpu and torch.cuda.is_available()


    def _report_settings(self):
        ''' Report the settings '''

        str = '-' * 16
        print('%sEnvironment Versions%s' % (str, str))
        print("- Python    : {}".format(sys.version.strip().split('|')[0]))
        print("- PyTorch   : {}".format(torch.__version__))
        print("- TorchVison: {}".format(torchvision.__version__))
        print("- use_gpu   : {}".format(self.device))
        print('-'*52)


    def _model_loader(self):

        self.model['backbone']  = PFLDbackbone().cuda(0) if self.device else PFLDbackbone()
        self.model['auxilnet']  = AuxiliaryNet().cuda(0) if self.device else AuxiliaryNet()
        self.model['criterion'] = PFLDLoss(use_gpu=self.device)
        self.model['optimizer'] = torch.optim.Adam(
                                      [{'params': self.model['backbone'].parameters()},
                                       {'params': self.model['auxilnet'].parameters()}],
                                      lr=self.args.base_lr,
                                      weight_decay=self.args.weight_decay)
        self.model['scheduler'] = torch.optim.lr_scheduler.ReduceLROnPlateau(
                                      self.model['optimizer'], mode='min', patience=self.args.lr_patience, \
                                      verbose=True)

        if self.device and len(self.args.gpu_ids) > 1:
            self.model['backbone'] = torch.nn.DataParallel(self.model['backbone'], device_ids=self.args.gpu_ids)
            self.model['auxilnet'] = torch.nn.DataParallel(self.model['auxilnet'], device_ids=self.args.gpu_ids)
            torch.backends.cudnn.benchmark = True
            print('Parallel mode was going ...')
        elif self.device:
            print('Single-gpu mode was going ...')
        else:
            print('CPU mode was going ...')

        if len(self.args.resume) > 2:
            checkpoint = torch.load(self.args.resume, map_location=lambda storage, loc: storage)
            self.args.start_epoch = checkpoint['epoch']
            self.model['backbone'].load_state_dict(checkpoint['backbone'])
            self.model['auxilnet'].load_state_dict(checkpoint['auxilnet'])
            print('Resuming the train process at %3d epoches ...' % self.args.start_epoch)
        print('Model loading was finished ...')


    def _data_loader(self):

        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        self.data['train_loader'] = DataLoader(WLFWDatasets(self.args.train_file, transform),
                                        batch_size=self.args.train_batchsize,
                                        shuffle=True,
                                        num_workers=self.args.workers,
                                        drop_last=False)
        self.data['eval_loader']  = DataLoader(WLFWDatasets(self.args.eval_file, transform),
                                        batch_size=self.args.val_batchsize,
                                        shuffle=False,
                                        num_workers=self.args.workers)
        print('Data loading was finished ...')


    def _model_train(self, epoch = 0):

        self.model['backbone'].train()
        self.model['auxilnet'].train()

        loss_recorder = []
        for idx, (img, landmark_gt, attribute_gt, euler_angle_gt) in enumerate(self.data['train_loader']):

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
            euler_angle         = self.model['auxilnet'](features)
            weighted_loss, loss = self.model['criterion'](attribute_gt, landmark_gt, euler_angle_gt,
                                        euler_angle, landmarks, self.args.train_batchsize)

            self.model['optimizer'].zero_grad()
            weighted_loss.backward()
            self.model['optimizer'].step()
            loss_recorder.append([weighted_loss.item(), loss.item()])

            if (idx + 1) % self.args.print_freq == 0:
                ave_loss = np.mean(np.array(loss_recorder), axis=0)
                print('cur_epoch : %3d|%3d, weighted_loss : %.4f, loss : %.4f' % \
                      (epoch, self.args.end_epoch, ave_loss[0], ave_loss[1]))

        return np.mean(np.array(loss_recorder), axis=0)


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

                loss = torch.mean(torch.sum((landmark_gt - landmark)**2, dim=1))
                losses.append(loss.cpu().numpy())
        ave_loss = np.mean(losses)
        print('eval_loss was : %.4f' % ave_loss)
        return ave_loss


    def _main_loop(self):
        
        min_loss = 1e6
        for epoch in range(self.args.start_epoch, self.args.end_epoch + 1):

            weighted_train_loss, train_loss = self._model_train()
            
            if (epoch + 1) % self.args.save_freq == 0:
                filename = os.path.join(str(self.args.snapshot), "checkpoint_epoch_" + str(epoch) + '.pth.tar')
                torch.save({
                    'epoch'   : epoch,
                    'backbone': self.model['backbone'].state_dict(),
                    'auxilnet': self.model['auxilnet'].state_dict()
                }, filename)
                
            val_loss = self._model_eval()
            
            if val_loss < min_loss:
                min_loss = val_loss
                filename = os.path.join(str(self.args.snapshot), 'sota.pth.tar')
                torch.save({
                    'epoch'   : epoch,
                    'backbone': self.model['backbone'].state_dict(),
                    'auxilnet': self.model['auxilnet'].state_dict()
                }, filename)
                print('sota performance was updated ...')
            self.model['scheduler'].step()


    def train_runner(self):

        self._report_settings()

        self._model_loader()

        self._data_loader()

        self._main_loop()


if __name__ == "__main__":

    pfld_trainer = PFLDTrain(training_args())
    pfld_trainer.train_runner()
