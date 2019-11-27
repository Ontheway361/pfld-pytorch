#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from IPython import embed

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.constant_(m.bias, 0.1)


class LossFn:

    def __init__(self, cls_factor = 1, box_factor = 1, landmark_factor = 1):

        self.cls_factor  = cls_factor
        self.box_factor  = box_factor
        self.land_factor = landmark_factor
        self.loss_cls    = nn.BCELoss() # binary cross entropy
        self.loss_box    = nn.MSELoss() # mean square error
        self.loss_landmark = nn.MSELoss()


    def cls_loss(self, gt_label, pred_label):

        pred_label = torch.squeeze(pred_label)
        gt_label   = torch.squeeze(gt_label)
        mask = torch.ge(gt_label, 0)  # filter the part-face
        valid_gt_label = torch.masked_select(gt_label,mask)
        valid_pred_label = torch.masked_select(pred_label,mask)

        return self.loss_cls(valid_pred_label,valid_gt_label) * self.cls_factor


    def box_loss(self, gt_label, gt_offset, pred_offset):

        pred_offset = torch.squeeze(pred_offset)
        gt_offset   = torch.squeeze(gt_offset)
        gt_label    = torch.squeeze(gt_label)

        #get the mask element which != 0
        unmask = torch.eq(gt_label,0)
        mask = torch.eq(unmask,0)
        #convert mask to dim index
        chose_index = torch.nonzero(mask.data)
        chose_index = torch.squeeze(chose_index)
        #only valid element can effect the loss
        valid_gt_offset = gt_offset[chose_index,:]
        valid_pred_offset = pred_offset[chose_index,:]
        return self.loss_box(valid_pred_offset,valid_gt_offset) * self.box_factor


    def landmark_loss(self, gt_label, gt_landmark, pred_landmark):

        pred_landmark = torch.squeeze(pred_landmark)
        gt_landmark   = torch.squeeze(gt_landmark)
        gt_label      = torch.squeeze(gt_label)
        mask          = torch.eq(gt_label, -2)   # this is why set the label of lmk_pts to -2

        chose_index = torch.nonzero(mask.data)
        chose_index = torch.squeeze(chose_index)

        valid_gt_landmark   = gt_landmark[chose_index, :]
        valid_pred_landmark = pred_landmark[chose_index, :]

        if sum(mask) == 0:
            return 0
        else:
            return self.loss_landmark(valid_pred_landmark, valid_gt_landmark) * self.land_factor


class PNet(nn.Module):
    ''' PNet '''

    def __init__(self, is_train = False, use_cuda = True):

        super(PNet, self).__init__()

        self.is_train = is_train
        self.use_cuda = use_cuda

        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=3, stride=1),  # conv1
            nn.PReLU(),                                 # PReLU1
            nn.MaxPool2d(kernel_size=2, stride=2),      # pool1
            nn.Conv2d(10, 16, kernel_size=3, stride=1), # conv2
            nn.PReLU(),                                 # PReLU2
            nn.Conv2d(16, 32, kernel_size=3, stride=1), # conv3
            nn.PReLU()                                  # PReLU3
        )

        self.conv4_1 = nn.Conv2d(32, 1, kernel_size=1, stride=1)     # detection
        self.conv4_2 = nn.Conv2d(32, 4, kernel_size=1, stride=1)     # bbox regresion
        self.conv4_3 = nn.Conv2d(32, 10, kernel_size=1, stride=1)   # 68-points | 5-points

        self.apply(weights_init)


    def forward(self, x):

        x = self.pre_layer(x)
        label    = torch.sigmoid(self.conv4_1(x))
        offset   = self.conv4_2(x)
        landmark = self.conv4_3(x)

        return label, offset, landmark


class RNet(nn.Module):
    ''' RNet '''

    def __init__(self,is_train=False, use_cuda=True):

        super(RNet, self).__init__()

        self.is_train = is_train
        self.use_cuda = use_cuda

        # backbone
        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 28, kernel_size=3, stride=1),  # conv1
            nn.PReLU(),                                 # prelu1
            nn.MaxPool2d(kernel_size=3, stride=2),      # pool1
            nn.Conv2d(28, 48, kernel_size=3, stride=1), # conv2
            nn.PReLU(),                                 # prelu2
            nn.MaxPool2d(kernel_size=3, stride=2),      # pool2
            nn.Conv2d(48, 64, kernel_size=2, stride=1), # conv3
            nn.PReLU()                                  # prelu3

        )

        self.conv4   = nn.Linear(64*2*2, 128)  # conv4
        self.prelu4  = nn.PReLU()              # prelu4
        self.conv5_1 = nn.Linear(128, 1)       # detection
        self.conv5_2 = nn.Linear(128, 4)       # bounding box regression
        self.conv5_3 = nn.Linear(128, 10)     # lanbmark localization

        self.apply(weights_init)


    def forward(self, x):

        x = self.pre_layer(x)
        x = x.view(x.size(0), -1)
        x = self.conv4(x)
        x = self.prelu4(x)

        det = torch.sigmoid(self.conv5_1(x))
        box = self.conv5_2(x)
        lmk = self.conv5_3(x)

        return det, box, lmk


class ONet(nn.Module):
    ''' RNet '''

    def __init__(self, is_train = False, use_cuda = True):

        super(ONet, self).__init__()

        self.is_train = is_train
        self.use_cuda = use_cuda

        # backbone
        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1),  # conv1
            nn.PReLU(),                                 # prelu1
            nn.MaxPool2d(kernel_size=3, stride=2),      # pool1
            nn.Conv2d(32, 64, kernel_size=3, stride=1), # conv2
            nn.PReLU(),                                 # prelu2
            nn.MaxPool2d(kernel_size=3, stride=2),      # pool2
            nn.Conv2d(64, 64, kernel_size=3, stride=1), # conv3
            nn.PReLU(),                                 # prelu3
            nn.MaxPool2d(kernel_size=2,stride=2),       # pool3
            nn.Conv2d(64,128,kernel_size=2,stride=1),   # conv4
            nn.PReLU()                                  # prelu4
        )

        self.conv5  = nn.Linear(128*2*2, 256)           # conv5
        self.prelu5 = nn.PReLU()                        # prelu5

        self.conv6_1 = nn.Linear(256, 1)
        self.conv6_2 = nn.Linear(256, 4)
        self.conv6_3 = nn.Linear(256, 10)              # 68 <--> 136; 5 <--> 10

        self.apply(weights_init)

    def forward(self, x):

        # backend
        x = self.pre_layer(x)
        x = x.view(x.size(0), -1)
        x = self.conv5(x)
        x = self.prelu5(x)

        # detection
        det  = torch.sigmoid(self.conv6_1(x))
        box  = self.conv6_2(x)
        lmk  = self.conv6_3(x)

        return det, box, lmk
