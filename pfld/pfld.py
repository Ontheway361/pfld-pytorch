#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn

from IPython import embed

def conv_bn(inp, oup, kernel, stride, padding = 1):
    return nn.Sequential(
               nn.Conv2d(inp, oup, kernel, stride, padding, bias=False),
               nn.BatchNorm2d(oup),
               nn.ReLU(inplace=True))


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
               nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
               nn.BatchNorm2d(oup),
               nn.ReLU(inplace=True))


class InvertedResidual(nn.Module):

    def __init__(self, inp, oup, stride, use_res_connect, expand_ratio=6):

        super(InvertedResidual, self).__init__()

        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = use_res_connect
        hid_channels = inp * expand_ratio
        self.conv = nn.Sequential(
            nn.Conv2d(inp, hid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid_channels, hid_channels, 3, stride, 1, groups=hid_channels, bias=False),
            nn.BatchNorm2d(hid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid_channels, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )


    def forward(self, x):

        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class PFLDbackbone(nn.Module):

    def __init__(self):

        super(PFLDbackbone, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)

        self.conv3_1  = InvertedResidual(64, 64, 2, False, 2)
        self.block3_2 = InvertedResidual(64, 64, 1, True, 2)
        self.block3_3 = InvertedResidual(64, 64, 1, True, 2)
        self.block3_4 = InvertedResidual(64, 64, 1, True, 2)
        self.block3_5 = InvertedResidual(64, 64, 1, True, 2)

        self.conv4_1  = InvertedResidual(64, 128, 2, False, 2)

        self.conv5_1  = InvertedResidual(128, 128, 1, False, 4)
        self.block5_2 = InvertedResidual(128, 128, 1, True, 4)
        self.block5_3 = InvertedResidual(128, 128, 1, True, 4)
        self.block5_4 = InvertedResidual(128, 128, 1, True, 4)
        self.block5_5 = InvertedResidual(128, 128, 1, True, 4)
        self.block5_6 = InvertedResidual(128, 128, 1, True, 4)

        self.conv6_1  = InvertedResidual(128, 16, 1, False, 2)  # [16, 14, 14]

        self.conv7 = conv_bn(16, 32, 3, 2)  # [32, 7, 7]
        self.conv8 = nn.Conv2d(32, 128, 7, 1, 0)  # [128, 1, 1]
        self.bn8   = nn.BatchNorm2d(128)

        self.avg_pool1 = nn.AvgPool2d(14)
        self.avg_pool2 = nn.AvgPool2d(7)
        self.fc        = nn.Linear(176, 196)

    def forward(self, x):  # x: 3, 112, 112

        x = self.relu(self.bn1(self.conv1(x)))  # [64, 56, 56]
        x = self.relu(self.bn2(self.conv2(x)))  # [64, 56, 56]
        x = self.conv3_1(x)
        x = self.block3_2(x)
        x = self.block3_3(x)
        x = self.block3_4(x)
        out1 = self.block3_5(x)

        x = self.conv4_1(out1)
        x = self.conv5_1(x)
        x = self.block5_2(x)
        x = self.block5_3(x)
        x = self.block5_4(x)
        x = self.block5_5(x)
        x = self.block5_6(x)
        x = self.conv6_1(x)
        x1 = self.avg_pool1(x)
        x1 = x1.view(x1.size(0), -1)

        x = self.conv7(x)
        x2 = self.avg_pool2(x)
        x2 = x2.view(x2.size(0), -1)

        x3 = self.relu(self.conv8(x))
        x3 = x3.view(x1.size(0), -1)

        multi_scale = torch.cat([x1, x2, x3], 1)
        landmarks = self.fc(multi_scale)

        return out1, landmarks


class AuxiliaryNet(nn.Module):

    def __init__(self):

        super(AuxiliaryNet, self).__init__()

        self.conv1 = conv_bn(64, 128, 3, 2)
        self.conv2 = conv_bn(128, 128, 3, 1)
        self.conv3 = conv_bn(128, 32, 3, 2)
        self.conv4 = conv_bn(32, 128, 7, 1)
        self.max_pool1 = nn.MaxPool2d(3)
        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, 3)

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.max_pool1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

# attribute [pose, expression, illumination, make-up, occlusion, blur]
class PFLDLoss(nn.Module):

    def __init__(self):
        super(PFLDLoss, self).__init__()

    def forward(self, attribute_gt, landmark_gt, euler_angle_gt, angle, \
                landmarks, train_batchsize):

        weight_angle = torch.sum(1 - torch.cos(angle - euler_angle_gt), axis=1)
        attributes_w_n = attribute_gt[:, 1:6].float()
        mat_ratio = torch.mean(attributes_w_n, axis=0)
        # mat_ratio = torch.Tensor([1.0 / (x) if x > 0 else train_batchsize for x in mat_ratio]).cuda()   # default
        mat_ratio = torch.Tensor([1.0 / (x) if x > 0 else train_batchsize for x in mat_ratio])
        weight_attribute = torch.sum(attributes_w_n.mul(mat_ratio), axis=1)
        l2_distant  = torch.sum((landmark_gt - landmarks) * (landmark_gt - landmarks), axis=1)
        weight_loss = torch.mean(weight_angle * weight_attribute * l2_distant)
        ave_l2_loss = torch.mean(l2_distant)
        return weight_loss, ave_l2_loss


# if __name__ == '__main__':
#     input = torch.randn(1, 3, 112, 112)
#     plfd_backbone = PFLDbackbone()
#     auxiliarynet = AuxiliaryNet()
#     features, landmarks = plfd_backbone(input)
#     angle = auxiliarynet(features)

#     print("angle.shape:{0:}, landmarks.shape: {1:}".format(
#         angle.shape, landmarks.shape))
