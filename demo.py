#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import os
import cv2
import sys
import torch
import argparse
import numpy as np
import torchvision

from mtcnn import MtcnnDetector
from pfld  import PFLDbackbone, AuxiliaryNet

from IPython import embed


class PFLDDemo(object):

    def __init__(self, args):

        self.args   = args
        self.model  = dict()
        self.data   = dict()
        self.device = args.use_gpu and torch.cuda.is_available()


    def _report_settings(self):
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
        self.model['mtcnn']     = MtcnnDetector(use_gpu=self.device)
        
        if self.device and len(self.args.gpu_ids) > 1:
            self.model['backbone'] = torch.nn.DataParallel(self.model['backbone'], device_ids=self.args.gpu_ids)
            self.model['auxilnet'] = torch.nn.DataParallel(self.model['auxilnet'], device_ids=self.args.gpu_ids)
            self.backends.cudnn.benchmark = True
            print('Parallel mode was going ...')
        elif self.device:
            print('Single-gpu mode was going ...')
        else:
            print('CPU mode was going ...')

        if len(self.args.resume) > 2:
            checkpoint = torch.load(self.args.resume, map_location=lambda storage, loc: storage)
            self.model['backbone'].load_state_dict(checkpoint['backbone'])
            self.model['auxilnet'].load_state_dict(checkpoint['auxilnet'])
            print('checkpoint was trained %3d epoches ...' % checkpoint['epoch'])
        print('Model loading was finished ...')
    
    
    def _face_detector(self, img):
        
        bbox, landmark = self.model['mtcnn'].detect_face(img)
        return bbox, landmark
        
        
    def _eval_face(self, img_list = None):
        
        self.model['backbone'].eval()
        
        if img_list is None and len(self.args.test_folder) > 2:
            img_list = os.listdir(self.args.test_folder)
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        with torch.no_grad():
            for idx, img_name in enumerate(img_list):
            
                try:
                    img = cv2.imread(os.path.join('data/imgs', img_name))
                    img = cv2.resize(img, (self.args.in_size, self.args.in_size))
                    img_copy = img.copy()
                except Exception as e:
                    print(e)
                    continue
                else:
                    img = transform(img).unsqueeze(0)
                    if self.device:
                        img = img.cuda(0)
                    _, lmkpts = self.model['backbone'](img)
                    lmkpts = (lmkpts.cpu().numpy().reshape(-1, 2) * self.args.in_size).astype(np.int32)
                    for (x, y) in lmkpts:
                        cv2.circle(img_copy, (x, y), 1, (255,0,0), -1)
                    img_name = 'data/results/r_' + img_name
                    cv2.imwrite(img_name, img_copy)
            print('Evalution was finished ...')
    
    
    def _eval_imgs(self, img_list = None):
        
        self.model['backbone'].eval()
        
        if img_list is None and len(self.args.test_folder) > 2:
            img_list = os.listdir(self.args.test_folder)
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        with torch.no_grad():
            for idx, img_name in enumerate(img_list):
            
                try:
                    img = cv2.imread(os.path.join('data/imgs', img_name))
                    img_copy = img.copy()
                except Exception as e:
                    print(e)
                    continue
                else:
                    bboxes, _ = self._face_detector(img)
                    for bbox in bboxes.astype(np.int32):
                        
                        crop_face = img[bbox[1]:bbox[3]+1, bbox[0]:bbox[2]+1, :]
                        face_h, face_w, _ = crop_face.shape
                        crop_face = cv2.resize(crop_face, (self.args.in_size, self.args.in_size))
                        crop_face = transform(crop_face).unsqueeze(0)
                        if self.device:
                            crop_face = crop_face.cuda(0)
                        _, lmkpts = self.model['backbone'](crop_face)
                        lmkpts = (lmkpts.cpu().numpy().reshape(-1, 2) * np.array([face_w, face_h]) + bbox[:2]).astype(np.int32)   # TODO
                         
                        for (x, y) in lmkpts:
                            cv2.circle(img_copy, (x, y), 1, (255,0,0), -1)
                    img_name = 'data/results/r_' + img_name
                    cv2.imwrite(img_name, img_copy)
            print('Evalution was finished ...')


    def demo_runner(self):

        self._report_settings()

        self._model_loader()

        self._eval_imgs()


        
def demo_args():
    parser = argparse.ArgumentParser(description='Evaluate Practical Facial Landmark Detector')

    # env
    parser.add_argument('--use_gpu',  type=bool, default=False)
    parser.add_argument('--gpu_ids',  type=list, default=[0, 1])

    # checkpoint
    parser.add_argument('--in_size', type=int, default=112)
    parser.add_argument('--resume',  type=str, default='checkpoint/checkpoint_ori.pth.tar')
    
    # test-folder
    parser.add_argument('--test_folder', type=str, default='data/imgs/')

    args = parser.parse_args()
    return args

        
if __name__ == "__main__":

    pfld_trainer = PFLDDemo(demo_args())
    pfld_trainer.demo_runner()
