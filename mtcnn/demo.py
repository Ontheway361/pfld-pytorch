#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
from vision import vis_face
from detect import MtcnnDetector

from IPython import embed

if __name__ == '__main__':
    
    model_info = 'self'
    imglists   = [s.split('.')[0] for s in os.listdir('imgs/')]

    mtcnn_detector = MtcnnDetector(use_gpu=False)
    
    for img_name in imglists:
        
        img    = cv2.imread('imgs/%s.jpg' % img_name)
        img_bg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        print('img.shape : ', img.shape)
        bboxs, landmarks = mtcnn_detector.detect_face(img)

        save_name = 'result/r_%s_%s.jpg' % (img_name, model_info)
        print('save img name : %s' % save_name)
        vis_face(img_bg, bboxs, landmarks, save_name)
