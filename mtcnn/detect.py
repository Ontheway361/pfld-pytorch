#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import time
import torch
import argparse
import torchvision
import numpy as np
from mtcnn.models import PNet, RNet, ONet
from torch.autograd.variable import Variable

from IPython import embed


def fix_args():
    
    parser = argparse.ArgumentParser(description='Face detector for PFLD')

    # checkpoint
    parser.add_argument('--pnet', type=str, default='mtcnn/checkpoint/pnet.pt')
    parser.add_argument('--rnet', type=str, default='mtcnn/checkpoint/rnet.pt')
    parser.add_argument('--onet', type=str, default='mtcnn/checkpoint/onet.pt')

    args = parser.parse_args()
    
    return args


def convert_chwTensor_to_hwcNumpy(tensor):
    """convert a group images pytorch tensor(count * c * h * w) to numpy array images(count * h * w * c)
            Parameters:
            ----------
            tensor: numpy array , count * c * h * w

            Returns:
            -------
            numpy array images: count * h * w * c
            """

    if isinstance(tensor, Variable):
        return np.transpose(tensor.data.numpy(), (0,2,3,1))
    elif isinstance(tensor, torch.FloatTensor):
        return np.transpose(tensor.numpy(), (0,2,3,1))
    else:
        raise Exception("covert b*c*h*w tensor to b*h*w*c numpy error.This tensor must have 4 dimension.")


class MtcnnDetector(object):
    ''' P,R,O net face detection and landmarks align '''

    def  __init__(self, use_gpu = False, gpu_ids = [0]):
  
        self.use_gpu       = use_gpu  
        self.gpu_ids       = gpu_ids
        self.min_face_size = 24   # for what ?
        self.thresh        = [0.6, 0.7, 0.7]
        self.scale_factor  = 0.709
        self.args          = fix_args()
        self.create_mtcnn_net()
    
    
    def create_mtcnn_net(self):
        ''' Create the mtcnn model '''
        pnet, rnet, onet = None, None, None

        if len(self.args.pnet) > 0:
            pnet = PNet(use_cuda=self.use_gpu)
            if self.use_gpu:
                pnet.load_state_dict(torch.load(self.args.pnet))
                pnet = torch.nn.DataParallel(pnet, device_ids=self.gpu_ids)
            else:
                pnet.load_state_dict(torch.load(self.args.pnet, map_location=lambda storage, loc: storage))
            pnet.eval()

        if len(self.args.rnet) > 0:
            rnet = RNet(use_cuda=self.use_gpu)
            if self.use_gpu:
                rnet.load_state_dict(torch.load(self.args.rnet))
                rnet = torch.nn.DataParallel(rnet, device_ids=self.gpu_ids)
            else:
                rnet.load_state_dict(torch.load(self.args.rnet, map_location=lambda storage, loc: storage))
            rnet.eval()

        if len(self.args.onet) > 0:
            onet = ONet(use_cuda=self.use_gpu)
            if self.use_gpu:
                onet.load_state_dict(torch.load(self.args.onet))
                onet = torch.nn.DataParallel(onet, device_ids=self.gpu_ids)
            else:
                onet.load_state_dict(torch.load(self.args.onet, map_location=lambda storage, loc: storage))
            onet.eval()

        self.pnet_detector = pnet
        self.rnet_detector = rnet
        self.onet_detector = onet
    
    @staticmethod
    def nms(dets, thresh, mode = 'Union'):

        x1, y1 = dets[:, 0], dets[:, 1]
        x2, y2 = dets[:, 2], dets[:, 3]
        order  = dets[:, 4].argsort()[::-1]  # descending order
        areas  = (x2 - x1 + 1) * (y2 - y1 + 1)

        keep = []
        while order.size > 0:

            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h

            if mode == 'Union':
                ovr = inter / (areas[i] + areas[order[1:]] - inter)
            elif mode == 'Minimum':
                ovr = inter / np.minimum(areas[i], areas[order[1:]])

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1] # +1: eliminates the first element in order

        return keep
    
    
    def unique_image_format(self, im):

        if not isinstance(im, np.ndarray):
            if im.mode == 'I':
                im = np.array(im, np.int32, copy=False)
            elif im.mode == 'I;16':
                im = np.array(im, np.int16, copy=False)
            else:
                im = np.asarray(im)
        return im


    def square_bbox(self, bbox):
        """
            convert bbox to square
        Parameters:
        ----------
            bbox: numpy array , shape n x m
                input bbox
        Returns:
        -------
            a square bbox
        """
        square_bbox = bbox.copy()
        h = bbox[:, 3] - bbox[:, 1] + 1
        w = bbox[:, 2] - bbox[:, 0] + 1
        l = np.maximum(h, w)

        square_bbox[:, 0] = bbox[:, 0] + (w - l) * 0.5
        square_bbox[:, 1] = bbox[:, 1] + (h - l) * 0.5
        square_bbox[:, 2] = square_bbox[:, 0] + l - 1
        square_bbox[:, 3] = square_bbox[:, 1] + l - 1

        return square_bbox


    def generate_bbox(self, score_map, reg, scale, threshold):
        """
            generate bbox from feature map
        Parameters:
        ----------
            score_map : numpy array , n x m x 1, detect score for each position
            reg       : numpy array , n x m x 4, bbox
            landmark  : numpy array,  n x m x 136, bbox
            scale     : float number, scale of this detection
            threshold : float number, detect threshold
        Returns:
        ----------
            bbox array
        """

        stride, cellsize = 2, 12
        t_index = np.where(score_map > threshold)  # row, col | y, x

        if t_index[0].size == 0:
            return np.array([])

        reg   = np.array([reg[0, t_index[0], t_index[1], i] for i in range(4)])
        score = score_map[t_index[0], t_index[1], 0]

        # BUG
        bbox = np.vstack([np.round((stride * t_index[1]) / scale),            # x1 of prediction box in original image
                          np.round((stride * t_index[0]) / scale),            # y1 of prediction box in original image
                          np.round((stride * t_index[1] + cellsize) / scale), # x2 of prediction box in original image
                          np.round((stride * t_index[0] + cellsize) / scale), # y2 of prediction box in original image
                          score, reg]).T
        return bbox


    def resize_image(self, img, scale):
        """
            resize image and transform dimention to [batchsize, channel, height, width]
        Parameters:
        ----------
            img: numpy array , height x width x channel, input image, channels in BGR order here
            scale: float number, scale factor of resize operation
        Returns:
        -------
            transformed image tensor , 1 x channel x height x width
        """
        height, width, channels = img.shape
        new_height = int(height * scale)     # resized new height
        new_width  = int(width * scale)       # resized new width
        new_dim    = (new_width, new_height)
        img_resized = cv2.resize(img, new_dim, interpolation=cv2.INTER_LINEAR)      # resized image
        return img_resized


    def boundary_check(self, bboxes, img_w, img_h):
        """
        deal with the boundary-beyond question
        Parameters:
        ----------
            bboxes: numpy array, n x 5, input bboxes
            w: float number, width of the input image
            h: float number, height of the input image
        Returns :
        ------
            x1, y1 : numpy array, n x 1, start point of the bbox in target image
            x2, y2 : numpy array, n x 1, end point of the bbox in target image
            anchor_y1, anchor_x1 : numpy array, n x 1, start point of the bbox in original image
            anchor_x1, anchor_x2 : numpy array, n x 1, end point of the bbox in original image
            box_h, box_w         : numpy array, n x 1, height and width of the bbox
        """

        nbox = bboxes.shape[0]

        # width and height
        box_w = (bboxes[:, 2] - bboxes[:, 0] + 1).astype(np.int32)
        box_h = (bboxes[:, 3] - bboxes[:, 1] + 1).astype(np.int32)

        x1, y1 = np.zeros((nbox,)), np.zeros((nbox,))
        x2, y2 = box_w.copy() - 1, box_h.copy() - 1
        anchor_x1, anchor_y1 = bboxes[:, 0], bboxes[:, 1],
        anchor_x2, anchor_y2 = bboxes[:, 2], bboxes[:, 3]

        idx      = np.where(anchor_x2 > img_w - 1)
        x2[idx]  = box_w[idx] + img_w - 2 - anchor_x2[idx]
        anchor_x2[idx]  = img_w - 1

        idx      = np.where(anchor_y2 > img_h-1)
        y2[idx]  = box_h[idx] + img_h - 2 - anchor_y2[idx]
        anchor_y2[idx]  = img_h - 1

        idx     = np.where(anchor_x1 < 0)
        x1[idx] = 0 - anchor_x1[idx]
        anchor_x1[idx] = 0

        idx     = np.where(anchor_y1 < 0)
        y1[idx] = 0 - anchor_y1[idx]
        anchor_y1[idx] = 0

        return_list = [y1, y2, x1, x2, anchor_y1, anchor_y2, anchor_x1, anchor_x2, box_w, box_h]
        return_list = [item.astype(np.int32) for item in return_list]

        return return_list


    def detect_pnet(self, im):
        """Get face candidates through pnet

        Parameters:
        ----------
        im: numpy array, input image array, one batch

        Returns:
        -------
        boxes: numpy array, detected boxes before calibration
        boxes_align: numpy array, boxes after calibration
        """

        # im = self.unique_image_format(im)
        h, w, c = im.shape
        net_size = 12

        current_scale = float(net_size) / self.min_face_size   # scale = 1.0
        im_resized    = self.resize_image(im, current_scale)
        current_height, current_width, _ = im_resized.shape

        all_boxes = list()
        while min(current_height, current_width) > net_size:

            feed_imgs = []
            image_tensor = torchvision.transforms.ToTensor()(im_resized)
            feed_imgs.append(image_tensor)
            feed_imgs = Variable(torch.stack(feed_imgs))

            if self.pnet_detector.use_cuda:
                feed_imgs = feed_imgs.cuda()

            cls_map, reg, _ = self.pnet_detector(feed_imgs)   # CORE， Don't look landmark

            cls_map_np  = convert_chwTensor_to_hwcNumpy(cls_map.cpu())
            reg_np      = convert_chwTensor_to_hwcNumpy(reg.cpu())

            # boxes = [x1, y1, x2, y2, score, reg]
            boxes = self.generate_bbox(cls_map_np[ 0, :, :], reg_np, current_scale, self.thresh[0])

            # generate pyramid images
            current_scale *= self.scale_factor # self.scale_factor = 0.709
            im_resized = self.resize_image(im, current_scale)
            current_height, current_width, _ = im_resized.shape

            if boxes.size == 0:
                continue

            # non-maximum suppresion
            keep = self.nms(boxes[:, :5], 0.5, 'Union')
            boxes = boxes[keep]
            all_boxes.append(boxes)

        if len(all_boxes) == 0:
            return None

        all_boxes = np.vstack(all_boxes)

        keep = self.nms(all_boxes[:, :5], 0.7, 'Union')
        all_boxes = all_boxes[keep]

        bw = all_boxes[:, 2] - all_boxes[:, 0] + 1
        bh = all_boxes[:, 3] - all_boxes[:, 1] + 1

        # all_boxes = [x1, y1, x2, y2, score, reg]
        align_x1 = all_boxes[:, 0] + all_boxes[:, 5] * bw
        align_y1 = all_boxes[:, 1] + all_boxes[:, 6] * bh
        align_x2 = all_boxes[:, 2] + all_boxes[:, 7] * bw
        align_y2 = all_boxes[:, 3] + all_boxes[:, 8] * bh

        boxes_align = np.vstack([align_x1, align_y1, align_x2, align_y2, all_boxes[:, 4],]).T

        return boxes_align


    def detect_rnet(self, im, dets):
        """Get face candidates using rnet

        Parameters:
        ----------
        im: numpy array, input image array
        dets: numpy array, detection results of pnet

        Returns:
        -------
        boxes: numpy array, detected boxes before calibration
        boxes_align: numpy array, boxes after calibration
        """
        # im: an input image
        h, w, c = im.shape

        if dets is None:
            return None


        dets = self.square_bbox(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])

        [y1, y2, x1, x2, anchor_y1, anchor_y2, \
         anchor_x1, anchor_x2, box_w, box_h] = self.boundary_check(dets, w, h)

        num_boxes = dets.shape[0]

        cropped_ims_tensors = []
        for i in range(num_boxes):

            tmp_img = np.zeros((box_h[i], box_w[i], 3), dtype=np.uint8)
            tmp_img[y1[i]:y2[i]+1, x1[i]:x2[i]+1, :] = im[anchor_y1[i]:anchor_y2[i]+1, anchor_x1[i]:anchor_x2[i]+1, :]
            crop_im = cv2.resize(tmp_img, (24, 24))
            crop_im_tensor = torchvision.transforms.ToTensor()(crop_im)
            cropped_ims_tensors.append(crop_im_tensor)

        feed_imgs = Variable(torch.stack(cropped_ims_tensors))

        if self.rnet_detector.use_cuda:
            feed_imgs = feed_imgs.cuda()

        cls_map, reg, _ = self.rnet_detector(feed_imgs)  # CORE

        cls_map = cls_map.cpu().data.numpy()
        reg = reg.cpu().data.numpy()

        keep_inds = np.where(cls_map > self.thresh[1])[0]

        if len(keep_inds) > 0:
            boxes = dets[keep_inds]    # NOTE :: det_box from Pnet
            cls   = cls_map[keep_inds]
            reg   = reg[keep_inds]
        else:
            return None

        keep = self.nms(boxes, 0.7)

        if len(keep) == 0:
            return None

        keep_cls   = cls[keep]
        keep_boxes = boxes[keep]
        keep_reg   = reg[keep]

        bw = keep_boxes[:, 2] - keep_boxes[:, 0] + 1
        bh = keep_boxes[:, 3] - keep_boxes[:, 1] + 1

        align_x1 = keep_boxes[:,0] + keep_reg[:,0] * bw
        align_y1 = keep_boxes[:,1] + keep_reg[:,1] * bh
        align_x2 = keep_boxes[:,2] + keep_reg[:,2] * bw
        align_y2 = keep_boxes[:,3] + keep_reg[:,3] * bh

        boxes_align = np.vstack([align_x1, align_y1, align_x2, align_y2, keep_cls[:, 0]]).T

        return boxes_align


    def detect_onet(self, im, dets):
        """Get face candidates using onet

        Parameters:
        ----------
        im: numpy array, input image array
        dets: numpy array, detection results of rnet

        Returns:
        -------
        boxes_align: numpy array, boxes after calibration
        landmarks_align: numpy array, landmarks after calibration

        """
        h, w, c = im.shape

        if dets is None:
            return None, None

        dets = self.square_bbox(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])

        [y1, y2, x1, x2, anchor_y1, anchor_y2, \
         anchor_x1, anchor_x2, box_w, box_h] = self.boundary_check(dets, w, h)
        num_boxes = dets.shape[0]

        cropped_ims_tensors = []
        for i in range(num_boxes):

            tmp_img = np.zeros((box_h[i], box_w[i], 3), dtype=np.uint8)
            tmp_img[y1[i]:y2[i]+1, x1[i]:x2[i]+1, :] = im[anchor_y1[i]:anchor_y2[i]+1, anchor_x1[i]:anchor_x2[i]+1, :]
            crop_im = cv2.resize(tmp_img, (48, 48))
            crop_im_tensor = torchvision.transforms.ToTensor()(crop_im)
            cropped_ims_tensors.append(crop_im_tensor)

        feed_imgs = Variable(torch.stack(cropped_ims_tensors))

        if self.rnet_detector.use_cuda:
            feed_imgs = feed_imgs.cuda()

        cls_map, reg, landmark = self.onet_detector(feed_imgs)  # look all

        cls_map   = cls_map.cpu().data.numpy()
        reg       = reg.cpu().data.numpy()
        landmark  = landmark.cpu().data.numpy()
        keep_inds = np.where(cls_map > self.thresh[2])[0]

        if len(keep_inds) > 0:
            boxes    = dets[keep_inds]
            cls      = cls_map[keep_inds]
            reg      = reg[keep_inds]
            landmark = landmark[keep_inds]
        else:
            return None, None

        keep =self.nms(boxes, 0.7, mode="Minimum")

        if len(keep) == 0:
            return None, None

        keep_cls      = cls[keep]
        keep_boxes    = boxes[keep]
        keep_reg      = reg[keep]
        keep_landmark = landmark[keep]

        bw = keep_boxes[:, 2] - keep_boxes[:, 0] + 1
        bh = keep_boxes[:, 3] - keep_boxes[:, 1] + 1


        align_x1 = keep_boxes[:, 0] + keep_reg[:, 0] * bw
        align_y1 = keep_boxes[:, 1] + keep_reg[:, 1] * bh
        align_x2 = keep_boxes[:, 2] + keep_reg[:, 2] * bw
        align_y2 = keep_boxes[:, 3] + keep_reg[:, 3] * bh

        boxes_align = np.vstack([align_x1, align_y1, align_x2, align_y2, keep_cls[:, 0]]).T

        # TODO :: 68 <--> 5
        lmk_pts = keep_landmark.copy()
        x_idx   = [2*s for s in range(5)]
        y_idx   = [2*s+1 for s in range(5)]
        for idx in range(lmk_pts.shape[0]):
            lmk_pts[idx, x_idx] = keep_boxes[idx, 0] + lmk_pts[idx, x_idx] * bw[idx]
            lmk_pts[idx, y_idx] = keep_boxes[idx, 1] + lmk_pts[idx, y_idx] * bh[idx]

        return boxes_align, lmk_pts


    def detect_face(self, img):
        ''' Detect face over image '''

        boxes_align    = np.array([])
        landmark_align = np.array([])

        t = time.time()

        # pnet
        if self.pnet_detector:
            boxes_align = self.detect_pnet(img)   # CORE
            if boxes_align is None:
                return np.array([]), np.array([])

            t1 = time.time() - t
            t = time.time()

        # rnet
        if self.rnet_detector:
            boxes_align = self.detect_rnet(img, boxes_align)  # CORE
            if boxes_align is None:
                return np.array([]), np.array([])

            t2 = time.time() - t
            t = time.time()

        # onet
        if self.onet_detector:
            boxes_align, landmark_align = self.detect_onet(img, boxes_align)  # CORE
            if boxes_align is None: 
                return np.array([]), np.array([])

            t3 = time.time() - t
            t = time.time()
#             print("time cost " + '{:.3f}'.format(t1+t2+t3) + '  pnet {:.3f}  rnet {:.3f}  onet {:.3f}'.format(t1, t2, t3))

        return boxes_align, landmark_align
