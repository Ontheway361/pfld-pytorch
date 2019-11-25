#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on 2019/11/22
author: relu
'''
import argparse
from pfld import WLFWDate

root_dir  = '/Users/relu/data/benchmark_images/faceu/face_landmark/WFLW'

def data_args():
    parser = argparse.ArgumentParser(description='Generate the train/test dataset')

    parser.add_argument('--data_dir',     type=str,  default=root_dir)
    parser.add_argument('--anno_file',    type=str,  default='wflw_anno/raw_anno_test.txt')   # TODO
    parser.add_argument('--mirror_file',  type=str,  default='wflw_anno/Mirror98.txt')

    parser.add_argument('--in_size',      type=int,  default=112)
    parser.add_argument('--is_debug',     type=str,  default=False)
    parser.add_argument('--augment',      type=bool, default=False)            # TODO
    parser.add_argument('--num_faces',    type=int,  default=10)
    parser.add_argument('--save_folder',  type=str,  default='wflw_eval')      # TODO


    args = parser.parse_args()
    return args


if __name__ == '__main__':

    wlfw_data = WLFWDate(data_args())
    wlfw_data.process_engine()
