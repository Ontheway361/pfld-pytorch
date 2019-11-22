#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on 2019/11/22
author: relu
'''

import os
import cv2
import shutil
import numpy as np

from IPython import embed
#--------------------------------------------------------
#  0-195  :landmark | (x1,y1, ..., x98, y98)
# 196-199 :  bbox   | (x1,y1, x2, y2)
# 200-205 :face_attr| [pose, expression, illumination, \
#                      make-up, occlusion, blur]
#   206   :img_name
#--------------------------------------------------------

class WLFWDate(object):

    def __init__(self, args):

        self.args        = args
        self._fetch_anno()


    def _fetch_anno(self):
        ''' Fetch the info of annotations '''
        try:
            with open(os.path.join(self.args.data_dir, self.args.anno_file), 'r') as f:
                self.anno_file = [info.strip().split(' ') for info in f.readlines()]
            f.close()
            with open(os.path.join(self.args.data_dir, self.args.mirror_file), 'r') as f:
                self.mirror_idx = list(map(int, f.readlines()[0].strip().split(',')))
            f.close()
        except:
            print(self.args.anno_file, self.args.mirror_file)
            raise TypeError('Wrong file_path, please check...')
        else:
            print('Fetch-anno was finished ...')


    @staticmethod
    def _rotate_img(angle, center, landmark):
        ''' Clockwise rotate image according to the center '''
        rad   = angle * np.pi / 180.0
        cos_r, sin_r = np.cos(rad), np.sin(rad)
        r_matrix = np.zeros((2,3), dtype=np.float32)
        r_matrix[0, 0], r_matrix[0, 1] = cos_r, sin_r
        r_matrix[1, 0], r_matrix[1, 1] = -sin_r, cos_r
        r_matrix[0, 2] = (1-cos_r) * center[0] - sin_r * center[1]
        r_matrix[1, 2] = sin_r * center[0] + (1-cos_r) * center[1]

        r_landmark = np.asarray([(r_matrix[0,0] * x + r_matrix[0,1] * y + r_matrix[0,2],
                                 r_matrix[1,0] * x + r_matrix[1,1] * y + r_matrix[1,2]) \
                                 for (x,y) in landmark])
        return r_matrix, r_landmark


    def _augment_face(self, img, landmark):
        '''
        Augment the crop_face with rotation and filp

        step - 1. prepare the basic anno-info of image
        step - 2. get the info of rotation and rotate the img
        step - 3. trans landmark and crop the face
        step - 4. flip crop_face and landmark or not
        '''
        # step - 1
        min_xy = np.min(landmark, axis=0).astype(np.int32)
        max_xy = np.max(landmark, axis=0).astype(np.int32)
        box_wh = max_xy - min_xy + 1
        center = min_xy + box_wh // 2
        longer = int(np.max(box_wh) * 1.2)
        face_list, landmark_list = [], []
        while(len(face_list) < self.args.num_faces):

            # step - 2
            rotate_angle = np.random.randint(-30, 30)
            center_x, center_y = center
            center_x += int(np.random.randint(-0.1 * longer, 0.1 * longer))
            center_y += int(np.random.randint(-0.1 * longer, 0.1 * longer))
            rotate_mat, rotate_lmk = self._rotate_img(rotate_angle, (center_x, center_y), landmark)
            loose_size = (int(img.shape[1] * 1.1), int(img.shape[0] * 1.1))
            trans_face = cv2.warpAffine(img, rotate_mat, loose_size)

            # step - 3
            wh = np.ptp(rotate_lmk, axis=0).astype(np.int32) + 1
            crop_size = np.random.randint(int(np.min(wh)), np.ceil(np.max(wh) * 1.25))
            left_up   = np.asarray((center_x - crop_size // 2, center_y - crop_size // 2), dtype=np.int32)
            rotate_lmk = (rotate_lmk - left_up) / crop_size
            if (rotate_lmk < 0).any() or (rotate_lmk > 1).any():
                continue

            x1, y1 = left_up
            x2, y2 = left_up + crop_size
            height, width, _ = trans_face.shape
            sx, sy = max(0, -x1), max(0, -y1)
            ex, ey = max(0, x2 - width), max(0, y2 - height)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)

            crop_face = trans_face[y1:y2, x1:x2]
            if (sx > 0 or sy > 0 or ex >0 or ey > 0):
                crop_face = cv2.copyMakeBorder(crop_face, sy, ey, sx, ex, cv2.BORDER_CONSTANT, 0)

            crop_face = cv2.resize(crop_face, (self.args.in_size, self.args.in_size))

            # step -4
            if self.mirror_idx is not None and np.random.choice((True, False)):
                rotate_lmk[:,0] = 1 - rotate_lmk[:,0]
                rotate_lmk = rotate_lmk[self.mirror_idx]
                crop_face = cv2.flip(crop_face, 1)

            face_list.append(crop_face)
            landmark_list.append(rotate_lmk)

        return face_list, landmark_list


    def _crop_face(self, img, landmark):
        ''' Crop the original face from image '''

        run_status, crop_face, lmk_pts = 1, None, None
        height, width, _ = img.shape
        min_xy = np.min(landmark, axis=0).astype(np.int32)
        max_xy = np.max(landmark, axis=0).astype(np.int32)
        box_wh = max_xy - min_xy + 1
        center = min_xy + box_wh // 2
        longer = int(np.max(box_wh) * 1.2)
        fix_pt = center - longer // 2
        x1, y1 = fix_pt
        x2, y2 = center + longer // 2
        sx, sy = max(0, -x1), max(0, -y1)
        ex, ey = max(0, x2 - width), max(0, y2 - height)
        x1, y1 = max(0, x1),  max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        crop_face = img[y1:y2, x1:x2]
        if (sx > 0 or sy > 0 or ex > 0 or ey > 0):
            crop_face = cv2.copyMakeBorder(crop_face, sy, ey, sx, ex, cv2.BORDER_CONSTANT, 0)

        if min(crop_face.shape) == 0:
            run_status = 0
        else:
            crop_face = cv2.resize(crop_face, (self.args.in_size, self.args.in_size))
            lmk_pts   = (landmark - fix_pt) / longer
            if (lmk_pts < 0).any() or (lmk_pts > 1).any():
                run_status = 0
        return (run_status, [crop_face], [lmk_pts])


    @staticmethod
    def _euler_angle(landmarks, cam_w=256, cam_h=256):
        ''' Calculate pitch, yaw and roll acoording to keypoints '''

        assert landmarks is not None, 'landmarks_2D is None'

        # Estimated camera matrix values.
        c_x, c_y = cam_w / 2, cam_h / 2
        f_x = c_x / np.tan(60 / 2 * np.pi / 180)
        f_y = f_x
        camera_matrix     = np.float32([[f_x, 0.0, c_x], [0.0, f_y, c_y], [0.0, 0.0, 1.0]])
        camera_distortion = np.float32([0.0, 0.0, 0.0, 0.0, 0.0])

        # dlib (68 landmark) trached points
        # TRACKED_POINTS = [17, 21, 22, 26, 36, 39, 42, 45, 31, 35, 48, 54, 57, 8]
        # wflw(98 landmark) trached points
        TRACKED_POINTS = [33, 38, 50, 46, 60, 64, 68, 72, 55, 59, 76, 82, 85, 16]

        # X-Y-Z with X pointing forward and Y on the left and Z up.
        # The X-Y-Z coordinates used are like the standard coordinates of ROS (robotic operative system)
        # OpenCV uses the reference usually used in computer vision:
        # X points to the right, Y down, Z to the front
        landmarks_3D = np.float32([
            [6.825897, 6.760612, 4.402142],  # LEFT_EYEBROW_LEFT,
            [1.330353, 7.122144, 6.903745],  # LEFT_EYEBROW_RIGHT,
            [-1.330353, 7.122144, 6.903745],  # RIGHT_EYEBROW_LEFT,
            [-6.825897, 6.760612, 4.402142],  # RIGHT_EYEBROW_RIGHT,
            [5.311432, 5.485328, 3.987654],  # LEFT_EYE_LEFT,
            [1.789930, 5.393625, 4.413414],  # LEFT_EYE_RIGHT,
            [-1.789930, 5.393625, 4.413414],  # RIGHT_EYE_LEFT,
            [-5.311432, 5.485328, 3.987654],  # RIGHT_EYE_RIGHT,
            [-2.005628, 1.409845, 6.165652],  # NOSE_LEFT,
            [-2.005628, 1.409845, 6.165652],  # NOSE_RIGHT,
            [2.774015, -2.080775, 5.048531],  # MOUTH_LEFT,
            [-2.774015, -2.080775, 5.048531],  # MOUTH_RIGHT,
            [0.000000, -3.116408, 6.097667],  # LOWER_LIP,
            [0.000000, -7.415691, 4.070434],  # CHIN
        ])
        landmarks_2D = landmarks[TRACKED_POINTS]

        # Applying the PnP solver to find the 3D pose of the head from the 2D position of the landmarks.
        # retval - bool
        # rvec - Output rotation vector that, together with tvec, brings points from the world coordinate system to the camera coordinate system.
        # tvec - Output translation vector. It is the position of the world origin (SELLION) in camera co-ords
        _, rvec, tvec = cv2.solvePnP(landmarks_3D, landmarks_2D, camera_matrix, camera_distortion)
        rmat, _ = cv2.Rodrigues(rvec)
        pose_mat = cv2.hconcat((rmat, tvec))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
        pitch, yaw, roll = map(lambda k: k[0], euler_angles)
        return (pitch, yaw, roll )


    def _save_faces(self, faces, lmkpts, img_anno):

        prefix_name = img_anno[-1].strip().split('/')[-1].split('.')[0]
        attr_str    = ' '.join(img_anno[200:206])
        save_folder = os.path.join(self.args.data_dir, self.args.save_folder, 'images')
        faces_info  = []

        check_folder = self.args.data_dir
        for folder in [self.args.save_folder, 'images']:

            check_folder = os.path.join(check_folder, folder)
            if not os.path.exists(check_folder):
                os.mkdir(check_folder)

        for idx, (face, lmkpt) in enumerate(zip(faces, lmkpts)):

            if lmkpt.shape != (98, 2):
                continue
            face_name = prefix_name + '_' + str(idx) + '.png'
            save_path = os.path.join(save_folder, face_name)
            if cv2.imwrite(save_path, face):
                pitch, yaw, roll = self._euler_angle(lmkpt)
                angle_str = ' '.join([str(pitch), str(yaw), str(roll)])
                lmks_str  = ' '.join(list(map(str,lmkpt.reshape(-1).tolist())))
                face_info = '{} {} {} {}\n'.format(save_path, lmks_str, attr_str, angle_str)
                faces_info.append(face_info)
        return faces_info


    def _face_runner(self, img_anno):

        try:
            img_path = os.path.join(self.args.data_dir, 'wlfw_images', img_anno[-1])
            img = cv2.imread(img_path)
        except Exception as e:
            print(e)
            faceinfo = None
        else:
            landmark = np.asarray(list(map(float, img_anno[:196])), dtype=np.float32).reshape(-1, 2)
            run_status, faces, lmkpts = self._crop_face(img, landmark)
            if self.args.augment:
                face_list, lmk_list = self._augment_face(img, landmark)
                if run_status:
                    faces.extend(face_list)
                    lmkpts.extend(lmk_list)
                else:
                    faces  = face_list
                    lmkpts = lmk_list
            faceinfo = self._save_faces(faces, lmkpts, img_anno)
        return faceinfo


    def process_engine(self):

        faces_info = []
        for idx, img_anno in enumerate(self.anno_file):

            faceinfo = self._face_runner(img_anno)
            if faceinfo is not None:
                faces_info.extend(faceinfo)
            else:
                continue
            if (idx + 1) % 100 == 0:
                print('already processed %4d|%4d ...' % (idx + 1, len(self.anno_info)))
            if self.args.is_debug and (idx + 1) % 20 == 0:
                break
        save_file = os.path.join(self.args.data_dir, 'wflw_anno', self.args.save_folder + '.txt')
        with open(save_file, 'w') as f:
            for face_info in faces_info:
                f.writelines(face_info)
        print('Attention, file was saved in %s' % save_file)
