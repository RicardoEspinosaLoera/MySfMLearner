from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil
import cv2

from .mono_dataset import MonoDataset


class SCAREDDataset(MonoDataset):
    def __init__(self, *args, **kwargs):
        super(SCAREDDataset, self).__init__(*args, **kwargs)

        """self.K = np.array([[0.82, 0, 0.5, 0],
                           [0, 1.02, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)"""
        #256 / 320
        #fx769.807403688120 fy769.720558534159 cx675.226397736271 cy548.903474592445 k1-0.454260397098776 k20.179156666748519 k3-0.0285017743214105 p1-0.00134889190333418 p20.000738912923806121 skew-0.141152521412316
        """self.K = np.array([[2.40, -0.141152521412316, 2.11, 0],
                           [0, 3.00,2.14, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)"""
        #RNNSLAM synthetic dataset
        #480 / 640
        #fx = 232.5044678; fy = 232.5044678; cx = 240.0; cy = 320.0; baseline = 4.5; %unit in milimeter
        """self.K = np.array([[0.3632, 0, 0.375, 0],
                           [0, 0.4843,0.666, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)"""
        #Colon10k dataset
        #270 / 216
        #Camera Intrinsics: Pinhole fx=145.4410 fy=145.4410 cx=135.6993 cy=107.8946 width=270 height=216
        self.K = np.array([[0.6733, 0, 0.6282, 0],
                           [0, 0.5386,0.3996, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        # self.full_res_shape = (1280, 1024)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    def check_depth(self):
        
        return False

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))
        
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color


class SCAREDRAWDataset(SCAREDDataset):
    def __init__(self, *args, **kwargs):
        super(SCAREDRAWDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        #f_str = "{:010d}{}".format(frame_index, self.img_ext)
        #image_path = os.path.join(self.data_path, folder, "image_0{}/data".format(self.side_map[side]), f_str)
        #print(frame_index)
        """
        if(len(str(frame_index))==1):
            f_str = ("0000"+str(frame_index) + self.img_ext)
        elif(len(str(frame_index))==2):
            f_str = ("000"+str(frame_index) + self.img_ext)
        elif(len(str(frame_index))==3):
            f_str = ("00"+str(frame_index) + self.img_ext)
        elif(len(str(frame_index))==4):
            f_str = ("0"+str(frame_index) + self.img_ext)
        """
        f_str=str(frame_index) + self.img_ext
        image_path = os.path.join(self.data_path, folder, f_str)
            
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        f_str = "scene_points{:06d}.tiff".format(frame_index-1)

        depth_path = os.path.join(
            self.data_path,
            folder,
            "image_0{}/data/groundtruth".format(self.side_map[side]),
            f_str)

        depth_gt = cv2.imread(depth_path, 3)
        depth_gt = depth_gt[:, :, 0]
        depth_gt = depth_gt[0:1024, :]
        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt


