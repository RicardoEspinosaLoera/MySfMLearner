from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
from collections import OrderedDict


class PoseDecoder(nn.Module):
    def __init__(self, num_ch_enc, num_input_features, num_frames_to_predict_for=None, stride=1):
        super(PoseDecoder, self).__init__()

        #self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features
        #self.num_ch_enc = [128,128,256,512,1024]
        self.num_ch_enc = [1024,512,256,128,128]

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for

        #self.convs = OrderedDict()
        self.convs : Dict[string] = {} 
        #self.convs["squeeze"] = nn.Conv2d(int(self.num_ch_enc[-1]), 256, 1)
        self.squeeze = nn.Conv2d(int(self.num_ch_enc[-1] * 2), 256, 1)
        #self.convs["pose_0"] = nn.Conv2d(int(num_input_features * 256), 256, 3, stride, 1)
        self.pose_0 = nn.Conv2d(int(num_input_features * 256), 256, 3, stride, 1)
        #self.convs["pose_1"] = nn.Conv2d(256, 256, 3, stride, 1)
        self.pose_1 = nn.Conv2d(256, 256, 3, stride, 1)
        #self.convs["pose_2"] = nn.Conv2d(256, 6 * num_frames_to_predict_for, 1)
        self.pose_2 = nn.Conv2d(256, 6 * num_frames_to_predict_for, 1)

        self.relu = nn.ReLU()

        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, input_features):
        last_features = [f[-1] for f in input_features]
        #print(last_features[0].shape)
        cat_features = [self.relu(self.squeeze(f)) for f in last_features]
        cat_features = torch.cat(cat_features, 1)

        out = cat_features

        out = self.pose_0(out)
        out = self.relu(out)
        out = self.pose_1(out)
        out = self.relu(out)
        out = self.pose_2(out)

        """
        for i in range(3):
            out = self.convs["pose_"+str(i)](out)
            if i != 2:
                out = self.relu(out)
        """
        out = out.mean(3).mean(2)

        out = 0.001*out.view(-1, self.num_frames_to_predict_for, 1, 6)

        axisangle = out[..., :3]
        translation = out[..., 3:]

        return axisangle, translation
