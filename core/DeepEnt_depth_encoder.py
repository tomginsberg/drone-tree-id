import math
import torch
import fvcore.nn.weight_init as weight_init
import torch.nn.functional as F
import torchvision
from torch import nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from copy import deepcopy

from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone.resnet import build_resnet_backbone

from DeepEnt_depth_encoder import *

def make_layers_from_size(channels):
	layers = []
	for size in channels:
		layers += [nn.Conv2d(size[0], size[1], kernel_size=3, padding=1), nn.BatchNorm2d(size[1],momentum = 0.1), nn.ReLU(inplace=True)]
	return nn.Sequential(*layers)

#@TODO: fix this implementation, fix dimensionality 
class depth_encoder(Backbone):
    def __init__(feature_map_shapes):
        stage_channels = [3,64,128,256,512,512]
        self.CBR1_DEPTH_ENC = make_layers_from_size(((stage_channels[0], stage_channels[1]), (stage_channels[1], stage_channels[1])))
        self.pool1_d = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.CBR2_DEPTH_ENC = make_layers_from_size(((stage_channels[1], stage_channels[2]), (stage_channels[2], stage_channels[2])))
        self.pool2_d = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.CBR3_DEPTH_ENC = make_layers_from_size(((stage_channels[2], stage_channels[3]), (stage_channels[3], stage_channels[3]),(stage_channels[3], stage_channels[3])))
        self.pool3_d = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.dropout3_d = nn.Dropout(p=0.4)

        self.CBR4_DEPTH_ENC = make_layers_from_size(((stage_channels[3], stage_channels[4]), (stage_channels[4], stage_channels[4]),(stage_channels[4], stage_channels[4])))
        self.pool4_d = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.dropout4_d = nn.Dropout(p=0.4)

        self.CBR5_DEPTH_ENC = make_layers_from_size(((stage_channels[4], stage_channels[5]), (stage_channels[5], stage_channels[5]),(stage_channels[5], stage_channels[5])))

    def fwd(x):
        outputs = {}
        x_1 = self.CBR1_DEPTH_ENC(x)
        outputs["dep1"] = x_1
        x, id1_d = self.pool1_d(x_1)

        # Stage 2
        x_2 = self.CBR2_DEPTH_ENC(x)
        outputs["dep2"] = x_2 
        x, id2_d = self.pool2_d(x_2)

        # Stage 3
        x_3 = self.CBR3_DEPTH_ENC(x)
        outputs["dep3"] = x_3
        x, id3_d = self.pool4_d(x_3)
        x = self.dropout3_d(x)

        # Stage 4
        x_4 = self.CBR4_DEPTH_ENC(x)
        outputs["dep4"] = x_4
        x, id4_d = self.pool4_d(x_4)
        x = self.dropout4_d(x)

        # Stage 5
        x_5 = self.CBR5_DEPTH_ENC(x)
        outputs["dep5"] = x_5
        return outputs

def build_depth_encoder(cfg, feature_map_shapes):
    model = depth_encoder(feature_map_shapes)
    return model