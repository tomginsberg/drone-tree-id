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

def VGG16_initializator():
    layer_names =["conv1_1","conv1_2","conv2_1","conv2_2","conv3_1","conv3_2","conv3_3","conv4_1","conv4_2","conv4_3","conv5_1","conv5_2","conv5_3"]
    layers = list(torchvision.models.vgg16(pretrained=True).features.children())
    layers = [x for x in layers if isinstance(x, nn.Conv2d)]
    layer_dic = dict(zip(layer_names,layers))
    return layer_dic

def make_layers_from_names(names,model_dic,bn_dim,existing_layer=None):
    layers = []
    if existing_layer is not None:
    	layers = [existing_layer,nn.BatchNorm2d(bn_dim,momentum = 0.1),nn.ReLU(inplace=True)]
    for name in names:
        layers += [deepcopy(model_dic[name]), nn.BatchNorm2d(bn_dim,momentum = 0.1), nn.ReLU(inplace=True)]

    return nn.Sequential(*layers)

#@TODO: fix this implementation, fix dimensionality 
class depth_encoder(Backbone):
    def __init__(feature_map_shapes):
        feats_depth = list(torchvision.models.vgg16(pretrained=True).features.children())
        avg = torch.mean(feats_depth[0].weight.data, dim=1)
        avg = avg.unsqueeze(1)

        conv11d = nn.Conv2d(1, feature_map_sizes[0], kernel_size=3,padding=1)
        conv11d.weight.data = avg
        model_dic = VGG16_initializator()
        self.CBR1_DEPTH_ENC = make_layers_from_names(["conv1_2"], model_dic, feature_map_sizes[0], conv11d)
        self.pool1_d = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.CBR2_DEPTH_ENC = make_layers_from_names(["conv2_1","conv2_2"], model_dic)
        self.pool2_d = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.CBR3_DEPTH_ENC = make_layers_from_names(["conv3_1","conv3_2","conv3_3"], model_dic)
        self.pool3_d = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.dropout3_d = nn.Dropout(p=0.4)

        self.CBR4_DEPTH_ENC = make_layers_from_names(["conv4_1","conv4_2","conv4_3"], model_dic)
        self.pool4_d = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.dropout4_d = nn.Dropout(p=0.4)

        self.CBR5_DEPTH_ENC = make_layers_from_names(["conv5_1","conv5_2","conv5_3"], model_dic, feature_map_sizes[4])

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