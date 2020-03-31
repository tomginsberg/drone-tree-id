import torch

from detectron2.config import CfgNode as CN


def add_deepent_config(cfg):
    """
    Add custom config.
    """
    _C = cfg

    _C.MODEL.RESNETS.IN_FEATURES = CN()
    # in-fusing location of depth encoder feature maps
    _C.MODEL.RESNETS.IN_FEATURES = ["res3", "res5"]
    _C.MODEL.RESNETS.FUSE_METHOD = CN()
    # fuse method = {sum, lateral}
    _C.MODEL.RESNETS.FUSE_METHOD = "sum"

    # Add config for depth encoder
    _C.MODEL.DEPTH_ENCODER = CN()

    _C.MODEL.DEPTH_ENCODER.STEM_OUT_CHANNELS = 64
    _C.MODEL.DEPTH_ENCODER.FREEZE_AT = 0
    # depth encoder feature maps to fuse into backbone resnet
    _C.MODEL.DEPTH_ENCODER.OUT_FEATURES = ["res3", "res5"]
    _C.MODEL.DEPTH_ENCODER.RES2_OUT_CHANNELS = 256

    if not torch.cuda.is_available():
        print('No CUDA, Defaulting to CPU :(')
        _C.MODEL.DEVICE = 'cpu'
