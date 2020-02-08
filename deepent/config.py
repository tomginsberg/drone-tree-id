import torch

from detectron2.config import CfgNode as CN


def add_deepent_config(cfg):
    """
    Add custom config.
    """
    _C = cfg

    _C.MODEL.RESNETS.IN_FEATURES = CN()
    _C.MODEL.RESNETS.IN_FEATURES = ["res2", "res3", "res4", "res5"]

    # Add config for depth encoder
    _C.MODEL.DEPTH_ENCODER = CN()

    _C.MODEL.DEPTH_ENCODER.STEM_OUT_CHANNELS = 64
    _C.MODEL.DEPTH_ENCODER.FREEZE_AT = 0
    _C.MODEL.DEPTH_ENCODER.OUT_FEATURES = ["d_res2", "d_res3", "d_res4", "d_res5"]
    _C.MODEL.DEPTH_ENCODER.RES2_OUT_CHANNELS = 256

    if not torch.cuda.is_available():
        print('No CUDA, Training on CPU :(')
        _C.MODEL.DEVICE = 'cpu'
