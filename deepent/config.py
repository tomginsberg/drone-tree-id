import torch

from detectron2.config import CfgNode as CN


def add_deepent_config(cfg):
    """
    Add custom config.
    """
    _C = cfg

    _C.MODEL.CUSTOM = CN()

    if not torch.cuda.is_available():
        print('No CUDA, Training on CPU :(')
        _C.MODEL.DEVICE = 'cpu'
