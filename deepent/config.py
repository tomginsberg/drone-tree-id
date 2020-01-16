from detectron2.config import CfgNode as CN


def add_deepent_config(cfg):
    """
    Add custom config.
    """
    _C = cfg

    _C.MODEL.CUSTOM = CN()
    _C.MODEL.DEVICE = 'cpu'
