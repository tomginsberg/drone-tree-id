from deepent.modelling.backbone.deepent_fuse_resnet import build_resnet_backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone.fpn import LastLevelMaxPool, ShapeSpec, FPN

__all__ = ["build_deepent_fpn_backbone"]


@BACKBONE_REGISTRY.register()
def build_deepent_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    :param cfg: a detectron2 CfgNode
    :param input_shape:
    :returns: backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_resnet_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone
