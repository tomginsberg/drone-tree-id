from lib.detectron2.detectron2.layers import FrozenBatchNorm2d
from lib.detectron2.detectron2.modeling.backbone.resnet import BottleneckBlock, ResNet, make_stage, BasicStem
from detectron2.modeling import BACKBONE_REGISTRY

def build_depth_encoder_backbone(cfg, input_shape):
    """
    Create a ResNet instance from config.

    Returns:
        ResNet: a :class:`ResNet` instance.
    """
    # need registration of new blocks/stems?
    norm = cfg.MODEL.RESNETS.NORM
    stem = BasicStem(
        in_channels=input_shape.channels,
        out_channels=cfg.MODEL.DEPTH_ENCODER.STEM_OUT_CHANNELS,
        norm=norm,
    )
    freeze_at = cfg.MODEL.DEPTH_ENCODER.FREEZE_AT

    if freeze_at >= 1:
        for p in stem.parameters():
            p.requires_grad = False
        stem = FrozenBatchNorm2d.convert_frozen_batchnorm(stem)

    # fmt: off
    out_features        = cfg.MODEL.DEPTH_ENCODER.OUT_FEATURES
    depth               = cfg.MODEL.RESNETS.DEPTH
    num_groups          = cfg.MODEL.RESNETS.NUM_GROUPS
    width_per_group     = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
    bottleneck_channels = num_groups * width_per_group
    in_channels         = cfg.MODEL.DEPTH_ENCODER.STEM_OUT_CHANNELS
    out_channels        = cfg.MODEL.DEPTH_ENCODER.RES2_OUT_CHANNELS
    stride_in_1x1       = cfg.MODEL.RESNETS.STRIDE_IN_1X1
    res5_dilation       = cfg.MODEL.RESNETS.RES5_DILATION
    # fmt: on
    assert res5_dilation in {1, 2}, "res5_dilation cannot be {}.".format(res5_dilation)

    num_blocks_per_stage = {50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3]}[depth]

    stages = []

    # Avoid creating variables without gradients
    # It consumes extra memory and may cause allreduce to fail
    out_stage_idx = [{"res2": 2, "res3": 3, "res4": 4, "res5": 5}[f] for f in out_features]
    max_stage_idx = max(out_stage_idx)
    for idx, stage_idx in enumerate(range(2, max_stage_idx + 1)):
        dilation = res5_dilation if stage_idx == 5 else 1
        first_stride = 1 if idx == 0 or (stage_idx == 5 and dilation == 2) else 2
        stage_kargs = {"num_blocks": num_blocks_per_stage[idx], "first_stride": first_stride,
                       "in_channels": in_channels, "bottleneck_channels": bottleneck_channels,
                       "out_channels": out_channels, "num_groups": num_groups, "norm": norm,
                       "stride_in_1x1": stride_in_1x1, "dilation": dilation, "block_class": BottleneckBlock}
        blocks = make_stage(**stage_kargs)
        in_channels = out_channels
        out_channels *= 2
        bottleneck_channels *= 2

        if freeze_at >= stage_idx:
            for block in blocks:
                block.freeze()
        stages.append(blocks)
    return ResNet(stem, stages, out_features=out_features)
