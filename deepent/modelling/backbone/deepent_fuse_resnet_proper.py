from typing import List, Optional, Set

import numpy as np
from torch import nn

from detectron2.layers import (
    FrozenBatchNorm2d,
    ShapeSpec,
)
from detectron2.modeling import BACKBONE_REGISTRY
from detectron2.modeling import FPN
from detectron2.modeling.backbone.backbone import Backbone
from detectron2.modeling.backbone.fpn import LastLevelMaxPool
from detectron2.modeling.backbone.resnet import ResNetBlockBase, BottleneckBlock, DeformBottleneckBlock, make_stage, \
    BasicStem

"""
An RGB-D backbone for Mask R-CNN built off the detectron2 ResNet backbone 
and incorporating the D channel fusing strategy from FuseNet. 

Basic block architecture:

    RGB -> conv -> conv -> conv + - - > + - - - > (RGB-new) - - - >
     |                          ^       ^
     - - - - > (short cut) - - -|       |                                continue
                                        |
     D -> conv -> conv -> conv + - - - >|- - - > (D-new) - - - - >
     |                          ^
     - - - - > (short cut) - - -|
"""

__all__ = ["build_deepent_fpn_backbone"]


class FuseResNet(Backbone):
    def __init__(self, rgb_stem: BasicStem, d_stem: BasicStem, rgb_stages: List[List[ResNetBlockBase]],
                 d_stages: List[List[ResNetBlockBase]], out_features: Optional[List[str]] = None):
        """
        Args:
            rgb_stem (nn.Module): a stem module for RGB channel input
            d_stem (nn.Module): a stem module for D channel input
            rgb_stages (list[list[ResNetBlock]]): several (typically 4) stages,
                each contains multiple :class:`ResNetBlockBase`. For RGB feature extraction
            d_stages (list[list[ResNetBlock]]): several (typically 4) stages,
                each contains multiple :class:`ResNetBlockBase`. For Depth feature extraction
            out_features (list[str]): name of the layers whose outputs should
                be returned in forward. Can be anything in "stem", or "resRGB2" ... "resRGBn"
                If None, will return the output of the last layer.
        """
        super().__init__()
        self.rgb_stem = rgb_stem
        self.d_stem = d_stem

        current_stride: int = self.rgb_stem.stride
        self._out_feature_strides = {"stem": current_stride}
        self._out_feature_channels = {"stem": self.rgb_stem.out_channels}

        self.rgb_stages_and_names, self.d_stages_and_names = [], []
        for i, blocks in enumerate(rgb_stages):
            stage = nn.Sequential(*blocks)
            # For simplicity we do not change the original names of the resnet blocks in the RGB encoder
            name = "res" + str(i + 2)
            self.add_module(name, stage)
            self.rgb_stages_and_names.append((stage, name))

            # For this model, only the RGB encoder provides output features
            self._out_feature_strides[name] = current_stride = int(
                current_stride * np.prod([k.stride for k in blocks])
            )
            self._out_feature_channels[name] = blocks[-1].out_channels

        # If no output features are specified for the FPN we automatically add the last RGB feature
        if out_features is None:
            # noinspection PyUnboundLocalVariable
            out_features = [name]

        current_stride = self.d_stem.stride
        for i, blocks in enumerate(d_stages):
            stage = nn.Sequential(*blocks)
            name = "res" + str(i + 2) + 'D'
            self.add_module(name, stage)
            self.d_stages_and_names.append((stage, name))

            # Check that output channels match for RGB and D after stage i
            # We remove the last character from the name to get the associated RGB stage name
            assert blocks[-1].out_channels == self._out_feature_channels[
                name[:-1]], f'RGB channels does not match D channels after stage {i}'

            current_stride = int(
                current_stride * np.prod([k.stride for k in blocks])
            )
            rgb_stride = self._out_feature_strides[name[:-1]]

            # Check that stride matches for RGB and D after stage i
            assert current_stride == rgb_stride, \
                f'At stage {i}, RGB stride of {rgb_stride} does not D stride of {current_stride}'

        # Make sure we have at least one output feature
        self._out_features: Set[str] = set(out_features)
        assert len(self._out_features)

        # This seems useless
        children = [x[0] for x in self.named_children()]
        for out_feature in self._out_features:
            assert out_feature in children, "Available children: {}".format(", ".join(children))

    # noinspection PyMethodOverriding
    def forward(self, x):
        assert x.shape[1] == 4, f'Input to FuseResNet should be [4, n, n] not {list(x.shape[1:])}'
        rgb, d = x[:, :3], x[:, 3:]

        outputs = {}
        rgb, d = self.rgb_stem(rgb), self.d_stem(d)
        if "stem" in self._out_features:
            outputs["stem"] = rgb
        for (rgb_stage, rgb_name), (d_stage, _) in zip(self.rgb_stages_and_names, self.d_stages_and_names):
            d = d_stage(d)
            # Fuse !
            rgb = rgb_stage(rgb) + d
            if rgb_name in self._out_features:
                outputs[rgb_name] = rgb

        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }


# noinspection PyUnresolvedReferences
@BACKBONE_REGISTRY.register()
def build_deepent_fuse_resnet_backbone(cfg, input_shape):
    """
    Create a ResNet instance from config.

    Returns:
        ResNet: a :class:`ResNet` instance.
    """
    # need registration of new blocks/stems?
    norm = cfg.MODEL.RESNETS.NORM
    assert input_shape.channels == 4, f'{input_shape.channels} input channels specified, should be 4'
    rgb_stem = BasicStem(
        in_channels=3,
        out_channels=cfg.MODEL.RESNETS.STEM_OUT_CHANNELS,
        norm=norm,
    )

    # Could make a different Depth stem here
    d_stem = BasicStem(
        in_channels=1,
        out_channels=cfg.MODEL.RESNETS.STEM_OUT_CHANNELS,
        norm=norm,
    )
    freeze_at = cfg.MODEL.BACKBONE.FREEZE_AT

    # freeze the RGB and the D stem if freeze_at >= 1
    if freeze_at >= 1:
        for p, q in zip(rgb_stem.parameters(), d_stem.parameters()):
            p.requires_grad = False
            q.requires_grad = False
        rgb_stem = FrozenBatchNorm2d.convert_frozen_batchnorm(rgb_stem)
        d_stem = FrozenBatchNorm2d.convert_frozen_batchnorm(d_stem)

    # fmt: off
    out_features = cfg.MODEL.RESNETS.OUT_FEATURES
    depth = cfg.MODEL.RESNETS.DEPTH
    num_groups = cfg.MODEL.RESNETS.NUM_GROUPS
    width_per_group = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
    bottleneck_channels = num_groups * width_per_group
    in_channels = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
    out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    stride_in_1x1 = cfg.MODEL.RESNETS.STRIDE_IN_1X1
    res5_dilation = cfg.MODEL.RESNETS.RES5_DILATION
    deform_on_per_stage = cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE
    deform_modulated = cfg.MODEL.RESNETS.DEFORM_MODULATED
    deform_num_groups = cfg.MODEL.RESNETS.DEFORM_NUM_GROUPS
    # fmt: on
    assert res5_dilation in {1, 2}, "res5_dilation cannot be {}.".format(res5_dilation)

    # for now we use the same stage architecture for RGB and D encoder
    num_blocks_per_stage = {50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3]}[depth]

    rgb_stages, d_stages = [], []

    # Avoid creating variables without gradients
    # It consumes extra memory and may cause allreduce to fail
    out_stage_idx = [{"res2": 2, "res3": 3, "res4": 4, "res5": 5}[f] for f in out_features]
    max_stage_idx = max(out_stage_idx)
    for idx, stage_idx in enumerate(range(2, max_stage_idx + 1)):
        dilation = res5_dilation if stage_idx == 5 else 1
        first_stride = 1 if idx == 0 or (stage_idx == 5 and dilation == 2) else 2
        stage_kargs = {
            "num_blocks": num_blocks_per_stage[idx],
            "first_stride": first_stride,
            "in_channels": in_channels,
            "bottleneck_channels": bottleneck_channels,
            "out_channels": out_channels,
            "num_groups": num_groups,
            "norm": norm,
            "stride_in_1x1": stride_in_1x1,
            "dilation": dilation,
        }
        if deform_on_per_stage[idx]:
            stage_kargs["block_class"] = DeformBottleneckBlock
            stage_kargs["deform_modulated"] = deform_modulated
            stage_kargs["deform_num_groups"] = deform_num_groups
        else:
            stage_kargs["block_class"] = BottleneckBlock
        rgb_blocks = make_stage(**stage_kargs)
        d_blocks = make_stage(**stage_kargs)
        in_channels = out_channels
        out_channels *= 2
        bottleneck_channels *= 2

        if freeze_at >= stage_idx:
            for b1, b2 in zip(rgb_blocks, d_blocks):
                b1.freeze()
                b2.freeze()
        rgb_stages.append(rgb_blocks)
        d_stages.append(d_blocks)
    return FuseResNet(rgb_stem=rgb_stem, d_stem=d_stem,
                      rgb_stages=rgb_stages, d_stages=d_stages, out_features=out_features)


# noinspection PyCallingNonCallable
@BACKBONE_REGISTRY.register()
def build_deepent_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    :param cfg: a detectron2 CfgNode
    :param input_shape:
    :returns: backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    # noinspection PyTypeChecker
    bottom_up = build_deepent_fuse_resnet_backbone(cfg, input_shape)
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
