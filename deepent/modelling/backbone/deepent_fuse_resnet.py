import fvcore.nn.weight_init as weight_init
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.layers import (
    Conv2d,
    DeformConv,
    FrozenBatchNorm2d,
    ModulatedDeformConv,
    ShapeSpec,
    get_norm,
)
from detectron2.modeling import BACKBONE_REGISTRY
from detectron2.modeling import FPN
from detectron2.modeling.backbone.backbone import Backbone
from detectron2.modeling.backbone.fpn import LastLevelMaxPool
from detectron2.modeling.backbone.resnet import ResNetBlockBase

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


class BottleneckBlock(ResNetBlockBase):
    def __init__(
            self,
            in_channels,
            out_channels,
            *,
            bottleneck_channels,
            stride=1,
            num_groups=1,
            norm="BN",
            stride_in_1x1=False,
            dilation=1,
            fuse_block=False,
            cat_outs=True
    ):
        """
        Args:
            norm (str or callable): a callable that takes the number of
                channels and return a `nn.Module`, or a pre-defined string
                (one of {"FrozenBN", "BN", "GN"}).
            stride_in_1x1 (bool): when stride==2, whether to put stride in the
                first 1x1 convolution or the bottleneck 3x3 convolution.
            :param in_channels: The number of channels for each of the RGB and the D feature maps
            :param out_channels:
            :param bottleneck_channels:
            :param stride:
            :param num_groups:
            :param norm:
            :param stride_in_1x1:
            :param dilation:
            :param fuse_block:
        """
        super().__init__(in_channels, out_channels, stride)
        self.fuse_block = fuse_block
        self.cat_outs = cat_outs
        if in_channels != out_channels:
            self.shortcuts = [Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
                norm=get_norm(norm, out_channels),
            ) for _ in range(2)]
        else:
            self.shortcuts = [None, None]

        # The original MSRA ResNet models have stride in the first 1x1 conv
        # The subsequent fb.torch.resnet and Caffe2 ResNe[X]t implementations have
        # stride in the 3x3 conv
        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)

        self.conv1 = [Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=stride_1x1,
            bias=False,
            norm=get_norm(norm, bottleneck_channels),
        ) for _ in range(2)]

        self.conv2 = [Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride_3x3,
            padding=dilation,
            bias=False,
            groups=num_groups,
            dilation=dilation,
            norm=get_norm(norm, bottleneck_channels),
        ) for _ in range(2)]

        self.conv3 = [Conv2d(
            bottleneck_channels,
            out_channels,
            kernel_size=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        ) for _ in range(2)]

        for layers in [self.conv1, self.conv2, self.conv3, self.shortcuts]:
            for layer in layers:
                if layer is not None:  # shortcut can be None
                    weight_init.c2_msra_fill(layer)

    def forward(self, x):
        # dims of x [batch_size, channels rgb + channels d = 2 * channels rgb, w, h]

        ins = x[:, :self.in_channels, :, :], x[:, self.in_channels:, :, :]
        outs = [None, None]

        for i, (in_, conv1, conv2, conv3, shortcut_fn) in enumerate(
                zip(ins, self.conv1, self.conv2, self.conv3, self.shortcuts)):

            outs[i] = conv1(in_)
            outs[i] = F.relu_(outs[i])

            outs[i] = conv2(outs[i])
            outs[i] = F.relu_(outs[i])

            outs[i] = conv3(outs[i])

            if shortcut_fn is not None:
                shortcut_val = shortcut_fn(in_)
            else:
                shortcut_val = in_

            outs[i] += shortcut_val
            outs[i] = F.relu_(outs[i])

        if self.fuse_block:
            outs[0] += outs[1]

        if self.cat_outs:
            return torch.cat(outs, 1)
        return outs[0]


class DeformBottleneckBlock(ResNetBlockBase):
    def __init__(
            self,
            in_channels,
            out_channels,
            *,
            bottleneck_channels,
            stride=1,
            num_groups=1,
            norm="BN",
            stride_in_1x1=False,
            dilation=1,
            deform_modulated=False,
            deform_num_groups=1,
            fuse_block=False,
            cat_outs=True
    ):
        """
        Similar to :class:`BottleneckBlock`, but with deformable conv in the 3x3 convolution.
        """
        super().__init__(in_channels, out_channels, stride)

        self.deform_modulated = deform_modulated
        self.fuse_block = fuse_block
        self.cat_outs = cat_outs

        if in_channels != out_channels:
            self.shortcuts = [Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
                norm=get_norm(norm, out_channels),
            ) for _ in range(2)]
        else:
            self.shortcuts = [None, None]

        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)

        self.conv1 = [Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=stride_1x1,
            bias=False,
            norm=get_norm(norm, bottleneck_channels),
        ) for _ in range(2)]

        if deform_modulated:
            deform_conv_op = ModulatedDeformConv
            # offset channels are 2 or 3 (if with modulated) * kernel_size * kernel_size
            offset_channels = 27
        else:
            deform_conv_op = DeformConv
            offset_channels = 18

        self.conv2_offset = [Conv2d(
            bottleneck_channels,
            offset_channels * deform_num_groups,
            kernel_size=3,
            stride=stride_3x3,
            padding=1 * dilation,
            dilation=dilation,
        ) for _ in range(2)]

        self.conv2 = [deform_conv_op(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride_3x3,
            padding=1 * dilation,
            bias=False,
            groups=num_groups,
            dilation=dilation,
            deformable_groups=deform_num_groups,
            norm=get_norm(norm, bottleneck_channels),
        ) for _ in range(2)]

        self.conv3 = [Conv2d(
            bottleneck_channels,
            out_channels,
            kernel_size=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        ) for _ in range(2)]

        for layers in [self.conv1, self.conv2, self.conv3, self.shortcuts]:
            for layer in layers:
                if layer is not None:  # shortcut can be None
                    weight_init.c2_msra_fill(layer)

        for conv in self.conv2_offset:
            nn.init.constant_(conv.weight, 0)
            nn.init.constant_(conv.bias, 0)

    def forward(self, x):
        ins = x[:, :self.in_channels, :, :], x[:, self.in_channels:, :, :]
        outs = [None, None]

        for i, (in_, conv1, conv2, conv3, conv2_offset, shortcut_fn) in enumerate(
                zip(ins, self.conv1, self.conv2, self.conv3, self.conv2_offset, self.shortcuts)):

            outs[i] = conv1(in_)
            outs[i] = F.relu_(outs[i])

            if self.deform_modulated:
                offset_mask = conv2_offset(in_)
                offset_x, offset_y, mask = torch.chunk(offset_mask, 3, dim=1)
                offset = torch.cat((offset_x, offset_y), dim=1)
                mask = mask.sigmoid()
                outs[i] = conv2(outs[i], offset, mask)
            else:
                offset = conv2_offset(outs[i])
                outs[i] = conv2(outs[i], offset)
            outs[i] = F.relu_(outs[i])

            outs[i] = conv3(outs[i])

            if shortcut_fn is not None:
                shortcut_val = shortcut_fn(in_)
            else:
                shortcut_val = in_

            outs[i] += shortcut_val
            outs[i] = F.relu_(outs[i])

        if self.fuse_block:
            outs[0] += outs[1]

        if self.cat_outs:
            return torch.cat(outs, 1)
        return outs[0]


def make_stage(block_class, num_blocks, first_stride, **kwargs):
    """
    Create a resnet stage by creating many blocks.
    Args:
        block_class (class): a subclass of ResNetBlockBase
        num_blocks (int):
        first_stride (int): the stride of the first block. The other blocks will have stride=1.
            A `stride` argument will be passed to the block constructor.
        kwargs: other arguments passed to the block constructor.

    Returns:
        list[nn.Module]: a list of block module.
    """
    blocks = []
    for i in range(num_blocks):
        if i == num_blocks - 1:
            blocks.append(block_class(stride=first_stride if i == 0 else 1, fuse_block=True, **kwargs))
        else:
            blocks.append(block_class(stride=first_stride if i == 0 else 1, **kwargs))
        kwargs["in_channels"] = kwargs["out_channels"]
    return blocks


class BasicStem(nn.Module):
    def __init__(self, rgb_channels=3, d_channels=1, out_channels=64, norm="BN"):
        """
        Args:
            norm (str or callable): a callable that takes the number of
                channels and return a `nn.Module`, or a pre-defined string
                (one of {"FrozenBN", "BN", "GN"}).
        """
        super().__init__()
        self.rgb_channels, self.d_channels = rgb_channels, d_channels
        self.out_channels = out_channels

        self.conv1_rgb = Conv2d(
            self.rgb_channels,
            out_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

        self.conv1_d = Conv2d(
            self.d_channels,
            out_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

        for i in (self.conv1_rgb, self.conv1_d):
            weight_init.c2_msra_fill(i)

    def forward(self, x):
        _, channels, _, _ = x.shape
        assert channels == self.rgb_channels + self.d_channels

        in_rgb, in_d = x[:, :self.rgb_channels, :, :], x[:, self.rgb_channels:, :, :]

        # for i, (in_, conv1) in enumerate(zip(ins, self.conv1)):
        #     outs[i] = conv1(in_)
        #     outs[i] = F.relu_(outs[i])
        #     outs[i] = F.max_pool2d(outs[i], kernel_size=3, stride=2, padding=1)

        out_rgb = self.conv1_rgb(in_rgb)
        out_rgb = F.relu_(out_rgb)
        out_rgb = F.max_pool2d(out_rgb, kernel_size=3, stride=2, padding=1)

        out_d = self.conv1_rgb(out_d)
        out_d = F.relu_(out_d)
        out_d = F.max_pool2d(out_d, kernel_size=3, stride=2, padding=1)
        
        return torch.cat((out_rgb, out_d), 1)

    @property
    def stride(self):
        return 4  # = stride 2 conv -> stride 2 max pool


# noinspection PyMethodOverriding,PyUnboundLocalVariable
class ResNet(Backbone):
    def __init__(self, stem: BasicStem, stages, out_features=None):
        """
        Args:
            stem (nn.Module): a stem module
            stages (list[list[ResNetBlock]]): several (typically 4) stages,
                each contains multiple :class:`ResNetBlockBase`.
            out_features (list[str]): name of the layers whose outputs should
                be returned in forward. Can be anything in "stem", "linear", or "res2" ...
                If None, will return the output of the last layer.
        """
        super(ResNet, self).__init__()
        self.stem = stem

        current_stride = self.stem.stride
        self._out_feature_strides = {"stem": current_stride}
        self._out_feature_channels = {"stem": self.stem.out_channels}

        self.stages_and_names = []
        for i, blocks in enumerate(stages):
            for block in blocks:
                assert isinstance(block, ResNetBlockBase), block
                curr_channels = block.out_channels
            stage = nn.Sequential(*blocks)

            name = "res" + str(i + 2)
            self.add_module(name, stage)
            self.stages_and_names.append((stage, name))
            self._out_feature_strides[name] = current_stride = int(
                current_stride * np.prod([k.stride for k in blocks])
            )
            self._out_feature_channels[name] = blocks[-1].out_channels

        if out_features is None:
            out_features = [name]
        self._out_features = out_features
        assert len(self._out_features)
        children = [x[0] for x in self.named_children()]
        for out_feature in self._out_features:
            assert out_feature in children, "Available children: {}".format(", ".join(children))

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        if "stem" in self._out_features:
            outputs["stem"] = x[:, :self.stem.out_channels, :, :]
        for stage, name in self.stages_and_names:
            x = stage(x)
            if name in self._out_features:
                outputs[name] = x[:, :self._out_feature_channels[name], :, :]
        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }


# noinspection PyUnresolvedReferences
def build_deepent_fuse_resnet_backbone(cfg, input_shape):
    """
    Create a ResNet instance from config.

    Returns:
        ResNet: a :class:`ResNet` instance.
    """
    # need registration of new blocks/stems?
    norm = cfg.MODEL.RESNETS.NORM
    stem = BasicStem(
        rgb_channels=input_shape.channels - 1,
        d_channels=1,
        out_channels=cfg.MODEL.RESNETS.STEM_OUT_CHANNELS,
        norm=norm,
    )
    freeze_at = cfg.MODEL.BACKBONE.FREEZE_AT

    if freeze_at >= 1:
        for p in stem.parameters():
            p.requires_grad = False
        stem = FrozenBatchNorm2d.convert_frozen_batchnorm(stem)

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

    num_blocks_per_stage = {50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3]}[depth]

    stages = []

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
        blocks = make_stage(**stage_kargs)
        in_channels = out_channels
        out_channels *= 2
        bottleneck_channels *= 2

        if freeze_at >= stage_idx:
            for block in blocks:
                block.freeze()
        stages.append(blocks)
    return ResNet(stem, stages, out_features=out_features)


# noinspection PyCallingNonCallable
@BACKBONE_REGISTRY.register()
def build_deepent_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    :param cfg: a detectron2 CfgNode
    :param input_shape:
    :returns: backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
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
