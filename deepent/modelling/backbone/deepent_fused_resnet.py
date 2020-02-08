import numpy as np

from torch import nn

from detectron2.modeling.backbone.resnet import BottleneckBlock, BasicStem
from detectron2.layers import ShapeSpec, FrozenBatchNorm2d
from detectron2.modeling import ResNet, Backbone, ResNetBlockBase, BACKBONE_REGISTRY, make_stage
from detectron2.modeling.backbone.fpn import LastLevelMaxPool, FPN

from deepent.modelling.backbone.depth_encoder import build_depth_encoder_backbone

__all__ = ["build_deepent_fpn_backbone"]


class FusedResNet(Backbone):
    def __init__(self, stem, stages, num_classes=None, depth_encoder=None, in_features=None, out_features=None):
        """
        Args:
            stem (nn.Module): a stem module
            stages (list[list[ResNetBlock]]): several (typically 4) stages,
                each contains multiple :class:`ResNetBlockBase`.
            num_classes (None or int): if None, will not perform classification.
            out_features (list[str]): name of the layers whose outputs should
                be returned in forward. Can be anything in "stem", "linear", or "res2" ...
                If None, will return the output of the last layer.
        """
        super(FusedResNet, self).__init__()
        #if depth_encoder is not None:
        #    assert isinstance(depth_encoder, Backbone)

        self.depth_encoder = depth_encoder
        self.in_features = in_features
        self.stem = stem
        self.num_classes = num_classes

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

        if num_classes is not None:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.linear = nn.Linear(curr_channels, num_classes)

            # Sec 5.1 in "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour":
            # "The 1000-way fully-connected layer is initialized by
            # drawing weights from a zero-mean Gaussian with standard deviation of 0.01."
            nn.init.normal_(self.linear.weight, std=0.01)
            name = "linear"

        if out_features is None:
            out_features = [name]
        self._out_features = out_features
        assert len(self._out_features)
        children = [x[0] for x in self.named_children()]
        # Ensure depth encoder output feature maps are speced to fuse with Resnet feature maps
        # For now depth encoder is just a resnet, just need to check that its output feature maps are a subset
        for in_feature in self.in_features:
            assert in_feature in children, "Available children: {}".format(", ".join(children))
        for out_feature in self._out_features:
            assert out_feature in children, "Available children: {}".format(", ".join(children))

    def forward(self, x):
        assert x.shape[1] == 4, f'Input to FuseResNet should be [4, n, n] not {list(x.shape[1:])}'
        rgb, d = x[:, :3], x[:, 3:]
        outputs = {}
        depth_encoder_features = self.depth_encoder(d)
        d = [depth_encoder_features[f] for f in self.in_features]
        x = self.stem(rgb)
        if "stem" in self._out_features:
            outputs["stem"] = x
        for stage, name in self.stages_and_names:
            x = stage(x)
            if name in self.in_features:
                x = x + d.pop(0)
            if name in self._out_features:
                outputs[name] = x
        if self.num_classes is not None:
            x = self.avgpool(x)
            x = self.linear(x)
            if "linear" in self._out_features:
                outputs["linear"] = x
        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }


def build_deepent_fused_resnet_backbone(cfg, input_shape):
    """
    Create a ResNet instance from config.

    Returns:
        ResNet: a :class:`ResNet` instance.
    """
    # need registration of new blocks/stems?
    norm = cfg.MODEL.RESNETS.NORM

    assert input_shape.channels == 4, f'{input_shape.channels} input channels specified, should be 4'
    depth_shape = input_shape._replace(channels=1)
    input_shape = input_shape._replace(channels=3)
    depth_encoder = build_depth_encoder_backbone(cfg, depth_shape)

    stem = BasicStem(
        in_channels=input_shape.channels,
        out_channels=cfg.MODEL.RESNETS.STEM_OUT_CHANNELS,
        norm=norm,
    )
    freeze_at = cfg.MODEL.BACKBONE.FREEZE_AT

    if freeze_at >= 1:
        for p in stem.parameters():
            p.requires_grad = False
        stem = FrozenBatchNorm2d.convert_frozen_batchnorm(stem)

    # fmt: off
    in_features = cfg.MODEL.RESNETS.IN_FEATURES
    out_features = cfg.MODEL.RESNETS.OUT_FEATURES
    depth = cfg.MODEL.RESNETS.DEPTH
    num_groups = cfg.MODEL.RESNETS.NUM_GROUPS
    width_per_group = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
    bottleneck_channels = num_groups * width_per_group
    in_channels = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
    out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    stride_in_1x1 = cfg.MODEL.RESNETS.STRIDE_IN_1X1
    res5_dilation = cfg.MODEL.RESNETS.RES5_DILATION
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
    return FusedResNet(stem, stages, depth_encoder=depth_encoder, in_features=in_features, out_features=out_features)


@BACKBONE_REGISTRY.register()
def build_deepent_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    :param cfg: a detectron2 CfgNode
    :param input_shape:
    :returns: backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_deepent_fused_resnet_backbone(cfg, input_shape)
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
