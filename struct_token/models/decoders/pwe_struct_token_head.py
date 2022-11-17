import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.utils.weight_init import (constant_init, trunc_normal_,
                                        trunc_normal_init, kaiming_init)
from torch.nn.modules.batchnorm import _BatchNorm

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead

from ..utils import (ResBlock,
                     PointWiseExtractionBlock)


@HEADS.register_module()
class PointWiseExtracStructTokenHead(BaseDecodeHead):
    def __init__(
            self,
            image_h=512,
            image_w=512,
            h_stride=16,
            w_stride=16,
            num_layers=2,
            mlp_ratio=4,
            drop_path_rate=0.1,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            interpolate_mode='bicubic',
            act_cfg=dict(type='GELU'),
            **kwargs):
        super(PointWiseExtracStructTokenHead, self).__init__(act_cfg=act_cfg,
                                                             **kwargs)

        del self.conv_seg
        self.H = image_h // h_stride
        self.W = image_w // w_stride
        self.interpolate_mode = interpolate_mode
        self.has_odd = self.H % 2 != 0 or self.W % 2 != 0
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        self.layers = nn.ModuleList([
            PointWiseExtractionBlock(feature_channels=self.channels,
                                     num_classes=self.num_classes,
                                     expand_ratio=mlp_ratio,
                                     drop_rate=drop_rate,
                                     attn_drop_rate=attn_drop_rate,
                                     drop_path_rate=dpr[i],
                                     ffn_feature_maps=True,
                                     norm_cfg=self.norm_cfg) for i in range(num_layers)])
        self.dec_proj = nn.Conv2d(self.in_channels,
                                  self.channels,
                                  1, 1)

        self.kernels = nn.Parameter(
            torch.randn(1, self.num_classes, self.H, self.W))

        # may use a large depth wise conv in the future
        self.residual_block = ResBlock(self.num_classes, self.num_classes)

    def init_weights(self):
        trunc_normal_(self.kernels, std=0.02)
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=.02, bias=0)
            elif isinstance(m, nn.Conv2d):
                kaiming_init(m, mode='fan_in', bias=0.)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm, nn.LayerNorm)):
                constant_init(m, val=1.0, bias=0.)
        for layer in self.layers:
            layer.init_weights()

    def cls_seg(self, feat):
        raise NotImplementedError

    def forward(self, inputs):
        feature_maps = self._transform_inputs(inputs)
        b, c, h, w = feature_maps.shape
        feature_maps = self.dec_proj(feature_maps)
        if (h, w) != (self.H, self.W):
            kernels = F.interpolate(input=self.kernels,
                                    size=(h, w),
                                    mode=self.interpolate_mode,
                                    align_corners=self.has_odd)
        else:
            kernels = self.kernels
        kernels = kernels.expand(b, -1, -1, -1)
        for layer in self.layers:
            kernels, feature_maps = layer(kernels, feature_maps)
        out = self.residual_block(kernels)
        return out
