from functools import partial

import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.runner import BaseModule

from .shape_convert import nchw2nlc2nchw


class ResBlock(nn.Module):
    def __init__(self, in_channels=19, channels=19):
        super(ResBlock, self).__init__()
        self.norm1 = nn.LayerNorm(in_channels, eps=1e-6)
        self.act1 = nn.GELU()
        self.conv1 = nn.Conv2d(in_channels,
                               channels,
                               3,
                               1,
                               1)
        self.norm2 = nn.LayerNorm(channels, eps=1e-6)
        self.act2 = nn.GELU()
        self.conv2 = nn.Conv2d(channels,
                               channels,
                               3,
                               1,
                               1)
        if channels != in_channels:
            self.identity_map = nn.Conv2d(in_channels,
                                          channels, 1, 1, 0)
        else:
            self.identity_map = nn.Identity()

    def forward(self, x):
        # refer to paper
        # Identity Mapping in Deep Residual Networks
        out = nchw2nlc2nchw(self.norm1, x)
        out = self.act1(out)
        out = self.conv1(out)
        out = nchw2nlc2nchw(self.norm2, out)
        out = self.act2(out)
        out = self.conv2(out)
        out = out + self.identity_map(x)

        return out


class DepthWiseConvBlock(BaseModule):
    def __init__(self, in_channels, out_channels):
        super(DepthWiseConvBlock, self).__init__()
        mid_channels = in_channels
        self.conv1 = nn.Conv2d(in_channels,
                               mid_channels,
                               1, 1)
        layer_norm = partial(nn.LayerNorm, eps=1e-6)
        self.norm1 = layer_norm(mid_channels)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(mid_channels,
                               mid_channels,
                               3, 1, 1, groups=mid_channels)
        self.norm2 = layer_norm(mid_channels)
        self.act2 = nn.GELU()
        self.conv3 = nn.Conv2d(mid_channels,
                               out_channels,
                               1, 1)
        self.norm3 = layer_norm(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = nchw2nlc2nchw(self.norm1, x)
        x = self.act1(x)

        x = self.conv2(x)
        x = nchw2nlc2nchw(self.norm2, x)
        x = self.act2(x)

        x = self.conv3(x)
        x = nchw2nlc2nchw(self.norm3, x)
        return x


class GroupConvBlock(BaseModule):
    def __init__(self,
                 embed_dims=150,
                 expand_ratio=6,
                 norm_cfg=dict(type='LN', eps=1e-6, requires_grad=True),
                 dropout_layer=None,
                 init_cfg=None):
        super(GroupConvBlock, self).__init__(init_cfg)
        self.pwconv1 = nn.Conv2d(embed_dims,
                                 embed_dims * expand_ratio,
                                 1, 1)
        self.norm1 = build_norm_layer(norm_cfg,
                                      embed_dims * expand_ratio)[1]
        self.act1 = nn.GELU()
        self.dwconv = nn.Conv2d(embed_dims * expand_ratio,
                                embed_dims * expand_ratio,
                                3, 1, 1, groups=embed_dims)
        self.norm2 = build_norm_layer(norm_cfg,
                                      embed_dims * expand_ratio)[1]
        self.act2 = nn.GELU()
        self.pwconv2 = nn.Conv2d(embed_dims * expand_ratio,
                                 embed_dims,
                                 1, 1)
        self.norm3 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.final_act = nn.GELU()
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else torch.nn.Identity()

    def forward(self, x, identity=None):
        input = x
        x = self.pwconv1(x)
        x = nchw2nlc2nchw(self.norm1, x)
        x = self.act1(x)

        x = self.dwconv(x)
        x = nchw2nlc2nchw(self.norm2, x)
        x = self.act2(x)

        x = self.pwconv2(x)
        x = nchw2nlc2nchw(self.norm3, x)

        if identity is None:
            x = input + self.dropout_layer(x)
        else:
            x = identity + self.dropout_layer(x)

        x = self.final_act(x)

        return x
