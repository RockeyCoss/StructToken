from functools import partial

import torch.nn as nn
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.runner import BaseModule

from .util_modules import DepthWiseConvBlock


class AttentionLayer(BaseModule):
    def __init__(self,
                 kv_dim=768,
                 query_dim=150,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=None,
                 init_cfg=None):
        super(AttentionLayer, self).__init__(init_cfg)
        self.attn_drop = nn.Dropout(attn_drop)
        self.query_map = DepthWiseConvBlock(query_dim, query_dim)
        self.key_map = DepthWiseConvBlock(kv_dim, kv_dim)
        self.value_map = DepthWiseConvBlock(kv_dim, kv_dim)
        self.out_project = DepthWiseConvBlock(query_dim, query_dim)

        self.proj_drop = nn.Dropout(proj_drop)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else nn.Identity()

    def forward(self, query, key, value):
        """x: B, C, H, W"""
        identity = query
        qb, qc, qh, qw = query.shape
        query = self.query_map(query).flatten(2)
        key = self.key_map(key).flatten(2)
        value = self.value_map(value).flatten(2)

        attn = (query @ key.transpose(-2, -1)) * (query.shape[-1]) ** -0.5
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ value
        x = x.reshape(qb, qc, qh, qw)
        x = self.out_project(x)
        return identity + self.dropout_layer(self.proj_drop(x))


class PWConvAttentionLayer(BaseModule):
    def __init__(self,
                 dim=150,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=None,
                 init_cfg=None):
        super(PWConvAttentionLayer, self).__init__(init_cfg)
        self.query_map = DepthWiseConvBlock(dim, dim)
        self.out_project = DepthWiseConvBlock(dim, dim)
        self.attn_pw_conv = nn.Conv2d(dim, dim, 1, 1)

        self.proj_drop = nn.Dropout(proj_drop)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else nn.Identity()

    def forward(self, query):
        """x: B, C, H, W"""
        identity = query
        query = self.query_map(query)
        x = self.attn_pw_conv(query)
        x = self.out_project(x)
        return identity + self.dropout_layer(self.proj_drop(x))

