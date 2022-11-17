import torch
from mmcv.runner import BaseModule

from .attention_layers import PWConvAttentionLayer
from .util_modules import GroupConvBlock


class PointWiseExtractionBlock(BaseModule):
    def __init__(self,
                 feature_channels=768,
                 num_classes=150,
                 expand_ratio=4,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_cfg=dict(type='LN'),
                 ffn_feature_maps=True):
        super(PointWiseExtractionBlock, self).__init__()
        self.ffn_feature_maps = ffn_feature_maps

        self.attn = PWConvAttentionLayer(dim=feature_channels + num_classes,
                                         attn_drop=attn_drop_rate,
                                         proj_drop=drop_rate,
                                         dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate))

        self.ffn1 = GroupConvBlock(embed_dims=feature_channels + num_classes,
                                   expand_ratio=expand_ratio,
                                   norm_cfg=norm_cfg,
                                   dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate))

    def forward(self, kernels, feature_maps):
        concated = torch.cat((kernels, feature_maps), dim=1)
        concated = self.attn(concated)

        concated = self.ffn1(concated, identity=concated)
        kernels = concated[:, :kernels.shape[1]]
        feature_maps = concated[:, kernels.shape[1]:]

        return kernels, feature_maps
