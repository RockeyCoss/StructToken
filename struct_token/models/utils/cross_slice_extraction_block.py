from mmcv.runner import BaseModule

from .attention_layers import AttentionLayer
from .util_modules import GroupConvBlock


class CrossSliceExtractionBlock(BaseModule):
    def __init__(self,
                 feature_channels=768,
                 num_classes=150,
                 expand_ratio=4,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_cfg=dict(type='LN'),
                 ffn_feature_maps=True):
        super(CrossSliceExtractionBlock, self).__init__()
        self.ffn_feature_maps = ffn_feature_maps

        self.cross_attn = AttentionLayer(kv_dim=feature_channels,
                                         query_dim=num_classes,
                                         attn_drop=attn_drop_rate,
                                         proj_drop=drop_rate,
                                         dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate))

        if ffn_feature_maps:
            self.ffn2 = GroupConvBlock(embed_dims=feature_channels,
                                       expand_ratio=expand_ratio,
                                       norm_cfg=norm_cfg,
                                       dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate))
        self.ffn1 = GroupConvBlock(embed_dims=num_classes,
                                   expand_ratio=expand_ratio,
                                   norm_cfg=norm_cfg,
                                   dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate))

    def forward(self, kernels, feature_maps):
        kernels = self.cross_attn(query=kernels,
                                  key=feature_maps,
                                  value=feature_maps)

        kernels = self.ffn1(kernels, identity=kernels)

        if self.ffn_feature_maps:
            feature_maps = self.ffn2(feature_maps, identity=feature_maps)

        return kernels, feature_maps
