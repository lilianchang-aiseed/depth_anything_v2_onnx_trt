"""Official Depth Anything V2 metric-depth architecture used by VKITTI.

This module is intentionally separate from the locally modified training model
in :mod:`dpt`.  It is imported lazily by ``train_fisheye_with_test.py`` only
when the optional VKITTI comparison is enabled.
"""

import torch.nn as nn

from .dinov2 import DINOv2
from .dpt import DPTHead


class VKITTIDepthAnythingV2(nn.Module):
    """Official sigmoid metric head with depth in the 0--80 metre range."""

    ARCHITECTURE_ID = 'dav2_metric_vkitti_sigmoid_v1'

    def __init__(
        self,
        encoder='vitl',
        features=256,
        out_channels=(256, 512, 1024, 1024),
        use_bn=False,
        use_clstoken=False,
        max_depth=80.0,
    ):
        super().__init__()
        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11],
            'vitl': [4, 11, 17, 23],
            'vitg': [9, 19, 29, 39],
        }
        self.max_depth = float(max_depth)
        self.encoder = encoder
        self.pretrained = DINOv2(model_name=encoder)
        self.depth_head = DPTHead(
            self.pretrained.embed_dim,
            features,
            use_bn,
            out_channels=list(out_channels),
            use_clstoken=use_clstoken,
        )

    def forward(self, x):
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
        features = self.pretrained.get_intermediate_layers(
            x,
            self.intermediate_layer_idx[self.encoder],
            return_class_token=True,
        )
        depth = self.depth_head(features, patch_h, patch_w) * self.max_depth
        return depth.squeeze(1)

    def get_model_config(self):
        return {
            'architecture_id': self.ARCHITECTURE_ID,
            'model_class': type(self).__name__,
            'model_module': type(self).__module__,
            'encoder': self.encoder,
            'head_type': 'official_metric_sigmoid',
            'output_representation': 'metric_depth_m',
            'output_activation': 'sigmoid',
            'max_depth': self.max_depth,
        }
