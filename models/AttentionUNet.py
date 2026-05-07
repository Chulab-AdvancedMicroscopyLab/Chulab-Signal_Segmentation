from monai.networks.nets import AttentionUnet as MonaiAttentionUnet
import torch.nn as nn
import torch

class AttentionUNet(nn.Module):
    """
    A wrapper for MONAI's AttentionUnet.
    Uses attention gates to focus on relevant spatial areas.
    """
    def __init__(
        self, 
        spatial_dims, 
        in_channels, 
        out_channels, 
        channels=(32, 64, 128, 256, 512), 
        strides=(2, 2, 2, 2),
        dropout=0.2
    ):
        super().__init__()
        self.model = MonaiAttentionUnet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            dropout=dropout
        )

    def forward(self, x):
        return self.model(x)
