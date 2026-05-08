from monai.networks.nets import UNet as MonaiUNet
import torch.nn as nn
import torch

class UNet(nn.Module):
    """
    A wrapper for MONAI's UNet.
    """
    def __init__(
        self, 
        spatial_dims, 
        in_channels, 
        out_channels, 
        channels=(32, 64, 128, 256, 512), 
        strides=(2, 2, 2, 2),
        num_res_units=2,
        dropout=0.2
    ):
        super().__init__()
        self.model = MonaiUNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
            dropout=dropout
        )

    def forward(self, x):
        return self.model(x)
