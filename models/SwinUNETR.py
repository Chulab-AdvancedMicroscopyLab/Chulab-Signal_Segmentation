from monai.networks.nets import SwinUNETR as MonaiSwinUNETR
import torch.nn as nn
import torch

class SwinUNETR(nn.Module):
    """
    A wrapper for MONAI's SwinUNETR.
    State-of-the-art Transformer-based encoder for 3D segmentation.
    Note: For MONAI 1.1+, img_size is no longer required.
    """
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        feature_size=48,
        use_checkpoint=False,
        spatial_dims=3
    ):
        super().__init__()
        self.model = MonaiSwinUNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims
        )

    def forward(self, x):
        return self.model(x)
