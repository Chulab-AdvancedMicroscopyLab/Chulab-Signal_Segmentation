from .UNet import UNet

def get_model(model_type, spatial_dims, in_channels, out_channels, **kwargs):
    """
    Factory function to create models. Currently supporting MONAI UNet.
    """
    if model_type == "monai_unet":
        channels = kwargs.get("channels", (32, 64, 128, 256, 512))
        strides = kwargs.get("strides", (2, 2, 2, 2))
        num_res_units = kwargs.get("num_res_units", 2)
        dropout = kwargs.get("dropout", 0.1)
        
        return UNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
            dropout=dropout
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")