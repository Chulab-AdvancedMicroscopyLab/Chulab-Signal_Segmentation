from .UNet import UNet

def build_model_from_config(config):
    """
    Factory function to create models from a configuration dictionary.
    """
    model_type = config.get("model_type", "monai_unet")
    
    if model_type == "monai_unet":
        return UNet(
            spatial_dims=config.get("spatial_dims", 3),
            in_channels=config.get("in_channels", 1),
            out_channels=config.get("out_channels", 1),
            channels=config.get("channels", (32, 64, 128, 256, 512)),
            strides=config.get("strides", (2, 2, 2, 2)),
            num_res_units=config.get("num_res_units", 2),
            dropout=config.get("dropout", 0.1)
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")