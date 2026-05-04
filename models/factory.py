from .UNet import UNet
from .AttentionUNet import AttentionUNet
from .SwinUNETR import SwinUNETR
from .VNet import VNet

def build_model_from_config(config):
    """
    Factory function to create models from a configuration dictionary.
    
    If config is the full config, it extracts model_type from the 'train' section
    and parameters from the 'model' section. Otherwise, it assumes model_type
    is directly in the config.
    """
    if "train" in config and "model" in config:
        # Full config provided
        model_type = config.get("train", {}).get("model_type", "unet")
        model_params = config.get("model", {}).get(model_type, {})
    else:
        # Sub-config or specific params provided
        model_type = config.get("model_type", "unet")
        model_params = config

    if model_type == "unet":
        return UNet(
            spatial_dims=model_params.get("spatial_dims", 3),
            in_channels=model_params.get("in_channels", 1),
            out_channels=model_params.get("out_channels", 1),
            channels=model_params.get("channels", (32, 64, 128, 256, 512)),
            strides=model_params.get("strides", (2, 2, 2, 2)),
            num_res_units=model_params.get("num_res_units", 2),
            dropout=model_params.get("dropout", 0.1)
        )
    
    elif model_type == "attention_unet":
        return AttentionUNet(
            spatial_dims=model_params.get("spatial_dims", 3),
            in_channels=model_params.get("in_channels", 1),
            out_channels=model_params.get("out_channels", 1),
            channels=model_params.get("channels", (32, 64, 128, 256, 512)),
            strides=model_params.get("strides", (2, 2, 2, 2)),
            dropout=model_params.get("dropout", 0.1)
        )
        
    elif model_type == "swin_unetr":
        return SwinUNETR(
            in_channels=model_params.get("in_channels", 1),
            out_channels=model_params.get("out_channels", 1),
            feature_size=model_params.get("feature_size", 48),
            use_checkpoint=model_params.get("use_checkpoint", False),
            spatial_dims=model_params.get("spatial_dims", 3)
        )
    
    elif model_type == "vnet":
        return VNet(
            spatial_dims=model_params.get("spatial_dims", 3),
            in_channels=model_params.get("in_channels", 1),
            out_channels=model_params.get("out_channels", 1),
            dropout_prob=model_params.get("dropout_prob", 0.5),
        )
        
    else:
        raise ValueError(f"Unknown model_type: {model_type}")