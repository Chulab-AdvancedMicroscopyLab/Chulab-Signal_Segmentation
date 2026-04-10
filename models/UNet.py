from monai.networks.nets import UNet as MonaiUNet
import torch.nn as nn
import torch
import torch.nn.functional as F

class UNet(nn.Module):
    """
    A wrapper for MONAI's UNet.
    Training loss (Soft Dice + BCE) is encapsulated within the model.
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

    def get_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculates a hybrid Focal + Tversky loss to handle extreme class imbalance.
        Tversky loss allows us to weight False Negatives more heavily to improve Recall.
        """
        # 1. Focal Loss (Stable implementation)
        gamma = 2.0
        alpha = 0.25
        
        # Binary Cross Entropy with Logits
        bce = F.binary_cross_entropy_with_logits(pred, target.float(), reduction='none')
        pt = torch.exp(-bce) # probability of the correct class
        focal_loss = (alpha * (1 - pt)**gamma * bce).mean()

        # 2. Tversky Loss (Differentiable)
        # alpha=0.3, beta=0.7 weights False Negatives more (improves Recall)
        # alpha + beta = 1.0 (standard Dice is 0.5/0.5)
        t_alpha = 0.5
        t_beta = 0.5
        
        pred_soft = pred.sigmoid()
        target_soft = target.float()
        
        dims = tuple(range(1, pred_soft.dim()))
        tp = (pred_soft * target_soft).sum(dim=dims)
        fp = (pred_soft * (1 - target_soft)).sum(dim=dims)
        fn = ((1 - pred_soft) * target_soft).sum(dim=dims)
        
        smooth = 1e-5
        tversky_index = (tp + smooth) / (tp + t_alpha * fp + t_beta * fn + smooth)
        tversky_loss = (1.0 - tversky_index).mean()

        # Hybrid weighting (Focus on Tversky for sparse signal)
        return 0.2 * focal_loss + 0.8 * tversky_loss