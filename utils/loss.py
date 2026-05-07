import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6, from_logits: bool = True):
        super().__init__()
        self.smooth = smooth
        self.from_logits = from_logits

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.from_logits:
            outputs = torch.sigmoid(outputs)
        
        # Flatten
        outputs = outputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (outputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (outputs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.8, gamma: float = 2.0, from_logits: bool = True):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.from_logits:
            bce_loss = F.binary_cross_entropy_with_logits(outputs, targets, reduction='none')
        else:
            bce_loss = F.binary_cross_entropy(outputs, targets, reduction='none')
        
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * bce_loss
        
        return focal_loss.mean()

class TverskyLoss(nn.Module):
    """
    Tversky Loss handle extreme class imbalance.
    alpha=0.5, beta=0.5 -> Dice Loss
    alpha=1, beta=0 -> Recall optimization
    alpha=0, beta=1 -> Precision optimization
    """
    def __init__(self, alpha: float = 0.5, beta: float = 0.5, smooth: float = 1e-6, from_logits: bool = True):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.from_logits = from_logits

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.from_logits:
            outputs = torch.sigmoid(outputs)
        
        outputs = outputs.view(-1)
        targets = targets.view(-1)
        
        tp = (outputs * targets).sum()
        fp = (outputs * (1 - targets)).sum()
        fn = ((1 - outputs) * targets).sum()
        
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1 - tversky

class LogCoshDiceLoss(nn.Module):
    """Log-Cosh Dice Loss provides smoother gradients than standard Dice."""
    def __init__(self, smooth: float = 1e-6, from_logits: bool = True):
        super().__init__()
        self.dice = DiceLoss(smooth=smooth, from_logits=from_logits)

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dice_loss = self.dice(outputs, targets)
        return torch.log(torch.cosh(dice_loss))

class CombinedLoss(nn.Module):
    def __init__(self, losses: Dict[str, nn.Module], weights: Dict[str, float]):
        super().__init__()
        self.losses = nn.ModuleDict(losses)
        self.weights = weights

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        total_loss = 0.0
        for name, loss_fn in self.losses.items():
            weight = self.weights.get(name, 1.0)
            total_loss += weight * loss_fn(outputs, targets)
        return total_loss

def build_loss_from_config(full_config: dict) -> nn.Module:
    """
    Builds a loss function by decoupling hyperparameters and weights.
    
    Config layout:
    "loss": {
        "dice": {"smooth": 1e-6},
        "focal": {"alpha": 0.8, "gamma": 2.0},
        "tversky": {"alpha": 0.7, "beta": 0.3},
        "log_cosh_dice": {"smooth": 1e-6}
    },
    "train": {
        "loss": {"dice": 0.4, "focal": 0.6}
    }
    """
    loss_params = full_config.get("loss", {})
    train_config = full_config.get("train", {})
    loss_weights = train_config.get("loss", {"dice": 1.0})

    if isinstance(loss_weights, str):
        loss_weights = {loss_weights: 1.0}

    # Registry of available losses
    registry = {
        "dice": lambda cfg: DiceLoss(smooth=cfg.get("smooth", 1e-6)),
        "focal": lambda cfg: FocalLoss(alpha=cfg.get("alpha", 0.8), gamma=cfg.get("gamma", 2.0)),
        "tversky": lambda cfg: TverskyLoss(alpha=cfg.get("alpha", 0.5), beta=cfg.get("beta", 0.5), smooth=cfg.get("smooth", 1e-6)),
        "log_cosh_dice": lambda cfg: LogCoshDiceLoss(smooth=cfg.get("smooth", 1e-6)),
        "bce": lambda _: nn.BCEWithLogitsLoss()
    }

    built_losses = {}
    final_weights = {}

    for name, weight in loss_weights.items():
        name_lower = name.lower()
        if name_lower in registry:
            params = loss_params.get(name_lower, {})
            built_losses[name_lower] = registry[name_lower](params)
            final_weights[name_lower] = weight
        else:
            raise ValueError(f"Unknown loss type requested in train config: {name}")

    if len(built_losses) == 1:
        name = list(built_losses.keys())[0]
        if final_weights[name] == 1.0:
            return built_losses[name]
    
    return CombinedLoss(built_losses, final_weights)
