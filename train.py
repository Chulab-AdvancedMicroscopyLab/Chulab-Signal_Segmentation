"""
Train a 2D/3D U-Net-style segmentation model on microscopy data using shared memory.
"""
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

import argparse
import os
import torch
import torch.optim as optim
import json
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional, Callable, Union

from monai.transforms.compose import Compose
from monai.transforms.utility.dictionary import ToTensord
from monai.transforms.spatial.dictionary import RandFlipd
from monai.transforms.intensity.dictionary import (
    GaussianSmoothd, RandAdjustContrastd, RandBiasFieldd, 
    RandShiftIntensityd, RandScaleIntensityd
)
from monai.transforms.post.dictionary import AsDiscreted
from monai.data.dataloader import DataLoader

from IO import load_train_dataset_from_config
from models import build_model_from_config
from utils.visualization import visualize_dataset, visualize_predictions
from utils.metrics import dice_score, bce_score, hard_dice_score
from utils.plot import save_learning_curves

# Initialize logging
logger = logging.getLogger(__name__)

# Metrics Configuration
MetricFn = Callable[[torch.Tensor, torch.Tensor], Union[torch.Tensor, float]]
METRICS_TO_COMPUTE: Dict[str, MetricFn] = {
    "dice_soft": lambda outputs, targets: dice_score(outputs, targets, from_logits=True),
    "dice_hard": lambda outputs, targets: hard_dice_score(outputs, targets, from_logits=True),
    "bce_score": lambda outputs, targets: bce_score(outputs, targets, from_logits=True),
}

# Transforms
train_transform = Compose([
    ToTensord(keys=["image", "mask"], dtype=torch.float32),
    GaussianSmoothd(keys=["mask"], sigma=0.1),
    AsDiscreted(keys=["mask"], threshold=0.5),
    RandFlipd(keys=["image", "mask"], spatial_axis=1, prob=0.5),
    RandAdjustContrastd(keys=["image"], prob=0.3),
    RandBiasFieldd(keys=["image"], prob=0.2),
    RandShiftIntensityd(keys=["image"], offsets=0.2, prob=0.3),
    RandScaleIntensityd(keys=["image"], factors=0.2, prob=0.3),
])

val_transform = Compose([
    ToTensord(keys=["image", "mask"], dtype=torch.float32),
    GaussianSmoothd(keys=["mask"], sigma=0.1),
    AsDiscreted(keys=["mask"], threshold=0.5),
])

def save_checkpoint(model: torch.nn.Module, weight_path: str, name: str):
    """Saves the model checkpoint."""
    path = os.path.join(weight_path, f"{name}.pth")
    torch.save(model, path)
    logger.info(f"[OK] Model saved to {path}")

def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int
) -> Dict[str, float]:
    """Runs a single training epoch."""
    model.train()
    desc = f"Training Epoch {epoch+1}"
    
    metric_sums: Dict[str, float] = {m: 0.0 for m in METRICS_TO_COMPUTE.keys()}
    total_loss = 0.0
    n_batches = max(1, len(loader))

    progress = tqdm(loader, desc=desc, leave=False)
    for images, masks in progress:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss_val = model.get_loss(outputs, masks)
        loss_val.backward()
        optimizer.step()

        batch_loss = float(loss_val.item())
        total_loss += batch_loss
        
        current_metrics = {"loss": batch_loss}
        for name, fn in METRICS_TO_COMPUTE.items():
            try:
                v = fn(outputs, masks)
                val = float(v.item()) if isinstance(v, torch.Tensor) else float(v)
                current_metrics[name] = val
                metric_sums[name] += val
            except Exception: pass

        progress.set_postfix({k: f"{v:.4f}" for k, v in current_metrics.items()})

    results = {"loss": total_loss / n_batches}
    for m in METRICS_TO_COMPUTE.keys():
        results[m] = metric_sums[m] / n_batches
    return results

def valid_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
    viz_path: Optional[str] = None,
    cache_size: int = 20
) -> Dict[str, float]:
    """Runs a single validation epoch and optionally handles visualization."""
    model.eval()
    desc = f"Validating Epoch {epoch+1}"
    
    metric_sums: Dict[str, float] = {m: 0.0 for m in METRICS_TO_COMPUTE.keys()}
    total_loss = 0.0
    n_batches = max(1, len(loader))

    # Visualization caching every 25 epochs
    is_viz_epoch = (epoch + 1) % 25 == 0 and viz_path is not None
    viz_cache = {"images": [], "masks": [], "outputs": []} if is_viz_epoch else None

    with torch.no_grad():
        progress = tqdm(loader, desc=desc, leave=False)
        for images, masks in progress:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            outputs = model(images)
            loss_val = model.get_loss(outputs, masks)

            batch_loss = float(loss_val.item())
            total_loss += batch_loss
            
            current_metrics = {"loss": batch_loss}
            for name, fn in METRICS_TO_COMPUTE.items():
                try:
                    v = fn(outputs, masks)
                    val = float(v.item()) if isinstance(v, torch.Tensor) else float(v)
                    current_metrics[name] = val
                    metric_sums[name] += val
                except Exception: pass
            
            if is_viz_epoch and viz_cache:
                current_count = len(viz_cache["images"])
                if current_count < cache_size:
                    n_to_add = min(cache_size - current_count, images.size(0))
                    viz_cache["images"].extend(images[:n_to_add].detach().cpu().numpy())
                    viz_cache["masks"].extend(masks[:n_to_add].detach().cpu().numpy())
                    viz_cache["outputs"].extend(outputs[:n_to_add].detach().cpu().numpy())

            progress.set_postfix({k: f"{v:.4f}" for k, v in current_metrics.items()})

    if is_viz_epoch and viz_cache and len(viz_cache["images"]) > 0:
        visualize_predictions(
            np.array(viz_cache["images"]),
            np.array(viz_cache["masks"]),
            np.array(viz_cache["outputs"]),
            save_path=viz_path,
            title=f"Epoch_{epoch+1}_Validation"
        )

    results = {"loss": total_loss / n_batches}
    for m in METRICS_TO_COMPUTE.keys():
        results[m] = metric_sums[m] / n_batches
    return results

def main():
    parser = argparse.ArgumentParser(description="Train U-Net for Microscopy Segmentation")
    parser.add_argument("--config", type=str, help="Path to config file")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        full_config = json.load(f)
        
    config = full_config.get("train", {})
    model_config = full_config.get("model", {})
    
    img_root, mask_root = config.get("img_path"), config.get("mask_path")
    data_root = config.get("data_path")
    save_root = config.get("save_path")
    model_name = config.get("model_name", "best_model")
    
    if not (data_root or (img_root and mask_root)) or not save_root:
        logging.error("Missing mandatory paths (data_path or img/mask_path) in config."); return 1
        
    # Paths setup
    model_save_path = os.path.join(save_root, model_name)
    viz_path = os.path.join(model_save_path, "visualization")
    weight_path = os.path.join(model_save_path, "weights")
    artifact_path = os.path.join(model_save_path, "artifacts")
    for p in [viz_path, weight_path, artifact_path]: os.makedirs(p, exist_ok=True)

    log_path = os.path.join(artifact_path, "train.log")
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(file_handler)

    # Archive config and model source
    import shutil
    shutil.copy2(args.config, os.path.join(artifact_path, "config.json"))
    
    model_type = model_config.get("model_type", "monai_unet")
    if model_type == "monai_unet":
        model_src = os.path.join("models", "UNet.py")
        if os.path.exists(model_src):
            shutil.copy2(model_src, os.path.join(artifact_path, "UNet.py"))

    # Dataset & Dataloaders
    train_ds, val_ds = load_train_dataset_from_config(config, train_transform, val_transform)
    
    if config.get("visualize_preview", False):
        visualize_dataset(train_ds, title="train_samples_preview", save_path=viz_path)
        visualize_dataset(val_ds, title="validation_samples_preview", save_path=viz_path)
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=config.get("training_batch_size", 8), 
        shuffle=True, 
        num_workers=config.get("training_num_workers", 4), 
        persistent_workers=True,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=config.get("training_batch_size", 8), 
        shuffle=False, 
        num_workers=config.get("training_num_workers", 4),
        persistent_workers=True,
        pin_memory=True
    )
    
    # Model
    spatial_dims = 3 if config.get("training_patch_size", [1, 64, 64])[0] > 1 else 2
    model_config["spatial_dims"] = spatial_dims
    model = build_model_from_config(model_config)
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)

    # Note: If loading an existing model for fine-tuning, you would use load_checkpoint(path) here.
    
    optimizer = optim.AdamW(model.parameters(), lr=config.get("learning_rate", 1e-4), weight_decay=config.get("weight_decay", 1e-5))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    history: Dict[str, Dict[str, List[float]]] = {n: {"train": [], "val": []} for n in list(METRICS_TO_COMPUTE.keys()) + ["loss"]}
    best_val_loss = float("inf")

    # Training Loop
    logging.info("Starting training...")
    for epoch in range(config.get("training_epochs", 30)):
        print("\n"); logger.info(f"Epoch {epoch + 1}")
        
        train_res = train_epoch(model, train_loader, optimizer, device, epoch)
        val_res = valid_epoch(model, val_loader, device, epoch, viz_path=viz_path)
        
        for m in history.keys():
            history[m]["train"].append(train_res[m]); history[m]["val"].append(val_res[m])
            
        logger.info(f"Loss -> Train: {train_res['loss']:.4f} | Val: {val_res['loss']:.4f}")
        for m in METRICS_TO_COMPUTE.keys():
            logger.info(f"{m.capitalize()} -> Train: {train_res[m]:.4f} | Val: {val_res[m]:.4f}")

        scheduler.step(val_res['loss'])
        if val_res['loss'] < best_val_loss:
            best_val_loss = val_res['loss']; save_checkpoint(model, weight_path, model_name)
        
        if (epoch + 1) % 25 == 0:
            save_checkpoint(model, weight_path, f"{model_name}_epoch_{epoch+1}")
            
        save_learning_curves(history, artifact_path, model_name)

    logging.info("Training complete.")

if __name__ == "__main__":
    main()
