import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Optional

def _draw_grid(
    images: np.ndarray,
    masks: np.ndarray,
    predictions: Optional[np.ndarray] = None,
    save_path: str = "./",
    title: str = "visualization",
    max_samples: int = 20,
    cols: int = 4
):
    """
    Core modular function to draw the comparison grid.
    Handles 2D (C, H, W) and 3D (C, D, H, W) data by squeezing channels
    and taking the middle slice of 3D volumes.
    """
    import matplotlib
    matplotlib.use("Agg", force=True)

    n_samples = min(len(images), max_samples)
    if n_samples == 0:
        print(f"Warning: No samples to visualize for {title}")
        return

    rows = int(np.ceil(n_samples / cols))
    
    has_preds = predictions is not None
    per_sample_cols = 3 if has_preds else 2
    
    fig, axes = plt.subplots(
        rows, cols * per_sample_cols, 
        figsize=(4 * cols * per_sample_cols, 4 * rows),
        squeeze=False
    )

    def process_for_plot(arr, slice_idx=None):
        """Squeeze channel and take specified slice if 3D."""
        # Convert torch to numpy if needed
        if hasattr(arr, "detach"):
            arr = arr.detach().cpu().numpy()
            
        # Squeeze leading channel if present (e.g., 1, D, H, W -> D, H, W)
        if arr.ndim > 2 and arr.shape[0] == 1:
            arr = np.squeeze(arr, axis=0)
            
        # If still 3D (D, H, W), take the specified slice
        if arr.ndim == 3:
            if slice_idx is not None:
                # Ensure index is within bounds
                idx = min(max(0, slice_idx), arr.shape[0] - 1)
                arr = arr[idx]
            else:
                arr = arr[arr.shape[0] // 2]
            
        return arr

    for i in range(n_samples):
        r = i // cols
        c = (i % cols) * per_sample_cols
        
        # Determine best slice index from mask (highest sum of signals)
        m = masks[i]
        if hasattr(m, "detach"): m = m.detach().cpu().numpy()
        if m.ndim > 2 and m.shape[0] == 1: m = np.squeeze(m, axis=0)
        
        best_slice_idx = None
        if m.ndim == 3:
            best_slice_idx = int(np.argmax(np.sum(m, axis=(1, 2))))

        # 1. Image
        img = process_for_plot(images[i], best_slice_idx)
        axes[r, c].imshow(img, cmap="gray")
        axes[r, c].set_title(f"S{i} Img (Z:{best_slice_idx if best_slice_idx is not None else 'mid'})")
        axes[r, c].axis("off")
        
        # 2. Mask (GT)
        msk = process_for_plot(masks[i], best_slice_idx)
        axes[r, c+1].imshow(msk, cmap="gray")
        axes[r, c+1].set_title(f"S{i} GT")
        axes[r, c+1].axis("off")
        
        # 3. Prediction (Optional)
        if has_preds:
            pred = process_for_plot(predictions[i], best_slice_idx)
            # Apply sigmoid if it looks like raw logits
            if pred.min() < 0 or pred.max() > 1:
                pred = 1 / (1 + np.exp(-pred)) # sigmoid
            
            axes[r, c+2].imshow(pred > 0.5, cmap="jet")
            axes[r, c+2].set_title(f"S{i} Pred")
            axes[r, c+2].axis("off")

    # Hide unused subplots
    for i in range(n_samples, rows * cols):
        r = i // cols
        c = (i % cols) * per_sample_cols
        for offset in range(per_sample_cols):
            axes[r, c + offset].axis("off")

    plt.tight_layout()
    plt.suptitle(title, fontsize=16)
    plt.subplots_adjust(top=0.92)
    
    full_path = os.path.join(save_path, f"{title}.png")
    plt.savefig(full_path)
    plt.close(fig)

def visualize_dataset(dataset, save_path: str = "./", title: str = "dataset_preview", max_samples: int = 20):
    """
    Displays Image and Mask pairs from a dataset (pre-training).
    """
    images, masks = [], []
    for i in range(min(len(dataset), max_samples)):
        sample = dataset[i]
        if isinstance(sample, dict):
            images.append(sample["image"])
            masks.append(sample["mask"])
        else:
            images.append(sample[0])
            masks.append(sample[1])
            
    _draw_grid(np.array(images), np.array(masks), None, save_path, title, max_samples)

def visualize_predictions(images: np.ndarray, masks: np.ndarray, predictions: Optional[np.ndarray] = None, save_path: str = "./", title: str = "model_results", max_samples: int = 50):
    """
    Displays Image, Mask (GT), and Model Prediction (post-inference).
    If Ground Truth is not available, pass predictions as the second argument.
    """
    _draw_grid(np.array(images), np.array(masks), predictions, save_path, title, max_samples)