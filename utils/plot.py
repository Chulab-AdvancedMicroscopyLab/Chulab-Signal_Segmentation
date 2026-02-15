import os
import numpy as np
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from typing import Dict, List

def save_learning_curves(
    metrics_history: Dict[str, Dict[str, List[float]]],
    save_path: str,
    model_name: str
):
    """
    Generates and saves learning curve plots for all tracked metrics.
    
    Args:
        metrics_history: Nested dict containing 'train' and 'val' lists for each metric.
        save_path: Directory where the plot image will be saved.
        model_name: Base name of the model, used in the filename.
    """
    metrics_to_plot = [k for k, v in metrics_history.items() if v["train"] and v["val"]]
    if not metrics_to_plot:
        return

    n = len(metrics_to_plot)
    rows = int(np.ceil(n / 2)) if n > 2 else 1
    cols = min(n, 2)
    
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)

    for idx, m in enumerate(metrics_to_plot):
        r, c = idx // cols, idx % cols
        ax = axes[r, c]
        ax.plot(metrics_history[m]["train"], label="Train")
        ax.plot(metrics_history[m]["val"], label="Val")
        ax.set_title(m.replace('_', ' ').capitalize())
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True)

    # Hide unused subplots
    for j in range(n, rows * cols):
        fig.delaxes(axes[j // cols, j % cols])

    plt.tight_layout()
    plot_file = os.path.join(save_path, f"{model_name}-metrics_curve.png")
    plt.savefig(plot_file)
    plt.close(fig)
