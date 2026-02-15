"""Type definitions and constants shared across IO components."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class VolumeMetadata:
    """Describe a volume's shape, dtype, estimated size, mean, and std.

    Attributes:
        shape (tuple[int,int,int]): Normalized (Z, Y, X) shape.
        dtype (np.dtype): NumPy dtype for the volume.
        size_gb (float): Estimated in-memory size in GiB.
        mean (float): Mean intensity value of the volume.
        std (float): Standard deviation of intensity values of the volume.
    """

    shape: tuple[int, int, int]
    dtype: np.dtype
    size_gb: float
    mean: float
    std: float

OUTPUT_CHOICES: tuple[str, ...] = (
    "OME-Zarr",
    "Zarr",
    "Tif",
    "Nifti",
    "Scroll-Tif",
    "Scroll-Nifti",
)
"""Human-friendly output selection labels exposed via the CLI."""

TYPE_MAP: dict[str, str] = {
    "OME-Zarr": "ome-zarr",
    "Zarr": "zarr",
    "Tif": "single-tiff",
    "Nifti": "single-nii",
    "Scroll-Tif": "scroll-tiff",
    "Scroll-Nifti": "scroll-nii",
}
"""Map UI labels to the internal writer keys used throughout the pipeline."""

VALID_SUFFIXES: tuple[str, ...] = (
    ".tif",
    ".tiff",
    ".nii",
    ".nii.gz",
    ".gz",
    ".png",
    ".jpg",
)
"""File suffixes that the reader is prepared to handle directly."""

__all__ = [
    "VolumeMetadata",
    "OUTPUT_CHOICES",
    "TYPE_MAP",
    "VALID_SUFFIXES",
]
