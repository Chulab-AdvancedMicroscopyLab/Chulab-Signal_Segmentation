"""Low-level open helpers backing the high-level ``IO.reader.FileReader``.

Provides format-specific loaders that return either NumPy arrays normalized
to Z, Y, X order, or metadata tuples describing shape, dtype, and size.
"""
import logging
import numpy as np
from pathlib import Path

# Initialize logging
logger = logging.getLogger(__name__)


# ——— Helper ———

def _estimate_size_gb(shape: tuple, dtype: np.dtype) -> float:
    """Estimate in-memory size in GiB for an array of given shape and dtype."""
    return float(np.prod(shape) * np.dtype(dtype).itemsize / (1024 ** 3))


def _ensure_3d(arr: np.ndarray) -> np.ndarray:
    """Promote a 2D (y,x) array to 3D (1,y,x); leave >3D untouched."""
    if arr.ndim == 2:
        return arr[np.newaxis, ...]
    return arr


def _apply_transpose(arr: np.ndarray, order: tuple[int, ...] | None) -> np.ndarray:
    """Apply a numpy-style axis permutation if provided."""
    if order is None:
        return arr
    return np.transpose(arr, order)


def _apply_shape_order(shape: tuple[int, ...], order: tuple[int, ...] | None) -> tuple[int, ...]:
    """Reorder a shape tuple according to the provided axis order."""
    if order is None:
        return shape
    if len(shape) != len(order):
        raise ValueError(f"Transpose order {order} does not match shape length {len(shape)}")
    return tuple(shape[idx] for idx in order)


# ——— Readers ———

def _reader_tiff(path: Path, read_to_array: bool = True, transpose_order: tuple[int, ...] | None = None):
    """Read TIFF imagery and normalize it to (Z, Y, X) orientation."""
    import tifffile

    def _collapse_shape_to_zyx(shape: tuple, axes: str) -> tuple[int, int, int]:
        axes = axes.upper()
        if 'Y' not in axes or 'X' not in axes:
            raise ValueError(f"TIFF series lacks Y/X axes: axes={axes}, shape={shape}")
        y = int(shape[axes.index('Y')])
        x = int(shape[axes.index('X')])
        z = 1
        for dim, ax in zip(shape, axes):
            if ax not in ('Y', 'X'):
                z *= int(dim)
        return int(z), y, x

    def _collapse_array_to_zyx(arr: np.ndarray, axes: str) -> np.ndarray:
        axes = axes.upper()
        if 'Y' not in axes or 'X' not in axes:
            raise ValueError(f"TIFF series lacks Y/X axes: axes={axes}, shape={arr.shape}")
        # Bring all non-(Y,X) dims first, then Y, X
        non_yx = [i for i, a in enumerate(axes) if a not in ('Y', 'X')]
        permute = non_yx + [axes.index('Y'), axes.index('X')]
        arr_t = np.transpose(arr, permute) if permute else arr
        if arr_t.ndim == 2:
            arr_t = arr_t[np.newaxis, ...]
        else:
            z = int(np.prod(arr_t.shape[:-2])) if arr_t.ndim > 2 else 1
            arr_t = arr_t.reshape((z, arr_t.shape[-2], arr_t.shape[-1]))
        return arr_t

    with tifffile.TiffFile(str(path)) as tf:
        # Choose a series that contains Y and X; prefer the one with largest Y*X
        candidates = [s for s in tf.series if ('Y' in s.axes and 'X' in s.axes)]
        series = max(candidates, key=lambda s: (s.shape[s.axes.index('Y')] * s.shape[s.axes.index('X')], s.size)) if candidates else tf.series[0]
        axes = series.axes
        dtype = series.dtype

        # Simplify: ensure all pages in the chosen series share the same (Y, X)
        xy_dims = set()
        try:
            for page in series.pages:
                ps = getattr(page, 'shape', None)
                if ps is not None and len(ps) >= 2:
                    xy_dims.add((int(ps[-2]), int(ps[-1])))
        except Exception:
            # If tifffile cannot iterate shapes consistently, fall back to series shape
            pass
        if len(xy_dims) > 1:
            raise ValueError(f"Mismatch in XY dimensions across TIFF pages: {xy_dims}")

        if not read_to_array:
            z, y, x = _collapse_shape_to_zyx(series.shape, axes)
            shape = _apply_shape_order((z, y, x), transpose_order)
            size_gb = _estimate_size_gb(shape, dtype)
            # Warn if multi-channel/time collapsed
            extra_axes = ''.join(a for a in axes if a not in ('Y', 'X', 'Z'))
            if any(a in axes for a in ('C', 'S', 'T')):
                logger.warning(f"Collapsing extra TIFF axes '{extra_axes}' into Z for metadata; resulting shape {shape}")
            return shape, dtype, size_gb

        # Read the selected series and collapse to (Z,Y,X)
        arr = series.asarray()
        if any(a in axes for a in ('C', 'S', 'T')):
            logger.warning(f"Collapsing extra TIFF axes to Z while reading: axes={axes}, shape={arr.shape}")
        arr_zyx = _collapse_array_to_zyx(arr, axes)
        return _apply_transpose(arr_zyx, transpose_order)


def _reader_nii_gz(path: Path, read_to_array: bool = True, transpose_order: tuple[int, ...] | None = None):
    """Load NIfTI volumes, optionally returning metadata only."""
    import nibabel as nib
    
    img = nib.load(str(path), mmap=True)
    if read_to_array:
        arr = np.asanyarray(img.dataobj)
        arr = arr[..., np.newaxis] if arr.ndim == 2 else arr
        arr = arr.swapaxes(0, 2)
        return _apply_transpose(arr, transpose_order)
    
    # metadata‐only
    shape = img.shape
    dtype = img.get_data_dtype()
    # normalize to 3D
    if len(shape) == 2:
        shape = (1, shape[1], shape[0])
    elif len(shape) == 3:
        shape = (shape[2], shape[1], shape[0])
    elif len(shape) > 3:
        raise ValueError(f"Unsupported NIfTI shape: {shape}")
    shape = _apply_shape_order(shape, transpose_order)
    size_gb = _estimate_size_gb(shape, dtype)
    return shape, dtype, size_gb


def _reader_zarr(path: Path, read_to_array: bool = True, transpose_order: tuple[int, ...] | None = None):
    """Open a Zarr store and either return the array or its metadata."""
    import zarr
    
    arr = zarr.open(str(path), mode='r')
    
    # If read_to_array is True, return the array
    if read_to_array:
        return arr
    
    # metadata‐only
    shape, dtype = tuple(arr.shape), arr.dtype
    shape = _apply_shape_order(shape, transpose_order)
    return shape, dtype, 0


def _reader_imageio(path: Path, read_to_array: bool = True, transpose_order: tuple[int, ...] | None = None):
    """Fallback reader using imageio for common image formats."""
    import imageio.v3 as iio
    
    if read_to_array:
        arr = iio.imread(str(path))
        arr = _ensure_3d(arr)
        return _apply_transpose(arr, transpose_order)
    
    # metadata‐only
    arr = iio.imread(str(path))
    shape, dtype = arr.shape, arr.dtype
    shape = _apply_shape_order(shape, transpose_order)
    size_gb = _estimate_size_gb(shape, dtype)
    return shape, dtype, size_gb


# ——— Dispatcher ———

def read_image(
    file_path: Path,
    suffix: str,
    read_to_array: bool = True,
    transpose_order: tuple[int, ...] | None = None
):
    """
    Unified reader. Returns either:
      - numpy array (if read_to_array=True)
      - (shape, dtype, size_gb, mean, std) tuple
    """
    
    if suffix in (".tif", ".tiff"):
        reader = _reader_tiff
    elif suffix in (".nii", ".nii.gz", ".gz"):
        reader = _reader_nii_gz
    elif ".zarr" in str(file_path):
        reader = _reader_zarr
    else:
        reader = _reader_imageio

    try:
        result = reader(file_path, read_to_array=read_to_array, transpose_order=transpose_order)
        if not read_to_array:
            # result is (shape, dtype, size_gb)
            # We need to add mean and std. Since we need to calculate them, 
            # we must read the data if not already provided by the reader.
            # Most readers currently don't return them.
            if len(result) == 3:
                shape, dtype, size_gb = result
                # To get mean/std, we actually need to load the data.
                # This might be slow for large files, but it's what's requested.
                arr = reader(file_path, read_to_array=True, transpose_order=transpose_order)
                mean = float(np.mean(arr))
                std = float(np.std(arr))
                return shape, dtype, size_gb, mean, std
        return result
    except Exception as e:
        logger.error(f"Error in read_image({file_path}): {e}")
        raise
