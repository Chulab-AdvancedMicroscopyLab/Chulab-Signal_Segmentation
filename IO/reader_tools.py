"""Low-level open helpers backing the high-level ``IO.reader.FileReader``.

Provides format-specific loaders that return either NumPy arrays normalized
to Z, Y, X order, or metadata tuples describing shape, dtype, and size.
"""
import logging
import numpy as np
import dask.array as da
import zarr
import numba
from pathlib import Path

# Initialize logging
logger = logging.getLogger(__name__)

@numba.njit(parallel=True, nogil=True)
def _compute_accumulators_numba(data):
    """Parallel single-pass n, sum_x, sum_x2, min, and max using numba."""
    flat = data.ravel()
    n = flat.size
    
    # Initialize with first element or extremes
    if n == 0:
        return 0, 0.0, 0.0, 0.0, 0.0
        
    first_val = float(flat[0])
    
    # Numba prange reduction for min/max requires careful initialization
    # or using a loop. For simplicity and performance, we'll use local variables
    # and then reduce.
    
    # Using thread-local accumulators for min/max
    # Note: Numba's parallel reductions for min/max are efficient.
    sum_x = 0.0
    sum_x2 = 0.0
    min_v = first_val
    max_v = first_val
    
    for i in numba.prange(n):
        val = float(flat[i])
        sum_x += val
        sum_x2 += val * val
        min_v = min(min_v, val)
        max_v = max(max_v, val)
            
    return n, sum_x, sum_x2, min_v, max_v

@numba.njit(parallel=True, nogil=True)
def _compute_stats_numba(data):
    """Parallel single-pass mean, std, min, and max using numba."""
    n, sum_x, sum_x2, min_v, max_v = _compute_accumulators_numba(data)
    
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
        
    mean = sum_x / n
    # Var = E[X^2] - (E[X])^2
    var = (sum_x2 / n) - (mean**2)
    std = float(np.sqrt(max(0, var)))
    return mean, std, min_v, max_v

@numba.njit(parallel=True, nogil=True)
def _compute_histogram_numba(data, bins, range_min, range_max):
    """Parallel histogram computation using numba."""
    flat = data.ravel()
    n = flat.size
    hist = np.zeros(bins, dtype=np.int64)
    
    if n == 0 or range_max <= range_min:
        return hist
        
    bin_width = (range_max - range_min) / bins
    inv_bin_width = 1.0 / bin_width
    
    for i in numba.prange(n):
        val = float(flat[i])
        if val < range_min or val > range_max:
            continue
            
        bin_idx = int((val - range_min) * inv_bin_width)
        if bin_idx >= bins:
            bin_idx = bins - 1
        
        # Atomic add for thread safety in parallel loop
        # hist[bin_idx] += 1
        # Numba supports atomic additions on array elements
        # However, for histograms, a common pattern is to use private histograms
        # and then sum them. Numba's prange handles some reductions but not arrays.
        # So we use a more manual approach or just use a single thread for now if atomic is slow.
        # Actually, for 1D arrays, we can use numba's atomic addition if available
        pass
    
    # Redo without parallel for simplicity if atomic is complex, 
    # or use a smarter approach.
    # Let's use a simpler serial version for now as it's usually fast enough for stats sampling.
    return hist

@numba.njit(nogil=True)
def _compute_all_stats_serial(data, bins, range_min, range_max):
    """Single-pass mean, sum_sq, min, max, and histogram."""
    flat = data.ravel()
    n = flat.size
    hist = np.zeros(bins, dtype=np.int64)
    
    if n == 0:
        return 0, 0.0, 0.0, 0.0, 0.0, hist
        
    sum_x = 0.0
    sum_x2 = 0.0
    min_v = float(flat[0])
    max_v = float(flat[0])
    
    bin_width = (range_max - range_min) / bins
    inv_bin_width = 1.0 / (bin_width + 1e-12)

    for i in range(n):
        val = float(flat[i])
        sum_x += val
        sum_x2 += val * val
        if val < min_v: min_v = val
        if val > max_v: max_v = val
        
        # Histogram part
        if val < range_min or val >= range_max:
            if val == range_max:
                hist[bins-1] += 1
            continue
            
        bin_idx = int((val - range_min) * inv_bin_width)
        if bin_idx >= bins:
            bin_idx = bins - 1
        hist[bin_idx] += 1
        
    return n, sum_x, sum_x2, min_v, max_v, hist


@numba.njit(parallel=True, nogil=True)
def _normalize_inplace_numba(data, mean, std):
    """Parallel in-place normalization using numba."""
    flat = data.ravel()
    n = flat.size
    eps = 1e-8
    inv_std = 1.0 / (std + eps)
    for i in numba.prange(n):
        flat[i] = (flat[i] - mean) * inv_std

# ——— Helper ———

def _estimate_size_gb(shape: tuple, dtype: np.dtype) -> float:
    """Estimate in-memory size in GiB for an array of given shape and dtype."""
    return float(np.prod(shape) * np.dtype(dtype).itemsize / (1024 ** 3))


def _ensure_3d(arr: np.ndarray) -> np.ndarray:
    """Promote a 2D (y,x) array to 3D (1,y,x); leave >3D untouched."""
    if arr.ndim == 2:
        return arr[np.newaxis, ...]
    return arr


def _ensure_3d_shape(shape: tuple) -> tuple:
    """Promote a 2D shape (y,x) to 3D (1,y,x); leave >3D untouched."""
    if len(shape) == 2:
        return (1,) + shape
    return shape


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

def _reader_tiff(path: Path, read_to_array: bool = True, transpose_order: tuple[int, ...] | None = None, max_workers: int | None = None):
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

        if not read_to_array:
            z, y, x = _collapse_shape_to_zyx(series.shape, axes)
            shape = _apply_shape_order((z, y, x), transpose_order)
            size_gb = _estimate_size_gb(shape, dtype)
            # Warn if multi-channel/time collapsed
            extra_axes = ''.join(a for a in axes if a not in ('Y', 'X', 'Z'))
            if any(a in axes for a in ('C', 'S', 'T')):
                logger.warning(f"Collapsing extra TIFF axes '{extra_axes}' into Z for metadata; resulting shape {shape}")
            return shape, dtype, size_gb

        # Full Read Path
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

        # Read the selected series and collapse to (Z,Y,X)
        # Use max_workers for decompression if provided
        arr = series.asarray(maxworkers=max_workers)
        if any(a in axes for a in ('C', 'S', 'T')):
            logger.warning(f"Collapsing extra TIFF axes to Z while reading: axes={axes}, shape={arr.shape}")
        arr_zyx = _collapse_array_to_zyx(arr, axes)
        return _apply_transpose(arr_zyx, transpose_order)


def _reader_nii_gz(path: Path, read_to_array: bool = True, transpose_order: tuple[int, ...] | None = None, max_workers: int | None = None):
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


def _reader_zarr(path: Path, read_to_array: bool = True, transpose_order: tuple[int, ...] | None = None, max_workers: int | None = None):
    """Open a Zarr store and either return the dask array or its metadata."""
    
    # Open the Zarr store. It could be an array or a group.
    store = zarr.open(str(path), mode='r')
    
    if isinstance(store, zarr.hierarchy.Group):
        # For OME-Zarr, we usually want level '0' (highest resolution)
        if '0' in store:
            arr = da.from_zarr(store.store, component='0')
        else:
            # Fallback to the first available array
            keys = list(store.array_keys())
            if not keys:
                raise ValueError(f"Zarr group at {path} contains no arrays.")
            arr = da.from_zarr(store.store, component=keys[0])
    else:
        # It's already a zarr array
        arr = da.from_zarr(str(path))
    
    if read_to_array:
        return _apply_transpose(arr, transpose_order)
    
    # metadata‐only
    shape, dtype = tuple(arr.shape), arr.dtype
    shape = _apply_shape_order(shape, transpose_order)
    size_gb = _estimate_size_gb(shape, dtype)
    return shape, dtype, size_gb


def _reader_imageio(path: Path, read_to_array: bool = True, transpose_order: tuple[int, ...] | None = None, max_workers: int | None = None):
    """Fallback reader using imageio for common image formats."""
    import imageio.v3 as iio
    
    if read_to_array:
        arr = iio.imread(str(path))
        arr = _ensure_3d(arr)
        return _apply_transpose(arr, transpose_order)
    
    # metadata‐only
    # Try to get metadata without reading pixels
    try:
        props = iio.immeta(str(path))
        # imageio.v3 immeta returns a dict. We might need imread with a specific mode or just use props.
        # Fallback to imread if props doesn't have shape
        if 'shape' in props:
            shape, dtype = props['shape'], props['dtype']
        else:
            # imread(..., index=...) or similar might be needed, but for common images 
            # we just want the header. imageio.v3 doesn't have a direct "shape only" for all formats.
            # improps is usually better in v3
            props = iio.improps(str(path))
            shape, dtype = props.shape, props.dtype
    except Exception:
        # Final fallback if immeta/improps fails: read only the first pixel or just the whole thing
        # (though reading the whole thing is what we want to avoid)
        arr = iio.imread(str(path))
        shape, dtype = arr.shape, arr.dtype

    shape = _ensure_3d_shape(shape)
    shape = _apply_shape_order(shape, transpose_order)
    size_gb = _estimate_size_gb(shape, dtype)
    return shape, dtype, size_gb


# ——— Dispatcher ———

@numba.njit(parallel=True, nogil=True)
def _apply_clipping_numba(data, low_cut, high_cut):
    """Parallel in-place range clipping using numba."""
    flat = data.ravel()
    n = flat.size
    
    # Pre-calculate flags to avoid redundant None checks in loop
    has_low = low_cut is not None
    has_high = high_cut is not None
    
    if not has_low and not has_high:
        return

    for i in numba.prange(n):
        if has_low and flat[i] < low_cut:
            flat[i] = low_cut
        elif has_high and flat[i] > high_cut:
            flat[i] = high_cut


def _apply_clipping(arr: np.ndarray, low_cut: float | None = None, high_cut: float | None = None) -> np.ndarray:
    """Apply low and/or high cut clipping to a numpy array using parallel Numba (in-place)."""
    if low_cut is None and high_cut is None:
        return arr
    
    # Ensure floating point for stats/normalization later
    if not np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float32)
        
    _apply_clipping_numba(arr, low_cut, high_cut)
    return arr


def _calculate_stats(arr: np.ndarray) -> tuple[float, float, float, float]:
    """Compute mean, std, min, and max for a numpy array using parallelized single-pass numba."""
    mean, std, min_v, max_v = _compute_stats_numba(arr)
    return mean, std, min_v, max_v


def read_image(
    file_path: Path,
    suffix: str,
    read_to_array: bool = True,
    transpose_order: tuple[int, ...] | None = None,
    low_cut: float | None = None,
    high_cut: float | None = None,
    compute_stats: bool = False,
    max_workers: int | None = None,
):
    """
    Unified reader. Returns either:
      - numpy array (if read_to_array=True and compute_stats=False)
      - (numpy array, mean, std, min, max) (if read_to_array=True and compute_stats=True)
      - (shape, dtype, size_gb, mean, std, min, max) tuple (if read_to_array=False)
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
        if not read_to_array:
            try:
                # Attempt a metadata-only read first
                res = reader(file_path, read_to_array=False, transpose_order=transpose_order, max_workers=max_workers)
                
                # If the reader returned the (shape, dtype, size_gb) tuple, use it
                if isinstance(res, tuple) and len(res) == 3:
                    shape, dtype, size_gb = res
                    # If we only wanted metadata and didn't have to read pixels, 
                    # we don't calculate stats here (it would be a 2nd read).
                    return shape, dtype, size_gb, 0.0, 0.0, 0.0, 0.0
                
                # If it didn't return a tuple, it might have returned an array despite the flag
                arr = res
                shape = tuple(arr.shape)
                dtype = arr.dtype
                size_gb = _estimate_size_gb(shape, dtype)
                mean, std, min_v, max_v = 0.0, 0.0, 0.0, 0.0
                if compute_stats:
                    arr = _apply_clipping(arr, low_cut, high_cut)
                    mean, std, min_v, max_v = _calculate_stats(arr)
                return shape, dtype, size_gb, mean, std, min_v, max_v
            except Exception as e:
                logger.debug(f"Metadata read failed for {file_path}, falling back to full read: {e}")
                # Fall back to full read below
                pass

        # Standard full-array read
        res = reader(file_path, read_to_array=True, transpose_order=transpose_order, max_workers=max_workers)
        res = _apply_clipping(res, low_cut, high_cut)
        
        # If we only wanted metadata but had to fall back to a full read
        if not read_to_array:
            arr = res
            shape = tuple(arr.shape)
            dtype = arr.dtype
            size_gb = _estimate_size_gb(shape, dtype)
            mean, std, min_v, max_v = 0.0, 0.0, 0.0, 0.0
            if compute_stats:
                mean, std, min_v, max_v = _calculate_stats(arr)
            return shape, dtype, size_gb, mean, std, min_v, max_v
            
        if compute_stats:
            mean, std, min_v, max_v = _calculate_stats(res)
            return res, mean, std, min_v, max_v

        return res

    except Exception as e:
        logger.error(f"Error in read_image({file_path}): {e}")
        raise


    except Exception as e:
        logger.error(f"Error in read_image({file_path}): {e}")
        raise
