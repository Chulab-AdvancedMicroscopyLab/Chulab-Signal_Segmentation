import numpy as np
import numba
from typing import List, Tuple, Union
from dataclasses import dataclass

@dataclass(frozen=True)
class PatchSlice:
    """A geometric description of a 3D patch location."""
    z_slice: slice
    y_slice: slice
    x_slice: slice
    global_coords: Tuple[int, int, int]

@numba.njit(fastmath=True, parallel=True)
def _numba_check_mask_patches(mask, starts, dims, threshold):
    """
    Highly optimized mask check using Numba with parallel execution.
    Returns a boolean array where True indicates a positive patch.
    """
    n = starts.shape[0]
    is_positive = np.zeros(n, dtype=np.bool_)
    dz, dy, dx = dims
    
    # Iterate over each patch in parallel
    for i in numba.prange(n):
        zs, ys, xs = starts[i]
        
        found = False
        for z in range(zs, zs + dz):
            for y in range(ys, ys + dy):
                for x in range(xs, xs + dx):
                    if mask[z, y, x] > threshold:
                        found = True
                        break
                if found: break
            if found: break
        is_positive[i] = found
        
    return is_positive

@numba.njit(fastmath=True, parallel=True)
def _numba_extract_patches(array, starts, dims):
    """
    Optimized patch extraction using Numba with parallel execution.
    Returns a single stacked array of shape (N, Z, Y, X).
    """
    n = starts.shape[0]
    dz, dy, dx = dims
    # Pre-allocate the full stack
    patches = np.empty((n, dz, dy, dx), dtype=array.dtype)
    
    for i in numba.prange(n):
        zs, ys, xs = starts[i]
        # Numba handles slicing efficiently
        patches[i] = array[zs:zs+dz, ys:ys+dy, xs:xs+dx]
        
    return patches

def compute_z_plan(volume_depth: int, patch_depth: int, z_overlap: int) -> List[Tuple[int, int]]:
    """
    Calculates a plan for chunking a volume along the Z axis.
    Returns a list of (z_start, actual_overlap) for each chunk.
    """
    assert patch_depth > 0 and z_overlap >= 0
    assert patch_depth > z_overlap

    if volume_depth <= patch_depth:
        return [(0, 0)]

    step = patch_depth - z_overlap
    patches: List[Tuple[int, int]] = []
    z = 0

    while z + patch_depth < volume_depth:
        patches.append((z, z_overlap))
        z += step

    # Handle the final chunk to ensure it reaches the exact end of the volume
    last_start = max(0, volume_depth - patch_depth)
    if not patches or patches[-1][0] != last_start:
        if patches:
            prev_start = patches[-1][0]
            actual_overlap = patch_depth - (last_start - prev_start)
            patches[-1] = (prev_start, actual_overlap)
        patches.append((last_start, 0))

    return patches

def generate_patch_indices(
    volume_shape: Tuple[int, int, int],
    patch_size: Tuple[int, int, int],
    overlap: Tuple[int, int, int],
    z_offset: int = 0
) -> List[PatchSlice]:
    """
    Generate a grid of patch indices covering a volume shape.
    Handles edges by shifting the final patches to align with the volume boundary.
    """
    zd, yh, xw = volume_shape
    pz, py, px = patch_size
    oz, oy, ox = overlap
    
    # Step size (stride)
    sz, sy, sx = pz - oz, py - oy, px - ox
    
    indices = []

    def get_slices(start, patch_dim, full_dim):
        s = start
        e = start + patch_dim
        if e > full_dim:
            e = full_dim
            s = max(0, e - patch_dim)
        return slice(s, e)

    # We use a while loop or range with manual edge handling to ensure full coverage
    for z in range(0, zd, sz):
        z_slc = get_slices(z, pz, zd)
        for y in range(0, yh, sy):
            y_slc = get_slices(y, py, yh)
            for x in range(0, xw, sx):
                x_slc = get_slices(x, px, xw)
                
                indices.append(PatchSlice(
                    z_slice=z_slc,
                    y_slice=y_slc,
                    x_slice=x_slc,
                    global_coords=(z_slc.start + z_offset, y_slc.start, x_slc.start)
                ))
                if x_slc.stop == xw: break
            if y_slc.stop == yh: break
        if z_slc.stop == zd: break
            
    return indices

def filter_indices_by_mask(
    mask: np.ndarray,
    indices: List[PatchSlice],
    neg_keep_ratio: float = 1.0,
    threshold: float = 0.5
) -> List[PatchSlice]:
    """
    Filters patch indices based on mask content (positive vs negative sampling).
    Uses Numba for high-performance mask evaluation.
    """
    if not indices:
        return []

    # Extract start coordinates for Numba
    starts = np.array([
        [idx.z_slice.start, idx.y_slice.start, idx.x_slice.start] 
        for idx in indices
    ], dtype=np.int64)
    
    # Extract dimensions (assuming all patches have same size as the first one)
    # The generate_patch_indices function guarantees this for volumes >= patch_size
    first = indices[0]
    dims = np.array([
        first.z_slice.stop - first.z_slice.start,
        first.y_slice.stop - first.y_slice.start,
        first.x_slice.stop - first.x_slice.start
    ], dtype=np.int64)
    
    # Parallelized/Optimized check
    is_positive = _numba_check_mask_patches(mask, starts, dims, threshold)
    
    pos_indices = []
    neg_indices = []
    
    for i in range(len(indices)):
        if is_positive[i]:
            pos_indices.append(indices[i])
        else:
            neg_indices.append(indices[i])
            
    n_keep_neg = int(len(pos_indices) * neg_keep_ratio)
    if neg_indices:
        # Shuffle negative indices and take a subset
        np.random.shuffle(neg_indices)
        selected_negs = neg_indices[:n_keep_neg]
        result = pos_indices + selected_negs
    else:
        result = pos_indices
        
    # Shuffle the final combined list for training randomization
    np.random.shuffle(result)
    return result

def extract_data_from_indices(
    array: np.ndarray,
    indices: List[PatchSlice],
    as_stack: bool = False
) -> Union[List[np.ndarray], np.ndarray]:
    """
    Extract pixel data from a volume given a list of patch indices.
    Uses Numba for high-performance extraction.
    
    Args:
        array: Input volume (D, H, W)
        indices: List of PatchSlice objects
        as_stack: If True, returns a single numpy array of shape (N, D, H, W).
                  If False, returns a list of numpy arrays.
    """
    if not indices:
        if as_stack:
            return np.empty((0, 0, 0, 0), dtype=array.dtype)
        return []

    # Prepare metadata for Numba
    starts = np.array([
        [idx.z_slice.start, idx.y_slice.start, idx.x_slice.start] 
        for idx in indices
    ], dtype=np.int64)
    
    first = indices[0]
    dims = np.array([
        first.z_slice.stop - first.z_slice.start,
        first.y_slice.stop - first.y_slice.start,
        first.x_slice.stop - first.x_slice.start
    ], dtype=np.int64)
    
    # Fast extraction
    patches = _numba_extract_patches(array, starts, dims)
    
    if as_stack:
        return patches
    
    # Default: return list of arrays (views into the stacked array)
    return [patches[i] for i in range(len(indices))]
