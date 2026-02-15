import numpy as np
from typing import List, Tuple
from dataclasses import dataclass

@dataclass(frozen=True)
class PatchSlice:
    """A geometric description of a 3D patch location."""
    z_slice: slice
    y_slice: slice
    x_slice: slice
    global_coords: Tuple[int, int, int]

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
    """
    pos_indices = []
    neg_indices = []
    
    for idx in indices:
        mask_patch = mask[idx.z_slice, idx.y_slice, idx.x_slice]
        if np.any(mask_patch > threshold):
            pos_indices.append(idx)
        else:
            neg_indices.append(idx)
            
    n_keep_neg = int(len(pos_indices) * neg_keep_ratio)
    if neg_indices:
        # Using a fixed seed for reproducible shuffling if needed, or just random
        np.random.shuffle(neg_indices)
        selected_negs = neg_indices[:n_keep_neg]
        result = pos_indices + selected_negs
    else:
        result = pos_indices
        
    # Shuffle the final combined list so training doesn't see all positives then all negatives
    np.random.shuffle(result)
    return result

def extract_data_from_indices(
    array: np.ndarray,
    indices: List[PatchSlice]
) -> List[np.ndarray]:
    """
    Extract pixel data from a volume given a list of patch indices.
    """
    return [array[idx.z_slice, idx.y_slice, idx.x_slice] for idx in indices]
