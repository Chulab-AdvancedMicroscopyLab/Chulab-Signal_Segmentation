import numpy as np
from skimage.transform import resize
import math
import numba

@numba.njit(fastmath=True)
def _numba_stitch_loop(reconstruction, weight, patches, positions, pd, ph, pw):
    """
    Highly optimized sequential accumulation loop using Numba.
    Sequential is used to avoid race conditions on overlapping pixels.
    """
    for i in range(len(patches)):
        z, y, x = positions[i]
        reconstruction[z:z+pd, y:y+ph, x:x+pw] += patches[i]
        weight[z:z+pd, y:y+ph, x:x+pw] += 1

def stitch_image_xy(patches, positions, original_shape, patch_size, resize_factor=(1, 1, 1)):
    """
    Reconstructs a full 3D volume from overlapping patches using weighted averaging.
    Uses Numba for JIT-accelerated accumulation.
    """
    reconstruction = np.zeros(original_shape, dtype=np.float32)
    weight = np.zeros(original_shape, dtype=np.float32)
    pd, ph, pw = patch_size

    # Ensure positions is a numpy array for Numba
    positions_arr = np.array(positions, dtype=np.int64)
    
    # Handle resizing if necessary (Skimage resize can't be JITted)
    has_resize = any(f != 1.0 for f in resize_factor)
    if has_resize:
        processed_patches = []
        for p in patches:
            p_resized = resize(
                p, (pd, ph, pw),
                order=1, mode='reflect', anti_aliasing=True, preserve_range=True
            ).astype(np.float32)
            processed_patches.append(p_resized)
        patches_to_stitch = np.stack(processed_patches)
    else:
        # Standard case: ensure patches is a contiguous float32 numpy array
        # patches is expected to be (N, pd, ph, pw)
        patches_to_stitch = np.ascontiguousarray(patches, dtype=np.float32)

    # Execute the JIT-optimized loop
    _numba_stitch_loop(reconstruction, weight, patches_to_stitch, positions_arr, pd, ph, pw)

    reconstruction /= np.maximum(weight, 1e-8)
    return reconstruction

def stitch_image_z(reconstruction: np.ndarray, prev_z_slices: np.ndarray, threshold=0.5):
    """
    Blends Z-overlaps and thresholds directly in logit space to avoid slow np.exp().
    """
    if prev_z_slices is not None:
        z_overlay = prev_z_slices.shape[0]
        reconstruction[:z_overlay] = (reconstruction[:z_overlay] + prev_z_slices) / 2

    # Math: Sigmoid(x) > threshold  <=> x > -ln(1/threshold - 1)
    if threshold == 0.5:
        logit_threshold = 0.0
    else:
        logit_threshold = -math.log(1.0 / threshold - 1.0)

    # Threshold directly into uint8 to save memory
    return ((reconstruction > logit_threshold) * 255).astype(np.uint8)

def stitch_image(patches, positions, original_shape, patch_size, resize_factor=(1, 1, 1), prev_z_slices=None, z_overlay=0):
    """
    Reconstructs the full 3D volume from patches and blends overlapping Z slices across chunks.
    """
    # 1. Accumulate XY patches using Numba
    reconstruct_xy = stitch_image_xy(patches, positions, original_shape, patch_size, resize_factor)
    
    # 2. Extract logits for NEXT chunk's overlap BEFORE thresholding
    next_prev_z = None
    if z_overlay > 0:
        next_prev_z = reconstruct_xy[-z_overlay:].copy()

    # 3. Blend Z-overlap and Threshold in logit space
    binary_mask = stitch_image_z(reconstruct_xy, prev_z_slices)
    
    # 4. Return current chunk (minus the part that overlaps with the next)
    if z_overlay > 0:
        return binary_mask[:-z_overlay], next_prev_z
    else:
        return binary_mask, None
