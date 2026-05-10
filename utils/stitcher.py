import numpy as np
from skimage.transform import resize
import math
import numba

@numba.njit(fastmath=True)
def _numba_stitch_loop(reconstruction, weight, patches, positions, pd, ph, pw):
    """
    Highly optimized sequential accumulation loop using Numba.
    Sequential is used to avoid race conditions on overlapping pixels.
    Handles potential size mismatches and negative offsets due to padding.
    """
    rd, rh, rw = reconstruction.shape
    for i in range(len(patches)):
        z, y, x = positions[i]
        
        # Calculate bounds in reconstruction space
        start_z = max(0, z)
        start_y = max(0, y)
        start_x = max(0, x)
        
        end_z = min(rd, z + pd)
        end_y = min(rh, y + ph)
        end_x = min(rw, x + pw)
        
        # Calculate bounds in patch space (compensating for negative offsets)
        p_start_z = max(0, -z)
        p_start_y = max(0, -y)
        p_start_x = max(0, -x)
        
        target_d = end_z - start_z
        target_h = end_y - start_y
        target_w = end_x - start_x
        
        if target_d > 0 and target_h > 0 and target_w > 0:
            reconstruction[start_z:end_z, start_y:end_y, start_x:end_x] += \
                patches[i][p_start_z:p_start_z+target_d, p_start_y:p_start_y+target_h, p_start_x:p_start_x+target_w]
            weight[start_z:end_z, start_y:end_y, start_x:end_x] += 1

@numba.njit(parallel=True, nogil=True)
def _numba_finalize_reconstruction(reconstruction, weight, prev_z_slices, logit_threshold):
    """
    Fused kernel: Division, Z-blending, and Thresholding.
    Executes in parallel on all CPU cores and releases the GIL.
    """
    z, y, x = reconstruction.shape
    binary_out = np.empty((z, y, x), dtype=np.uint8)
    
    z_overlay = prev_z_slices.shape[0]
    
    for i in numba.prange(z):
        for j in range(y):
            for k in range(x):
                # 1. Finalize XY weighted average
                w = weight[i, j, k]
                val = reconstruction[i, j, k] / max(w, 1e-8)
                
                # 2. Apply Z-blending if in overlap region
                if i < z_overlay:
                    val = (val + prev_z_slices[i, j, k]) / 2.0
                
                # 3. Threshold directly into uint8 (0 or 255)
                if val > logit_threshold:
                    binary_out[i, j, k] = 255
                else:
                    binary_out[i, j, k] = 0
                    
                # 4. Save normalized value back to reconstruction 
                # (so we can extract next_prev_z later)
                reconstruction[i, j, k] = val
                
    return binary_out

def stitch_image(patches, positions, original_shape, patch_size, resize_factor=(1, 1, 1), prev_z_slices=None, z_overlay=0, threshold=0.5, output_dtype=np.uint8):
    """
    Reconstructs the full 3D volume from patches and blends overlapping Z slices across chunks.
    Uses highly parallel Numba kernels for all major computations.
    """
    # 1. Allocate accumulation buffers
    reconstruction = np.zeros(original_shape, dtype=np.float32)
    weight = np.zeros(original_shape, dtype=np.float32)
    pd, ph, pw = patch_size

    # 2. Prepare patches (handling resize if needed)
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
        patches_to_stitch = np.ascontiguousarray(patches, dtype=np.float32)

    # 3. XY Patch Accumulation (Sequential loop, but JIT-accelerated)
    positions_arr = np.array(positions, dtype=np.int64)
    _numba_stitch_loop(reconstruction, weight, patches_to_stitch, positions_arr, pd, ph, pw)

    # 4. Finalize: Division, Z-blend, and Threshold (Fused Parallel Kernel)
    if threshold == 0.5:
        logit_threshold = 0.0
    else:
        # Avoid log(0)
        t = max(min(threshold, 0.999), 0.001)
        logit_threshold = -math.log(1.0 / t - 1.0)
    
    # Numba doesn't like Optional[Array] being None when indexing is involved.
    # We pass a dummy 3D array if prev_z_slices is None.
    safe_prev = prev_z_slices if prev_z_slices is not None else np.empty((0, 0, 0), dtype=reconstruction.dtype)
    
    binary_mask = _numba_finalize_reconstruction(
        reconstruction, weight, safe_prev, logit_threshold
    )
    
    # Scale to output_dtype (e.g., 255 for uint8, 65535 for uint16)
    if np.issubdtype(output_dtype, np.unsignedinteger):
        max_val = np.iinfo(output_dtype).max
        if max_val != 255:
            binary_mask = (binary_mask.astype(np.float32) / 255.0 * max_val).astype(output_dtype)
    
    # 5. Extract logits for NEXT chunk's overlap BEFORE returning
    next_prev_z = None
    if z_overlay > 0:
        next_prev_z = reconstruction[-z_overlay:].copy()
        return binary_mask[:-z_overlay], next_prev_z
    else:
        return binary_mask, None
