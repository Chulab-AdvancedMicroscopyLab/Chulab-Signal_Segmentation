import numpy as np
from skimage.transform import resize
import math
import numba


# --------------------------------------------------------------------------
# Weight window
# --------------------------------------------------------------------------

def _make_taper(n: int, ov: int) -> np.ndarray:
    """Create a one-dimensional tapered weight: both ends rise linearly from ~0 to 1, while the middle stays at 1.

    Uses the corresponding component of inference_overlay for ov. The weight
    is clamped to a minimum of 1e-3 to prevent it from becoming 0 when some
    voxels are covered by only one patch.
    """
    w = np.ones(n, dtype=np.float32)
    if ov > 0:
        ov = int(min(ov, n // 2))
        if ov > 0:
            ramp = (np.arange(ov, dtype=np.float32) + 0.5) / ov
            w[:ov] = ramp
            w[-ov:] = ramp[::-1]
    np.maximum(w, 1e-3, out=w)
    return w


def _make_window(patch_size, overlap) -> np.ndarray:
    """Create a separable 3D tapered window with shape equal to patch_size."""
    pd, ph, pw = patch_size
    od, oh, ow = overlap
    return np.ascontiguousarray(
        _make_taper(pd, od)[:, None, None]
        * _make_taper(ph, oh)[None, :, None]
        * _make_taper(pw, ow)[None, None, :],
        dtype=np.float32,
    )


# --------------------------------------------------------------------------
# Numba kernels
# --------------------------------------------------------------------------

@numba.njit(fastmath=True)
def _numba_stitch_loop(reconstruction, weight, patches, positions, window, pd, ph, pw):
    """Accumulate weighted values serially to avoid race conditions in overlaps.

    Unlike the original version, each voxel contribution is multiplied by
    window, and weight accumulates window instead of 1. This replaces the box
    average with a tapered weighted average.
    """
    rd, rh, rw = reconstruction.shape
    for i in range(len(patches)):
        z, y, x = positions[i]

        start_z = max(0, z)
        start_y = max(0, y)
        start_x = max(0, x)

        end_z = min(rd, z + pd)
        end_y = min(rh, y + ph)
        end_x = min(rw, x + pw)

        p_start_z = max(0, -z)
        p_start_y = max(0, -y)
        p_start_x = max(0, -x)

        target_d = end_z - start_z
        target_h = end_y - start_y
        target_w = end_x - start_x

        if target_d > 0 and target_h > 0 and target_w > 0:
            pw_blk = window[p_start_z:p_start_z + target_d,
                            p_start_y:p_start_y + target_h,
                            p_start_x:p_start_x + target_w]
            pv_blk = patches[i][p_start_z:p_start_z + target_d,
                                p_start_y:p_start_y + target_h,
                                p_start_x:p_start_x + target_w]
            reconstruction[start_z:end_z, start_y:end_y, start_x:end_x] += pv_blk * pw_blk
            weight[start_z:end_z, start_y:end_y, start_x:end_x] += pw_blk


@numba.njit(parallel=True, nogil=True)
def _numba_finalize_reconstruction(reconstruction, weight, prev_z_slices, logit_threshold, z_ramp):
    """Normalize by weight, blend across chunk Z overlap, and apply a threshold.

    Z blending uses a linear ramp: the first slice of the overlap almost
    completely uses the previous chunk's value (the current chunk is exactly
    at the patch edge there), and switches completely only at the end of the
    overlap. The original fixed 50/50 blend retained half of the poor
    prediction at the patch edge.
    """
    z, y, x = reconstruction.shape
    binary_out = np.empty((z, y, x), dtype=np.uint8)

    z_overlay = prev_z_slices.shape[0]

    for i in numba.prange(z):
        a = z_ramp[i] if i < z_overlay else np.float32(1.0)
        for j in range(y):
            for k in range(x):
                w = weight[i, j, k]
                val = reconstruction[i, j, k] / max(w, 1e-8)

                if i < z_overlay:
                    val = (1.0 - a) * prev_z_slices[i, j, k] + a * val

                if val > logit_threshold:
                    binary_out[i, j, k] = 255
                else:
                    binary_out[i, j, k] = 0

                reconstruction[i, j, k] = val

    return binary_out


# --------------------------------------------------------------------------
# Public interface
# --------------------------------------------------------------------------

def stitch_image(patches, positions, original_shape, patch_size, resize_factor=(1, 1, 1),
                 prev_z_slices=None, z_overlay=0, threshold=0.5, output_dtype=np.uint8,
                 overlap=None):
    """Reconstruct a 3D volume from patches and blend overlapping Z slices across chunks.

    Args:
        overlap: (oz, oy, ox). When provided, use a tapered weight window;
            when None, fall back to the original equal-weight behavior
            (weight is all 1) for compatibility with existing callers.
    """
    reconstruction = np.zeros(original_shape, dtype=np.float32)
    weight = np.zeros(original_shape, dtype=np.float32)
    pd, ph, pw = patch_size

    # Prepare patches and resize them when necessary.
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

    # Create the weight window.
    if overlap is None:
        window = np.ones((pd, ph, pw), dtype=np.float32)
    else:
        window = _make_window((pd, ph, pw), overlap)

    positions_arr = np.array(positions, dtype=np.int64)
    _numba_stitch_loop(reconstruction, weight, patches_to_stitch, positions_arr, window, pd, ph, pw)

    if threshold == 0.5:
        logit_threshold = 0.0
    else:
        t = max(min(threshold, 0.999), 0.001)
        logit_threshold = -math.log(1.0 / t - 1.0)

    safe_prev = prev_z_slices if prev_z_slices is not None else np.empty((0, 0, 0), dtype=reconstruction.dtype)

    n_prev = safe_prev.shape[0]
    if n_prev > 0:
        z_ramp = ((np.arange(n_prev, dtype=np.float32) + 0.5) / n_prev)
    else:
        z_ramp = np.empty(0, dtype=np.float32)

    binary_mask = _numba_finalize_reconstruction(
        reconstruction, weight, safe_prev, logit_threshold, z_ramp
    )

    if np.issubdtype(output_dtype, np.unsignedinteger):
        max_val = np.iinfo(output_dtype).max
        if max_val != 255:
            binary_mask = (binary_mask.astype(np.float32) / 255.0 * max_val).astype(output_dtype)

    next_prev_z = None
    if z_overlay > 0:
        next_prev_z = reconstruction[-z_overlay:].copy()
        return binary_mask[:-z_overlay], next_prev_z
    else:
        return binary_mask, None