"""Two-pass volume resizing utilities used by the Zarr/OME-Zarr writers.

The resampling strategy is split into:
  1) XY resize by Z-slab to a temporary Zarr store (parallel per-slice)
  2) Collapse XZ strips from the temp store to reach the final target shape

This keeps memory bounded and enables parallelism while producing a
consistent result for very large inputs.
"""
import logging
import tempfile
import shutil
import zarr
import numpy as np
import dask.array as da

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from skimage.transform import resize
from numcodecs import Blosc

# Set up module-level logger
logger = logging.getLogger(__name__)

def _resize_xy_worker(args):
    """Resize a single XY slice to the requested output dimensions."""
    slice_xy, ty, tx, dt, ord = args
    return resize(
        slice_xy,
        (ty, tx),
        order=ord,
        preserve_range=True,
        anti_aliasing=False,
    ).astype(dt)


def _resize_xz_worker(args):
    """Resize a single XZ slice to the requested output dimensions."""
    slice_xz, tz, tx, dt, ord = args
    return resize(
        slice_xz,
        (tz, tx),
        order=ord,
        preserve_range=True,
        anti_aliasing=False,
    ).astype(dt)


def _collect_resized_slices(executor: ProcessPoolExecutor, tasks, worker, stack_axis: int) -> np.ndarray:
    """Submit resize tasks and restack the results along the chosen axis.

    Args:
        executor (ProcessPoolExecutor): Pool used to process slices.
        tasks (Iterable[tuple]): Sequence of tuples passed to the worker.
        worker (Callable): Function mapping a single task tuple to an array.
        stack_axis (int): Axis index used to stack results.

    Returns:
        np.ndarray: The stacked result.
    """
    resized = list(executor.map(worker, tasks))
    return np.stack(resized, axis=stack_axis)


def write_chunk_to_zarr(array: np.ndarray, chunk_shape: tuple[int, int, int], target, region) -> None:
    """Persist a numpy block into the specified Zarr region using Dask chunking.

    Args:
        array (np.ndarray): Array to write.
        chunk_shape (tuple[int, int, int]): Dask chunk size used for IO.
        target: Zarr array/group dataset to write into.
        region: Region tuple of slices matching the target rank.
    """
    darr = da.from_array(array, chunks=chunk_shape)
    darr.to_zarr(target, region=region)


def _ensure_temp_store(
    temp_store_path: Path | str | None,
    current_shape: tuple[int, int, int] | None,
    target_y: int,
    target_x: int,
    dtype: np.dtype,
    chunk_size: tuple[int, int, int],
):
    """Create or open the temporary zarr store and array used for staging.

    Args:
        temp_store_path (Path | str | None): Existing store path or None to create a temp dir.
        current_shape (tuple[int,int,int] | None): Current (Z,Y,X) shape, required when creating.
        target_y (int): Target Y after XY resizing.
        target_x (int): Target X after XY resizing.
        dtype (np.dtype): Output dtype.
        chunk_size (tuple[int,int,int]): Storage chunk size.

    Returns:
        tuple[Path, zarr.Array]: The ensured temp store path and opened array.
    """
    compressor = Blosc(cname='lz4', clevel=5, shuffle=Blosc.SHUFFLE)

    if temp_store_path is None:
        tmp_root = Path(tempfile.mkdtemp(prefix="chulab_zarr_"))
        temp_store_path = tmp_root / "temp.zarr"
    else:
        temp_store_path = Path(temp_store_path)

    temp_store = zarr.DirectoryStore(temp_store_path)
    try:
        temp_arr = zarr.open_array(store=temp_store, mode='r+')
    except Exception:
        if current_shape is None:
            raise
        temp_arr = zarr.open_array(
            store=temp_store,
            mode='w',
            shape=(current_shape[0], target_y, target_x),
            chunks=chunk_size,
            dtype=dtype,
            compressor=compressor,
        )

    return Path(temp_store_path), temp_arr


def resize_xy_block_to_temp(
    *,
    block: np.ndarray,
    z_range: tuple[int, int],
    target_shape: tuple[int, int, int],
    dtype: np.dtype,
    order: int = 1,
    chunk_size: tuple[int, int, int] = (128, 128, 128),
    temp_store_path: Path | str | None = None,
    current_shape: tuple[int, int, int] | None = None,
) -> Path:
    """Resize a provided Z-slab block in XY and write to the temp store.

    The temp store is created on first call when ``current_shape`` is provided,
    otherwise the existing store at ``temp_store_path`` is reused.

    Args:
        block (np.ndarray): Input slab of shape ``(dz, Y, X)``.
        z_range (tuple[int,int]): Stitched-space Z indices for the slab.
        target_shape (tuple[int,int,int]): Target (Z,Y,X) shape.
        dtype (np.dtype): Output dtype.
        order (int): skimage order for interpolation.
        chunk_size (tuple[int,int,int]): Chunking for temp IO.
        temp_store_path (Path | str | None): Existing temp store path or None.
        current_shape (tuple[int,int,int] | None): Full current shape; required when creating.

    Returns:
        Path: Path to the temp store directory.
    """
    _, target_y, target_x = target_shape
    temp_store_path, temp_arr = _ensure_temp_store(
        temp_store_path, current_shape, target_y, target_x, dtype, chunk_size
    )

    z0, z1 = z_range
    with ProcessPoolExecutor(max_workers=8) as executor:
        tasks = [
            (block[i], target_y, target_x, dtype, order)
            for i in range(block.shape[0])
        ]
        arr = _collect_resized_slices(executor, tasks, _resize_xy_worker, stack_axis=0)
        logger.info(f"Writing volume to temp z: {z0} - {z1}")
        write_chunk_to_zarr(
            arr,
            chunk_size,
            temp_arr,
            (
                slice(z0, z1),
                slice(0, target_y),
                slice(0, target_x),
            ),
        )

    return temp_store_path


def resize_xy_volume_to_temp(
    *,
    input_source,
    current_shape: tuple[int, int, int],
    target_shape: tuple[int, int, int],
    dtype: np.dtype,
    order: int = 1,
    chunk_size: tuple[int, int, int] = (128, 128, 128),
    temp_store_path: Path | str | None = None,
) -> Path:
    """Resize an entire volume in XY, streaming along Z, writing to temp store.

    Args:
        input_source: Zarr array or array-like supporting slicing.
        current_shape (tuple[int,int,int]): Current (Z,Y,X) shape.
        target_shape (tuple[int,int,int]): Target (Z,Y,X) shape.
        dtype (np.dtype): Output dtype.
        order (int): skimage order for interpolation.
        chunk_size (tuple[int,int,int]): Chunking for temp IO.
        temp_store_path (Path | str | None): Optional pre-existing temp store path.

    Returns:
        Path: Path to the temp store directory containing the XY-resized data.
    """
    current_z, _, _ = current_shape
    _, target_y, target_x = target_shape

    temp_store_path, temp_arr = _ensure_temp_store(
        temp_store_path, current_shape, target_y, target_x, dtype, chunk_size
    )

    with ProcessPoolExecutor(max_workers=8) as executor:
        for z0 in range(0, current_z, chunk_size[0]):
            z1 = min(z0 + chunk_size[0], current_z)
            block_chunk = input_source[z0:z1]  # (dz, Y, X)
            tasks = [
                (block_chunk[i], target_y, target_x, dtype, order)
                for i in range(block_chunk.shape[0])
            ]
            arr = _collect_resized_slices(executor, tasks, _resize_xy_worker, stack_axis=0)
            logger.info(f"Writing volume to temp z: {z0} - {z1}")
            write_chunk_to_zarr(
                arr,
                chunk_size,
                temp_arr,
                (
                    slice(z0, z1),
                    slice(0, target_y),
                    slice(0, target_x),
                ),
            )

    return temp_store_path


def resize_xy_to_temp(
    input_source=None,
    current_shape: tuple[int, int, int] | None = None,
    target_shape: tuple[int, int, int] | None = None,
    dtype: np.dtype | None = None,
    *,
    order: int = 1,
    chunk_size: tuple[int, int, int] = (128, 128, 128),
    temp_store_path: Path | str | None = None,
    block: np.ndarray | None = None,
    z_range: tuple[int, int] | None = None,
) -> Path:
    """Backward-compatible wrapper that dispatches to block or volume handlers.

    Either ``block`` with ``z_range`` or ``input_source`` with ``current_shape``
    must be provided.

    Returns:
        Path: Path to the temp store.
    """
    if target_shape is None or dtype is None:
        raise ValueError("target_shape and dtype are required")

    if block is not None:
        if z_range is None:
            raise ValueError("When providing block, z_range=(z0, z1) is required")
        return resize_xy_block_to_temp(
            block=block,
            z_range=z_range,
            target_shape=target_shape,
            dtype=dtype,
            order=order,
            chunk_size=chunk_size,
            temp_store_path=temp_store_path,
            current_shape=current_shape,
        )

    if input_source is None or current_shape is None:
        raise ValueError("When block is None, input_source and current_shape are required")

    return resize_xy_volume_to_temp(
        input_source=input_source,
        current_shape=current_shape,
        target_shape=target_shape,
        dtype=dtype,
        order=order,
        chunk_size=chunk_size,
        temp_store_path=temp_store_path,
    )

def collapse_xz_from_temp(
    temp_store_path: Path | str,
    output_source,
    target_shape: tuple[int, int, int],
    dtype: np.dtype,
    *,
    order: int = 1,
    chunk_size: tuple[int, int, int] = (128, 128, 128),
) -> None:
    """Pass 2: collapse XZ strips from temp store into the destination array.

    Args:
        temp_store_path (Path | str): Path to temp store from the XY pass.
        output_source: Zarr array dataset to receive the final data.
        target_shape (tuple[int,int,int]): Final (Z,Y,X) shape.
        dtype (np.dtype): Output dtype.
        order (int): skimage order for interpolation.
        chunk_size (tuple[int,int,int]): Chunking used for IO.
    """
    target_z, target_y, target_x = target_shape

    temp_store = zarr.DirectoryStore(Path(temp_store_path))
    temp_arr = zarr.open_array(store=temp_store, mode='r')

    with ProcessPoolExecutor(max_workers=8) as executor:
        for y0 in range(0, target_y, chunk_size[1]):
            y1 = min(y0 + chunk_size[1], target_y)
            logger.info(f"Reading volume from temp y: {y0} - {y1}")
            block = temp_arr[:, y0:y1, :]  # (Z, dy, X)
            tasks = [
                (block[:, j, :], target_z, target_x, dtype, order)
                for j in range(block.shape[1])
            ]
            arr = _collect_resized_slices(executor, tasks, _resize_xz_worker, stack_axis=1)
            
            logger.info(f"Writing volume to zarr y: {y0} - {y1}")
            write_chunk_to_zarr(
                arr,
                chunk_size,
                output_source,
                (
                    slice(0, target_z),
                    slice(y0, y1),
                    slice(0, target_x),
                ),
            )

    # Attempt cleanup of the temporary directory created by resize_xy_to_temp
    try:
        tmp_root = Path(temp_store_path).parent
        if tmp_root.name.startswith("chulab_zarr_") and tmp_root.exists():
            shutil.rmtree(tmp_root, ignore_errors=True)
            logger.info("Cleaned temporary store: %s", tmp_root)
    except Exception as e:
        logger.warning("Failed to clean temporary store at %s: %s", temp_store_path, e)
