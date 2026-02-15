"""Tools for reading volume datasets and normalizing metadata.

High-level stitched volume reader built on top of low-level helpers in
``IO.reader_tools``. This module discovers input files, validates shapes,
and exposes a streaming ``FileReader.read(...)`` API.
"""
import logging
import numpy as np

from pathlib import Path
from itertools import accumulate
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


from .reader_tools import read_image
from .IO_types import VALID_SUFFIXES, VolumeMetadata


# Initialize logging
logger = logging.getLogger(__name__)


def _normalize_transpose_order(transpose_order: tuple[int, ...] | list[int] | None) -> tuple[int, ...] | None:
    """Validate and normalize a transpose tuple into a consistent form."""
    if transpose_order is None:
        return None
    order_tuple = tuple(int(axis) for axis in transpose_order)
    if len(order_tuple) != 3:
        raise ValueError("transpose_order must contain exactly three axes for (Z, Y, X)")
    if sorted(order_tuple) != [0, 1, 2]:
        raise ValueError("transpose_order must be a permutation of 0, 1, 2")
    return order_tuple


def _detect_suffix(path: Path) -> str:
    """Return a normalized suffix string for the given input path."""
    suffixes = [s.lower() for s in path.suffixes]
    if len(suffixes) >= 2 and suffixes[-2:] == ['.nii', '.gz']:
        return '.nii.gz'
    suffix = suffixes[-1] if suffixes else ''
    if suffix and suffix not in VALID_SUFFIXES:
        return ''
    return suffix


def _volume_name_from_path(path: Path, suffix: str) -> str:
    """Derive a readable volume name from the input path and suffix."""
    if suffix == '.nii.gz':
        return path.name[:-len(suffix)]
    return path.stem


def _is_zarr_path(path: Path) -> bool:
    """Return True when the provided path targets a Zarr store."""
    return '.zarr' in str(path)


def _gather_directory_files(directory: Path) -> tuple[list[Path], list[str]]:
    """Scan a directory for sequential image files with matching suffixes."""
    files: list[Path] = []
    types: list[str] = []
    expected_suffix: str | None = None

    for file in sorted(directory.iterdir()):
        if not file.is_file():
            continue

        suffix = _detect_suffix(file)
        if suffix not in VALID_SUFFIXES:
            continue

        if expected_suffix is None:
            expected_suffix = suffix
            files.append(file)
            types.append(suffix)
            continue

        if suffix == expected_suffix:
            files.append(file)
            types.append(suffix)
        else:
            logger.warning(
                f"Files in directory {file} have different suffixes: {suffix} vs {expected_suffix}"
            )

    return files, types


class FileReader:
    """Load volume data from files, directories, or Zarr stores on demand.

    This reader stitches stacks of files or reads single large containers and
    exposes a simple ``read(z/y/x ranges)`` API returning a NumPy array in
    Z, Y, X order. Transpose can be applied to inputs that are stored in a
    different axis order.

    Args:
        input_path (str | Path): File, directory, or Zarr path to read from.
        transpose_order (tuple[int, int, int] | None): Optional axis order to
            apply via ``np.transpose`` to the array/metadata, e.g. ``(1,0,2)``.
        memory_limit_gb (int): Soft cap used to avoid loading too much at once
            during multi-file reads.

    Attributes:
        volume_files (list[Path]): Ordered list of source files.
        volume_types (list[str]): Per-file suffix/type.
        volume_name (str): Base name for the dataset.
        volume_shape (tuple[int,int,int]): Full volume shape (Z, Y, X).
        volume_dtype (np.dtype): Dtype across the stitched volume.
        volume_cumulative_z (list[int]): Cumulative Z extents for each file.
    """

    def __init__(self, input_path, transpose_order=None, memory_limit_gb=32):
        self.input_path = Path(input_path)
        self.transpose_order = _normalize_transpose_order(transpose_order)
        self.memory_limit_bytes = memory_limit_gb * 1024 ** 3

        logger.info(f"Initializing FileReader with path: {self.input_path}")

        self.volume_files: list[Path]
        self.volume_types: list[str]
        self.volume_files, self.volume_types, self.volume_name = self._get_volume_files()
        self.volume_sizes: list[float] = []

        logger.info(f"Found {len(self.volume_files)} volumes")

        self.volume_shape: tuple
        self.volume_dtype: np.dtype
        self.volume_cumulative_z: list[int] = []
        self.volume_mean: float = 0.0
        self.volume_std: float = 0.0

        self._get_volume_info()

        logger.info(f"Volume name: {self.volume_name}")
        logger.info(f"Volume shape: {self.volume_shape}")
        logger.info(f"Volume dtype: {self.volume_dtype}")
        logger.info(f"Volume mean: {self.volume_mean}")
        logger.info(f"Volume std: {self.volume_std}")
        
    def read(self, z_start=0, z_end=None, y_start=0, y_end=None, x_start=0, x_end=None):
        """Load a sub-volume defined by Z/Y/X bounds into memory.

        Args:
            z_start (int): Inclusive starting Z index.
            z_end (int | None): Exclusive ending Z index; defaults to Z size.
            y_start (int): Inclusive starting Y index.
            y_end (int | None): Exclusive ending Y index; defaults to Y size.
            x_start (int): Inclusive starting X index.
            x_end (int | None): Exclusive ending X index; defaults to X size.

        Returns:
            np.ndarray: Array of shape ``(z_end-z_start, y_end-y_start, x_end-x_start)``
            with ``self.volume_dtype``.
        """
        # 1) defaults and clamping
        z0 = max(0, z_start)
        z1 = min(self.volume_shape[0], self.volume_shape[0] if z_end is None else z_end)
        y0 = max(0, y_start)
        y1 = min(self.volume_shape[1], self.volume_shape[1] if y_end is None else y_end)
        x0 = max(0, x_start)
        x1 = min(self.volume_shape[2], self.volume_shape[2] if x_end is None else x_end)

        logger.info(f"Reading volume z: {z0} - {z1}, y: {y0} - {y1} x: {x0} - {x1}")
        dz = z1 - z0
        dy = y1 - y0
        dx = x1 - x0

        if dz <= 0 or dy <= 0 or dx <= 0:
            return np.empty((max(0, dz), max(0, dy), max(0, dx)), dtype=self.volume_dtype)

        # 2) find which files overlap this Z-range
        needed = list(self._iter_needed_files(z0, z1))
        needed_indices = [idx for idx, *_ in needed]

        # 3) memory check
        mem_limit = self.memory_limit_bytes / (1024**3)
        total_to_load = sum(self.volume_sizes[i] for i in needed_indices)
        if total_to_load * 2 > mem_limit:
            raise MemoryError(f"Need {total_to_load*2:.2f}GiB but limit is {mem_limit:.2f}GiB")

        # 4) pre-allocate output
        out = np.empty((dz, dy, dx), dtype=self.volume_dtype)

        # 5) stream each file
        offset = 0
        for idx, base, file_z0, file_z1 in needed:
            length = file_z1 - file_z0

            # load just this file (can use mmap for npy, nibabel, etc)
            arr = read_image(
                self.volume_files[idx],
                self.volume_types[idx],
                True,
                self.transpose_order
            )
            # slice out only [file_z0:file_z1, y0:y1, x0:x1]
            slab = arr[file_z0:file_z1, y0:y1, x0:x1]
            if self.transpose_order is not None and self.volume_types[idx] == ".zarr":
                slab = np.transpose(slab, self.transpose_order)
            out[offset:offset+length, :, :] = slab

            # drop references immediately
            del arr, slab
            offset += length

        return out
    
    def _get_volume_files(self) -> tuple[list[Path], list[str], str]:
        """Collect source files and their suffixes for the input dataset.

        Returns:
            tuple[list[Path], list[str], str]: The files, their normalized
            suffixes, and a derived volume name.
        """
        suffix = _detect_suffix(self.input_path)
        volume_name = _volume_name_from_path(self.input_path, suffix)

        if suffix in VALID_SUFFIXES:
            files = [self.input_path]
            types = [suffix]
        elif _is_zarr_path(self.input_path):
            files = [self.input_path]
            types = [".zarr"]
        elif self.input_path.is_dir():
            files, types = _gather_directory_files(self.input_path)
        else:
            raise ValueError(f"Unsupported file type: {suffix or 'unknown'}")

        if not files or not types:
            raise FileNotFoundError(f"No valid volume files found in {self.input_path}")

        return files, types, volume_name
    
    def _get_volume_info(self):
        """Populate aggregate metadata (shape, dtype, sizes) for the volume.

        Raises:
            RuntimeError: If metadata collection fails for any source file.
            ValueError: If XY shapes or dtypes are inconsistent.
        """
        entries = self._collect_volume_metadata()
        if not entries:
            raise RuntimeError("Failed to collect volume info for all input files")

        self._ensure_consistent_xy(entries)
        self._ensure_consistent_dtype(entries)

        z_lengths = [entry.shape[0] for entry in entries]
        self.volume_cumulative_z = list(accumulate(z_lengths))

        first_shape = entries[0].shape
        self.volume_shape = (self.volume_cumulative_z[-1], first_shape[1], first_shape[2])
        self.volume_dtype = entries[0].dtype
        self.volume_sizes = [entry.size_gb for entry in entries]

        self.volume_mean, self.volume_std = self._calculate_aggregate_stats(entries)

    @staticmethod
    def _calculate_aggregate_stats(entries: list[VolumeMetadata]) -> tuple[float, float]:
        """Calculate weighted mean and pooled standard deviation across all volume entries.

        Args:
            entries: List of metadata objects containing per-file stats.

        Returns:
            tuple[float, float]: The (mean, std) for the combined volume.
        """
        total_pixels = 0
        sum_val = 0.0
        for entry in entries:
            n = np.prod(entry.shape)
            total_pixels += n
            sum_val += entry.mean * n

        if total_pixels == 0:
            return 0.0, 0.0

        mean = sum_val / total_pixels

        # Aggregate variance using the law of total variance:
        # E[X^2] = Var(X) + (E[X])^2
        sum_sq_val = 0.0
        for entry in entries:
            n = np.prod(entry.shape)
            sum_sq_val += (entry.std**2 + entry.mean**2) * n

        mean_sq = sum_sq_val / total_pixels
        std = float(np.sqrt(max(0, mean_sq - mean**2)))

        return mean, std

    def _collect_volume_metadata(self) -> list[VolumeMetadata]:
        """Gather per-file metadata concurrently for the assembled volume.

        Returns:
            list[VolumeMetadata]: Per-file metadata entries including shape,
            dtype, estimated size in GiB, mean, and std.
        """
        metadata: list[VolumeMetadata | None] = [None] * len(self.volume_files)

        def process(file: Path, suffix: str) -> VolumeMetadata:
            shape, dtype, size, mean, std = read_image(
                file,
                suffix,
                read_to_array=False,
                transpose_order=self.transpose_order,
            )
            shape_zyx = tuple(int(dim) for dim in shape)  # normalize to ints
            return VolumeMetadata(
                shape=shape_zyx, 
                dtype=dtype, 
                size_gb=float(size),
                mean=float(mean),
                std=float(std)
            )

        with ThreadPoolExecutor() as executor:
            future_to_idx = {
                executor.submit(process, file, suffix): i
                for i, (file, suffix) in enumerate(zip(self.volume_files, self.volume_types))
            }

            with tqdm(total=len(future_to_idx), desc="Gathering volume info", leave=False) as pbar:
                for future in as_completed(future_to_idx):
                    i = future_to_idx[future]
                    try:
                        metadata[i] = future.result()
                    except Exception as e:
                        file, suffix = self.volume_files[i], self.volume_types[i]
                        raise RuntimeError(f"Error reading file {file.name} with suffix {suffix}: {e}")
                    finally:
                        pbar.update(1)

        if any(entry is None for entry in metadata):
            raise RuntimeError("Failed to collect volume info for all input files")

        return [entry for entry in metadata if entry is not None]

    @staticmethod
    def _ensure_consistent_xy(entries: list[VolumeMetadata]) -> None:
        """Verify that every slice shares the same XY footprint.

        Raises:
            ValueError: When XY dimensions differ across entries.
        """
        shapes_xy = {entry.shape[1:] for entry in entries}
        if len(shapes_xy) > 1:
            raise ValueError(f"Mismatch in XY dimensions across slices: {shapes_xy}")

    @staticmethod
    def _ensure_consistent_dtype(entries: list[VolumeMetadata]) -> None:
        """Ensure the stitched result has a single, consistent dtype.

        Raises:
            ValueError: When multiple dtypes are encountered.
        """
        dtype_set = {entry.dtype for entry in entries}
        if len(dtype_set) > 1:
            raise ValueError(f"Mismatch in data types across volume: {dtype_set}")
    
    def _iter_needed_files(self, z0: int, z1: int):
        """Yield file indices and slice bounds that overlap the requested Z range.

        Args:
            z0 (int): Inclusive Z start in the stitched space.
            z1 (int): Exclusive Z end in the stitched space.

        Yields:
            tuple[int, int, int, int]: ``(file_index, base_z, file_z0, file_z1)``
            where ``base_z`` is the stitched-space start for the file and
            ``file_z0:file_z1`` are the local Z bounds to read from that file.
        """
        prev_cum = [0] + self.volume_cumulative_z[:-1]
        for idx, (cum, prev) in enumerate(zip(self.volume_cumulative_z, prev_cum)):
            if prev >= z1 or cum <= z0:
                continue
            file_z0 = max(0, z0 - prev)
            file_z1 = min(cum - prev, z1 - prev)
            yield idx, prev, file_z0, file_z1
