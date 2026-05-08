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


from .reader_tools import read_image, _compute_accumulators_numba, _normalize_inplace_numba
from .IO_types import VALID_SUFFIXES, VolumeMetadata


# Initialize logging
logger = logging.getLogger(__name__)

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

    def __init__(self, input_path, memory_limit_gb=32, io_workers=4, compute_stats=False, stats_sample_rate=1.0, low_cut=None, high_cut=None, compute_histogram=False, hist_bins=256):
        self.input_path = Path(input_path)
        self.memory_limit_bytes = memory_limit_gb * 1024 ** 3
        self.io_workers = io_workers
        self.compute_stats = compute_stats
        self.stats_sample_rate = float(stats_sample_rate)
        self.low_cut = low_cut
        self.high_cut = high_cut
        self.compute_histogram = compute_histogram
        self.hist_bins = hist_bins

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
        self.volume_min: float = 0.0
        self.volume_max: float = 0.0
        self.volume_histogram: np.ndarray | None = None

        self._get_volume_info()

        logger.info(f"Volume name: {self.volume_name}")
        logger.info(f"Volume shape: {self.volume_shape}")
        logger.info(f"Volume dtype: {self.volume_dtype}")
        logger.info(f"Volume mean: {self.volume_mean}")
        logger.info(f"Volume std: {self.volume_std}")
        logger.info(f"Volume min: {self.volume_min}")
        logger.info(f"Volume max: {self.volume_max}")
        if self.volume_histogram is not None:
            logger.info(f"Volume histogram computed with {self.hist_bins} bins")
        
    def read(self, z_start=0, z_end=None, y_start=0, y_end=None, x_start=0, x_end=None, out_dtype=None):
        """Load a sub-volume defined by Z/Y/X bounds into memory."""
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

        # Determine target dtype
        target_dtype = np.dtype(out_dtype) if out_dtype is not None else self.volume_dtype

        if dz <= 0 or dy <= 0 or dx <= 0:
            return np.empty((max(0, dz), max(0, dy), max(0, dx)), dtype=target_dtype)

        # 2) find which files overlap this Z-range
        needed = list(self._iter_needed_files(z0, z1))
        needed_indices = [idx for idx, *_ in needed]

        # 3) memory check
        mem_limit = self.memory_limit_bytes / (1024**3)
        total_to_load = sum(self.volume_sizes[i] for i in needed_indices)
        is_zarr = len(self.volume_types) > 0 and self.volume_types[0] == '.zarr'
        
        if (total_to_load * 2 > mem_limit) and not is_zarr:
            raise MemoryError(f"Need {total_to_load*2:.2f}GiB but limit is {mem_limit:.2f}GiB")

        # 4) pre-allocate output with final target dtype
        out = np.empty((dz, dy, dx), dtype=target_dtype)

        # 5) stream files (Sequential for Zarr, Multithreaded for others)
        if is_zarr:
            # Sequential loading for Zarr
            offset = 0
            for idx, _, file_z0, file_z1 in needed:
                length = file_z1 - file_z0
                arr = read_image(
                    self.volume_files[idx],
                    self.volume_types[idx],
                    read_to_array=True,
                    max_workers=self.io_workers,
                    low_cut=self.low_cut,
                    high_cut=self.high_cut
                )
                slab = arr[file_z0:file_z1, y0:y1, x0:x1]
                # In-place copy handles conversion if dtypes differ
                out[offset:offset+length, :, :] = slab
                del arr, slab
                offset += length
        else:
            # Multithreaded loading for standard files (TIFF, PNG, etc.)
            num_needed = len(needed)
            def _load_task(task):
                idx, f_z0, f_z1, out_offset, length = task
                dec_workers = 1 if num_needed > 1 else self.io_workers
                
                arr = read_image(
                    self.volume_files[idx],
                    self.volume_types[idx],
                    read_to_array=True,
                    max_workers=dec_workers,
                    low_cut=self.low_cut,
                    high_cut=self.high_cut
                )
                # In-place copy handles conversion if dtypes differ
                out[out_offset:out_offset+length, :, :] = arr[f_z0:f_z1, y0:y1, x0:x1]

            task_list = []
            current_offset = 0
            for idx, base_z, f_z0, f_z1 in needed:
                length = f_z1 - f_z0
                task_list.append((idx, f_z0, f_z1, current_offset, length))
                current_offset += length

            with ThreadPoolExecutor(max_workers=self.io_workers) as executor:
                list(executor.map(_load_task, task_list))

        return out
    
    def normalize_inplace(self, data: np.ndarray) -> np.ndarray:
        """Apply global volume statistics to normalize the provided array in-place.
        
        Uses parallel Numba for fast execution and minimal memory overhead.
        """
        _normalize_inplace_numba(data, self.volume_mean, self.volume_std)
        return data

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
        """Populate aggregate metadata for the volume.

        Raises:
            RuntimeError: If metadata collection fails for any source file.
            ValueError: If XY shapes or dtypes are inconsistent.
        """
        # Pass 1: Basic Metadata (Shape and Dtype only)
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

        # Pass 2: Unified Global Data Pass
        # We always do a fresh read pass if stats are requested to ensure 
        # perfect uniformity in sampling, clipping, and calculation.
        if self.compute_stats:
            self._compute_global_data()
        else:
            self.volume_mean, self.volume_std = 0.0, 0.0
            self.volume_min, self.volume_max = 0.0, 0.0

    def _compute_global_data(self):
        """Unified pass to calculate mean, std, min, max and optionally histogram."""
        slice_size_bytes = self.volume_shape[1] * self.volume_shape[2] * self.volume_dtype.itemsize
        target_chunk_bytes = self.memory_limit_bytes * 0.1
        chunk_size = max(1, int(target_chunk_bytes / slice_size_bytes))
        chunk_size = min(chunk_size, 128)
        
        step = max(1, int(1.0 / self.stats_sample_rate)) if self.stats_sample_rate < 1.0 else 1
        
        if self.compute_histogram:
            logger.info(f"Computing global volume stats and histogram ({self.hist_bins} bins) in chunks of {chunk_size} Z-slices...")
        else:
            logger.info(f"Computing global volume stats in chunks of {chunk_size} Z-slices...")

        all_z_ranges = []
        for z0 in range(0, self.volume_shape[0], chunk_size):
            z1 = min(z0 + chunk_size, self.volume_shape[0])
            all_z_ranges.append((z0, z1))
        
        z_ranges = all_z_ranges[::step]

        from .reader_tools import _compute_accumulators_numba, _compute_all_stats_serial

        # If we need a histogram, we need a preliminary range for the bins
        # We'll use the low_cut/high_cut if provided, otherwise assume data type limits
        r_min = self.low_cut if self.low_cut is not None else 0.0
        r_max = self.high_cut if self.high_cut is not None else (65535.0 if self.volume_dtype == np.uint16 else 255.0)

        def _process_chunk(z_range):
            data = self.read(z_start=z_range[0], z_end=z_range[1])
            if data.size == 0:
                return 0, 0.0, 0.0, 0.0, 0.0, np.zeros(self.hist_bins, dtype=np.int64)
            
            if self.compute_histogram:
                res = _compute_all_stats_serial(data, self.hist_bins, r_min, r_max)
            else:
                n, sx, sx2, v_min, v_max = _compute_accumulators_numba(data)
                res = (n, sx, sx2, v_min, v_max, None)
            
            del data
            return res

        total_n = 0
        total_sum_x = 0.0
        total_sum_x2 = 0.0
        global_min = float('inf')
        global_max = float('-inf')
        global_hist = np.zeros(self.hist_bins, dtype=np.int64) if self.compute_histogram else None

        with ThreadPoolExecutor(max_workers=2) as executor:
            curr_future = executor.submit(_process_chunk, z_ranges[0])
            for i in range(len(z_ranges)):
                if i + 1 < len(z_ranges):
                    next_future = executor.submit(_process_chunk, z_ranges[i+1])
                else:
                    next_future = None
                
                n, sx, sx2, v_min, v_max, h = curr_future.result()
                
                total_n += n
                total_sum_x += sx
                total_sum_x2 += sx2
                if v_min < global_min: global_min = v_min
                if v_max > global_max: global_max = v_max
                if h is not None: global_hist += h
                
                curr_future = next_future

        if total_n > 0:
            self.volume_mean = total_sum_x / total_n
            var = (total_sum_x2 / total_n) - (self.volume_mean**2)
            self.volume_std = float(np.sqrt(max(0, var)))
            self.volume_min = float(global_min)
            self.volume_max = float(global_max)
            self.volume_histogram = global_hist

    def _collect_volume_metadata(self) -> list[VolumeMetadata]:
        """Gather per-file metadata concurrently for the assembled volume.

        Returns:
            list[VolumeMetadata]: Per-file metadata entries including shape,
            dtype, estimated size in GiB, mean, std, min, and max.
        """
        metadata: list[VolumeMetadata | None] = [None] * len(self.volume_files)

        def process(file: Path, suffix: str) -> VolumeMetadata:
            res = read_image(
                file,
                suffix,
                read_to_array=False,
                low_cut=self.low_cut,
                high_cut=self.high_cut,
                compute_stats=self.compute_stats
            )
            # read_image(..., read_to_array=False) returns (shape, dtype, size, mean, std, min, max)
            shape, dtype, size, mean, std, min_v, max_v = res
            shape_zyx = tuple(int(dim) for dim in shape)  # normalize to ints
            return VolumeMetadata(
                shape=shape_zyx, 
                dtype=dtype, 
                size_gb=float(size),
                mean=float(mean),
                std=float(std),
                min=float(min_v),
                max=float(max_v)
            )

        with ThreadPoolExecutor(max_workers=self.io_workers) as executor:
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
