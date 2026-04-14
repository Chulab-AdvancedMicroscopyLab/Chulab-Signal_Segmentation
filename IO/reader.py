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

    def __init__(self, input_path, memory_limit_gb=32, io_workers=4, compute_stats=False, stats_sample_rate=1.0):
        self.input_path = Path(input_path)
        self.memory_limit_bytes = memory_limit_gb * 1024 ** 3
        self.io_workers = io_workers
        self.compute_stats = compute_stats
        self.stats_sample_rate = float(stats_sample_rate)

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
        is_zarr = len(self.volume_types) > 0 and self.volume_types[0] == '.zarr'
        
        if (total_to_load * 2 > mem_limit) and not is_zarr:
            raise MemoryError(f"Need {total_to_load*2:.2f}GiB but limit is {mem_limit:.2f}GiB")

        # 4) pre-allocate output
        out = np.empty((dz, dy, dx), dtype=self.volume_dtype)

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
                    max_workers=self.io_workers
                )
                slab = arr[file_z0:file_z1, y0:y1, x0:x1]
                out[offset:offset+length, :, :] = slab
                del arr, slab
                offset += length
        else:
            # Multithreaded loading for standard files (TIFF, PNG, etc.)
            num_needed = len(needed)
            def _load_task(task):
                idx, f_z0, f_z1, out_offset, length = task
                # If we are reading multiple files in parallel, we limit each file's 
                # internal decompression to 1 worker to avoid over-subscribing CPU.
                # If only one file is needed, we let it use all io_workers.
                dec_workers = 1 if num_needed > 1 else self.io_workers
                
                arr = read_image(
                    self.volume_files[idx],
                    self.volume_types[idx],
                    read_to_array=True,
                    max_workers=dec_workers
                )
                # Ensure we are only grabbing the requested sub-crop in Y and X as well
                out[out_offset:out_offset+length, :, :] = arr[f_z0:f_z1, y0:y1, x0:x1]

            task_list = []
            current_offset = 0
            for idx, base_z, f_z0, f_z1 in needed:
                length = f_z1 - f_z0
                task_list.append((idx, f_z0, f_z1, current_offset, length))
                current_offset += length

            with ThreadPoolExecutor(max_workers=self.io_workers) as executor:
                # Using map or submit here is cleaner than manual grouping 
                # unless you have thousands of tiny files.
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
        """Populate aggregate metadata (shape, dtype, sizes) for the volume.

        Raises:
            RuntimeError: If metadata collection fails for any source file.
            ValueError: If XY shapes or dtypes are inconsistent.
        """
        # Pass 1: Basic Metadata (+ Optional Stats if files are read anyway)
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

        # Pass 2: Global Stats
        if not self.compute_stats:
            self.volume_mean, self.volume_std = 0.0, 0.0
            return

        # Check if we already have stats for ALL files from the metadata pass
        if all(e.mean != 0.0 or e.std != 0.0 for e in entries):
            logger.info("Aggregating global statistics from metadata pass...")
            self.volume_mean, self.volume_std = self._calculate_aggregate_stats(entries)
        else:
            # We are missing stats for some files (likely large containers that used lightweight metadata)
            # Run the efficient double-buffered chunked pass
            self.volume_mean, self.volume_std = self._compute_global_stats()

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

        return float(mean), float(std)

    def _compute_global_stats(self) -> tuple[float, float]:
        """Calculate weighted mean and pooled standard deviation across the full volume.
        
        Uses a double-buffering approach: one background thread handles sequential IO 
        while the main thread performs calculations. This avoids IO competition.
        """
        slice_size_bytes = self.volume_shape[1] * self.volume_shape[2] * self.volume_dtype.itemsize
        target_chunk_bytes = self.memory_limit_bytes * 0.1
        chunk_size = max(1, int(target_chunk_bytes / slice_size_bytes))
        chunk_size = min(chunk_size, 128)
        
        # Calculate sampling step (e.g. 0.5 sample rate -> step of 2 chunks)
        step = max(1, int(1.0 / self.stats_sample_rate)) if self.stats_sample_rate < 1.0 else 1
        
        if step > 1:
            logger.info(f"Computing global volume statistics in chunks of {chunk_size} Z-slices (Sampling 1/{step} chunks)...")
        else:
            logger.info(f"Computing global volume statistics in chunks of {chunk_size} Z-slices...")

        all_z_ranges = []
        for z0 in range(0, self.volume_shape[0], chunk_size):
            z1 = min(z0 + chunk_size, self.volume_shape[0])
            all_z_ranges.append((z0, z1))
        
        # Apply chunk-wise sampling
        z_ranges = all_z_ranges[::step]

        def _process_chunk(z_range):
            """Load data and compute local sum/sum_sq in one pass to avoid returning large arrays."""
            data = self.read(z_start=z_range[0], z_end=z_range[1])
            if data.size == 0:
                return 0, 0.0, 0.0
            
            # Use parallel numba for fast single-pass stats
            res = _compute_accumulators_numba(data)
            
            del data # Explicitly free
            return res

        total_n = 0
        total_sum_x = 0.0
        total_sum_x2 = 0.0

        # Use 2 workers to allow one chunk to be processed (calc) while the next is being loaded.
        # This effectively overlaps the CPU-bound calc with the IO-bound read.
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Pre-submit the first chunk
            curr_future = executor.submit(_process_chunk, z_ranges[0])
            
            for i in range(len(z_ranges)):
                # 1. Pre-submit the NEXT chunk to the other worker immediately
                if i + 1 < len(z_ranges):
                    next_future = executor.submit(_process_chunk, z_ranges[i+1])
                else:
                    next_future = None
                
                # 2. Wait for the CURRENT chunk results (n, sum_x, sum_x2)
                n, sx, sx2 = curr_future.result()
                
                # 3. Accumulate stats for the current chunk
                total_n += n
                total_sum_x += sx
                total_sum_x2 += sx2
                
                # 4. Advance the future pointer for the next iteration
                curr_future = next_future

        if total_n == 0:
            return 0.0, 0.0

        mean = total_sum_x / total_n
        var = (total_sum_x2 / total_n) - (mean**2)
        std = float(np.sqrt(max(0, var)))
        
        return float(mean), std

    def _collect_volume_metadata(self) -> list[VolumeMetadata]:
        """Gather per-file metadata concurrently for the assembled volume.

        Returns:
            list[VolumeMetadata]: Per-file metadata entries including shape,
            dtype, estimated size in GiB, mean, and std.
        """
        metadata: list[VolumeMetadata | None] = [None] * len(self.volume_files)

        def process(file: Path, suffix: str) -> VolumeMetadata:
            res = read_image(
                file,
                suffix,
                read_to_array=False,
                compute_stats=self.compute_stats
            )
            # read_image(..., read_to_array=False) returns (shape, dtype, size, mean, std)
            shape, dtype, size, mean, std = res
            shape_zyx = tuple(int(dim) for dim in shape)  # normalize to ints
            return VolumeMetadata(
                shape=shape_zyx, 
                dtype=dtype, 
                size_gb=float(size),
                mean=float(mean),
                std=float(std)
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
