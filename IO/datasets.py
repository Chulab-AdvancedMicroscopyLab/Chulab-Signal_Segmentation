from __future__ import annotations
import logging
import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from IO.reader import FileReader
from utils.cropper import PatchSlice, generate_patch_indices, filter_indices_by_mask, extract_data_from_indices

logger = logging.getLogger(__name__)

def _volume_pad_widths(shape: Tuple[int, int, int], min_size: Tuple[int, int, int]) -> List[Tuple[int, int]]:
    """Return np.pad pad_width so each dim >= min_size. Z: symmetric. Y/X: end-only."""
    z_pad = max(0, min_size[0] - shape[0])
    return [
        (z_pad // 2, z_pad - z_pad // 2),   # Z symmetric
        (0, max(0, min_size[1] - shape[1])), # Y end
        (0, max(0, min_size[2] - shape[2])), # X end
    ]

def _div32_pad_widths(d: int, h: int, w: int, is_3d: bool) -> Tuple[int, int, int]:
    """Return (pad_d, pad_h, pad_w) to make dims divisible by 32."""
    pad_d = (32 - d % 32) % 32 if is_3d else 0
    pad_h = (32 - h % 32) % 32
    pad_w = (32 - w % 32) % 32
    return pad_d, pad_h, pad_w

def _pad_image(data: np.ndarray, pad_width, mode: str, fill: float = 0.0) -> np.ndarray:
    """Apply np.pad with correct kwargs for constant vs reflect/symmetric/edge modes."""
    kw = {"constant_values": fill} if mode == "constant" else {}
    return np.pad(data, pad_width, mode=mode, **kw)

@dataclass(frozen=True)
class PatchMetadata:
    """Metadata for a single patch.
    
    If is_patch_mode=True: volume_idx is the index into the flattened patch tensor.
    If is_patch_mode=False: volume_idx is the index into the image_tensors list, 
                            and slices defines the crop.
    """
    volume_idx: int
    slices: Optional[PatchSlice] = None

class BaseMicroscopyDataset(Dataset):
    def __init__(
        self,
        image_tensors: List[torch.Tensor],
        mask_tensors: Optional[List[torch.Tensor]] = None,
        patch_indices: List[PatchMetadata] = None,
        transform: Optional[Callable] = None,
        with_mask: bool = True,
        is_patch_mode: bool = False
    ):
        self.image_tensors = image_tensors
        self.mask_tensors = mask_tensors
        self.patch_indices = patch_indices or []
        self.transform = transform
        self.with_mask = with_mask and mask_tensors is not None
        self.is_patch_mode = is_patch_mode

        for t in self.image_tensors:
            if not t.is_shared():
                t.share_memory_()
        
        if self.mask_tensors:
            for t in self.mask_tensors:
                if not t.is_shared():
                    t.share_memory_()

    def __len__(self) -> int:
        return len(self.patch_indices)

    def __getitem__(self, idx: int):
        meta = self.patch_indices[idx]
        
        if self.is_patch_mode:
            image_patch = self.image_tensors[0][meta.volume_idx]
            sample = {"image": image_patch}
            if self.with_mask:
                sample["mask"] = self.mask_tensors[0][meta.volume_idx]
        else:
            slices = meta.slices
            image_vol = self.image_tensors[meta.volume_idx]
            image_patch = image_vol[:, slices.z_slice, slices.y_slice, slices.x_slice]
            sample = {"image": image_patch}
            if self.with_mask:
                mask_vol = self.mask_tensors[meta.volume_idx]
                sample["mask"] = mask_vol[:, slices.z_slice, slices.y_slice, slices.x_slice]

        # Handle 2D cases: if depth is 1, squeeze it to (C, H, W) for 2D model compatibility
        if sample["image"].shape[1] == 1:
            sample["image"] = sample["image"].squeeze(1)
            if self.with_mask:
                sample["mask"] = sample["mask"].squeeze(1)

        if self.transform is not None:
            sample = self.transform(sample)

        if not self.with_mask:
            return sample["image"]
        
        return sample["image"], sample["mask"]

class TrainMicroscopyDataset(BaseMicroscopyDataset):
    @classmethod
    def from_folders(
        cls,
        full_config: dict,
        image_root: Union[str, List[str]],
        mask_root: Union[str, List[str]],
        patch_size: Tuple[int, int, int],
        overlap: Tuple[int, int, int],
        transform: Optional[Callable] = None,
        neg_keep_ratio: float = 1.0,
        input_name: str = "Flatten_561",
        mask_name: str = "Flatten_561_mask",
        io_workers: int = 4
    ):
        from utils.normalization import build_normalizer_from_config

        all_image_patches = []
        all_mask_patches = []
        
        image_roots = [image_root] if isinstance(image_root, str) else image_root
        mask_roots = [mask_root] if isinstance(mask_root, str) else mask_root
        
        if len(image_roots) != len(mask_roots):
            if len(mask_roots) == 1:
                mask_roots = mask_roots * len(image_roots)
            elif len(image_roots) == 1:
                image_roots = image_roots * len(mask_roots)
            else:
                raise ValueError("image_root and mask_root lists must have the same length.")

        train_config = full_config.get("train", {})
        preprocess_config = train_config.get("preprocess", {})
        low_cut = preprocess_config.get("low_cut")
        high_cut = preprocess_config.get("high_cut")
        sample_rate = preprocess_config.get("sample_rate", 1.0)
        method = preprocess_config.get("normalize_mode", "z-score")
        pad_mode = preprocess_config.get("pad_mode", "constant")

        volumes_found = []
        for img_root, msk_root in zip(image_roots, mask_roots):
            img_root_path = Path(img_root)
            msk_root_path = Path(msk_root)
            
            for p in img_root_path.rglob("*"):
                if p.is_dir() and p.name == input_name:
                    rel_path = p.parent.relative_to(img_root_path)
                    m_path = msk_root_path / rel_path / mask_name
                    
                    if m_path.exists() and m_path.is_dir():
                        volumes_found.append((p, m_path))

        for img_path, msk_path in sorted(volumes_found):
            v_display_name = f"{img_path.parent.name}/{img_path.name}"
            
            img_reader = FileReader(
                img_path, 
                io_workers=io_workers, 
                compute_stats=True, 
                stats_sample_rate=sample_rate,
                low_cut=low_cut,
                high_cut=high_cut,
                compute_histogram=(method == "histogram")
            )
            img_data = img_reader.read(out_dtype=np.float32)
            
            # Read mask data
            msk_reader = FileReader(msk_path, io_workers=io_workers)
            msk_data = msk_reader.read(out_dtype=np.float32)

            # Pad volume to ensure it fits at least one patch (Z: symmetric, Y/X: end)
            vol_pad = _volume_pad_widths(img_data.shape, patch_size)
            if any(before + after > 0 for before, after in vol_pad):
                padded_shape = tuple(img_data.shape[i] + vol_pad[i][0] + vol_pad[i][1] for i in range(3))
                logger.info(f"Volume {img_path.parent.name}: Padded {img_data.shape} -> {padded_shape} (mode={pad_mode})")
                img_data = _pad_image(img_data, vol_pad, pad_mode, fill=0.0)
                msk_data = _pad_image(msk_data, vol_pad, pad_mode, fill=0.0)


            # Apply Normalization
            normalizer = build_normalizer_from_config(full_config, img_reader, mode="train")
            img_data = normalizer(img_data)
            
            indices = generate_patch_indices(img_data.shape, patch_size, overlap)
            filtered = filter_indices_by_mask(msk_data, indices, neg_keep_ratio)
            
            img_patches = extract_data_from_indices(img_data, filtered, as_stack=True)
            msk_patches = extract_data_from_indices(msk_data, filtered, as_stack=True)
            
            # Pad patches to divisibility-by-32 required by SwinUNETR (end-only, N dim untouched)
            n, d, h, w = img_patches.shape
            pad_d, pad_h, pad_w = _div32_pad_widths(d, h, w, is_3d=(d > 1))
            if pad_d > 0 or pad_h > 0 or pad_w > 0:
                patch_pad = ((0, 0), (0, pad_d), (0, pad_h), (0, pad_w))
                img_patches = _pad_image(img_patches, patch_pad, pad_mode, fill=normalizer.get_background_value())
                msk_patches = _pad_image(msk_patches, patch_pad, "constant", fill=0.0)
                logger.debug(f"Patches div-32 padded: ({d},{h},{w}) -> ({d+pad_d},{h+pad_h},{w+pad_w})")

            # Convert to torch and add channel dimension: (N, D, H, W) -> (N, 1, D, H, W)
            all_image_patches.append(torch.from_numpy(img_patches).unsqueeze(1))
            all_mask_patches.append(torch.from_numpy(msk_patches).unsqueeze(1))
            
            logger.info(f"Volume {v_display_name}: Extracted {len(filtered)} patches.")
            
        if not all_image_patches:
            raise RuntimeError(f"No valid patches were extracted from {image_root}")

        image_stack = torch.cat(all_image_patches).share_memory_()
        mask_stack = torch.cat(all_mask_patches).share_memory_()
        
        # patch_indices should be for the final total number of patches
        patch_indices = [PatchMetadata(volume_idx=i) for i in range(len(image_stack))]
        
        return cls(
            image_tensors=[image_stack],
            mask_tensors=[mask_stack],
            patch_indices=patch_indices,
            transform=transform,
            is_patch_mode=True
        )

    def split(
        self, 
        val_ratio: float = 0.2, 
        seed: int = 42,
        train_transform: Optional[Callable] = None,
        val_transform: Optional[Callable] = None
    ) -> tuple[TrainMicroscopyDataset, TrainMicroscopyDataset]:
        """Splits indices while keeping the underlying shared tensors identical."""
        n = len(self.patch_indices)
        indices = np.arange(n)
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
        
        val_n = int(n * val_ratio)
        val_idx = indices[:val_n]
        train_idx = indices[val_n:]
        
        train_ds = TrainMicroscopyDataset(
            image_tensors=self.image_tensors,
            mask_tensors=self.mask_tensors,
            patch_indices=[self.patch_indices[i] for i in train_idx],
            transform=train_transform or self.transform,
            with_mask=self.with_mask,
            is_patch_mode=True
        )
        
        val_ds = TrainMicroscopyDataset(
            image_tensors=self.image_tensors,
            mask_tensors=self.mask_tensors,
            patch_indices=[self.patch_indices[i] for i in val_idx],
            transform=val_transform or self.transform,
            with_mask=self.with_mask,
            is_patch_mode=True
        )
        
        return train_ds, val_ds

class InferenceMicroscopyDataset(BaseMicroscopyDataset):
    """
    Inference Dataset that pre-crops patches and stores them in a single shared stack.
    """
    def __init__(
        self,
        full_config: dict,
        image_reader: FileReader,
        z_range: Tuple[int, int],
        patch_size: Tuple[int, int, int],
        overlap: Tuple[int, int, int],
        transform: Optional[Callable] = None,
    ):
        from utils.normalization import build_normalizer_from_config

        z_start, z_end = z_range
        inf_preprocess = full_config.get("inference", {}).get("preprocess", {})
        pad_mode = inf_preprocess.get("pad_mode", "constant")

        # 1. Read the full window (optimized parallel load + conversion)
        img_data = image_reader.read(z_start=z_start, z_end=z_end, out_dtype=np.float32)
        
        # Pad volume to ensure it fits at least one patch (Z: symmetric, Y/X: end)
        vol_pad = _volume_pad_widths(img_data.shape, patch_size)
        self.pad_z_before = vol_pad[0][0]
        if any(before + after > 0 for before, after in vol_pad):
            orig_shape = img_data.shape
            img_data = _pad_image(img_data, vol_pad, pad_mode, fill=0.0)
            logger.debug(f"Chunk padded {orig_shape} -> {img_data.shape} to fit patch_size {patch_size} (mode={pad_mode})")

        # 2. Global normalization
        normalizer = build_normalizer_from_config(full_config, image_reader, mode="inference")
        normalizer(img_data)
        
        # 3. Generate geometry
        raw_indices = generate_patch_indices(img_data.shape, patch_size, overlap, z_offset=z_start)
        
        # 4. Pre-extract all patches (this ensures workers do zero slicing/computation)
        img_patches = extract_data_from_indices(img_data, raw_indices, as_stack=True)
        
        # 5. Pad patches to divisibility-by-32 required by SwinUNETR (end-only, N dim untouched)
        n, d, h, w = img_patches.shape
        pad_d, pad_h, pad_w = _div32_pad_widths(d, h, w, is_3d=(d > 1))
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            patch_pad = ((0, 0), (0, pad_d), (0, pad_h), (0, pad_w))
            img_patches = _pad_image(img_patches, patch_pad, pad_mode, fill=normalizer.get_background_value())
            logger.debug(f"Patches div-32 padded: ({d},{h},{w}) -> ({d+pad_d},{h+pad_h},{w+pad_w})")

        # 6. Pack into a single contiguous shared tensor
        # Shape: (N_patches, 1, D_padded, H_padded, W_padded)
        image_stack = torch.from_numpy(img_patches).unsqueeze(1).share_memory_()
        
        # 7. Map indices to the stack while preserving PatchSlice for stitching
        patch_indices = [
            PatchMetadata(volume_idx=i, slices=raw_indices[i]) 
            for i in range(len(raw_indices))
        ]
        
        super().__init__(
            image_tensors=[image_stack],
            patch_indices=patch_indices,
            transform=transform,
            with_mask=False,
            is_patch_mode=True
        )

def build_train_dataset_from_config(
    full_config: dict, 
    train_transform: Optional[Callable] = None, 
    val_transform: Optional[Callable] = None
) -> Tuple[TrainMicroscopyDataset, TrainMicroscopyDataset]:
    """
    Initializes and splits a training dataset based on the provided configuration dictionary.
    """
    config = full_config.get("train", {})
    resources = full_config.get("resources", {})
    io_workers = resources.get("io_workers", 4)
    
    data_root = config.get("data_path")
    img_root = config.get("img_path", data_root)
    mask_root = config.get("mask_path", data_root)
    
    if not img_root or not mask_root:
        raise ValueError("Missing 'data_path' (or 'img_path'/'mask_path') in training config.")

    patch_size = tuple(config.get("training_patch_size", [1, 64, 64]))
    overlap = tuple(config.get("training_overlay", [0, 0, 0]))
    neg_ratio = config.get("training_neg_keep_ratio", 1.0)
    val_ratio = config.get("val_ratio", 0.3)
    seed = config.get("seed", 42)

    logger.info(f"Loading training data from {img_root}...")
    full_dataset = TrainMicroscopyDataset.from_folders(
        full_config=full_config,
        image_root=img_root,
        mask_root=mask_root,
        patch_size=patch_size,
        overlap=overlap,
        neg_keep_ratio=neg_ratio,
        input_name=config.get("input_name", "images"),
        mask_name=config.get("mask_name", "images_mask"),
        io_workers=io_workers
    )

    return full_dataset.split(
        val_ratio=val_ratio, 
        seed=seed, 
        train_transform=train_transform, 
        val_transform=val_transform
    )
