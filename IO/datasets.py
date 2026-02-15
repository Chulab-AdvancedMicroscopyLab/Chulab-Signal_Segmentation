from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from IO.reader import FileReader
from utils.cropper import PatchSlice, generate_patch_indices, filter_indices_by_mask, extract_data_from_indices

logger = logging.getLogger(__name__)

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

        if self.transform is not None:
            sample = self.transform(sample)

        if not self.with_mask:
            return sample["image"]
        
        return sample["image"], sample["mask"]

class TrainMicroscopyDataset(BaseMicroscopyDataset):
    @classmethod
    def from_folders(
        cls,
        image_root: str,
        mask_root: str,
        patch_size: Tuple[int, int, int],
        overlap: Tuple[int, int, int],
        transform: Optional[Callable] = None,
        neg_keep_ratio: float = 1.0,
        input_name: str = "Flatten_561",
        mask_name: str = "Flatten_561_mask"
    ):
        all_image_patches = []
        all_mask_patches = []
        
        image_root_path = Path(image_root)
        mask_root_path = Path(mask_root)
        
        # We assume image_root and mask_root might be the same or different.
        # But based on the new structure, we want to find all 'input_name' folders.
        # We can search under image_root for all 'input_name' directories.
        
        volumes_found = []
        for p in image_root_path.rglob("*"):
            if p.is_dir() and p.name == input_name:
                # Find relative path from image_root
                rel_path = p.parent.relative_to(image_root_path)
                # Check if mask exists in corresponding position under mask_root
                m_path = mask_root_path / rel_path / mask_name
                
                if m_path.exists() and m_path.is_dir():
                    volumes_found.append((p, m_path))

        if not volumes_found:
            # Fallback for old flat structure if no nested volumes found
            # or if the user passed the direct parent of Flatten_561
            if image_root_path.name == input_name:
                m_path = mask_root_path.parent / mask_name
                if m_path.exists():
                    volumes_found.append((image_root_path, m_path))
            else:
                # Check if image_root contains input_name and mask_name directly
                p = image_root_path / input_name
                m = mask_root_path / mask_name
                if p.exists() and m.exists():
                    volumes_found.append((p, m))

        for img_path, msk_path in sorted(volumes_found):
            v_display_name = f"{img_path.parent.name}/{img_path.name}"
            
            img_reader = FileReader(img_path)
            if img_reader.volume_shape[0] < patch_size[0]:
                logger.warning(f"Skipping {v_display_name}: Z-size {img_reader.volume_shape[0]} < patch_size {patch_size[0]}")
                continue
                
            img_data = img_reader.read()
            img_data = (img_data - img_reader.volume_mean) / (img_reader.volume_std + 1e-8)
            
            msk_reader = FileReader(msk_path)
            msk_data = msk_reader.read().astype(np.float32)
            
            indices = generate_patch_indices(img_data.shape, patch_size, overlap)
            filtered = filter_indices_by_mask(msk_data, indices, neg_keep_ratio)
            
            img_p_list = extract_data_from_indices(img_data, filtered)
            msk_p_list = extract_data_from_indices(msk_data, filtered)
            
            all_image_patches.extend([torch.from_numpy(p).unsqueeze(0) for p in img_p_list])
            all_mask_patches.extend([torch.from_numpy(p).unsqueeze(0) for p in msk_p_list])
            
            logger.info(f"Volume {v_display_name}: Extracted {len(filtered)} patches.")
            
        if not all_image_patches:
            raise RuntimeError(f"No valid patches were extracted from {image_dir}")

        image_stack = torch.stack(all_image_patches).share_memory_()
        mask_stack = torch.stack(all_mask_patches).share_memory_()
        
        patch_indices = [PatchMetadata(volume_idx=i) for i in range(len(all_image_patches))]
        
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
        image_reader: FileReader,
        z_range: Tuple[int, int],
        patch_size: Tuple[int, int, int],
        overlap: Tuple[int, int, int],
        transform: Optional[Callable] = None,
    ):
        z_start, z_end = z_range
        # 1. Read and Normalize the full window once
        img_data = image_reader.read(z_start=z_start, z_end=z_end)
        img_data = (img_data - image_reader.volume_mean) / (image_reader.volume_std + 1e-8)
        
        # 2. Generate geometry
        raw_indices = generate_patch_indices(img_data.shape, patch_size, overlap, z_offset=z_start)
        
        # 3. Pre-extract all patches (this ensures workers do zero slicing/computation)
        img_p_list = extract_data_from_indices(img_data, raw_indices)
        
        # 4. Pack into a single contiguous shared tensor
        # Shape: (N_patches, 1, D, H, W)
        image_stack = torch.stack([torch.from_numpy(p).unsqueeze(0) for p in img_p_list]).share_memory_()
        
        # 5. Map indices to the stack while preserving PatchSlice for stitching
        patch_indices = [
            PatchMetadata(volume_idx=i, slices=raw_indices[i]) 
            for i in range(len(img_p_list))
        ]
        
        super().__init__(
            image_tensors=[image_stack],
            patch_indices=patch_indices,
            transform=transform,
            with_mask=False,
            is_patch_mode=True
        )

def load_train_dataset_from_config(
    config: dict, 
    train_transform: Optional[Callable] = None, 
    val_transform: Optional[Callable] = None
) -> Tuple[TrainMicroscopyDataset, TrainMicroscopyDataset]:
    """
    Initializes and splits a training dataset based on the provided configuration dictionary.
    """
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
        image_root=img_root,
        mask_root=mask_root,
        patch_size=patch_size,
        overlap=overlap,
        neg_keep_ratio=neg_ratio,
        input_name=config.get("input_name", "Flatten_561"),
        mask_name=config.get("mask_name", "Flatten_561_mask")
    )

    return full_dataset.split(
        val_ratio=val_ratio, 
        seed=seed, 
        train_transform=train_transform, 
        val_transform=val_transform
    )
