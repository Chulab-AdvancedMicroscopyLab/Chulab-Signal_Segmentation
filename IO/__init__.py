"""Public IO package exports for readers, writers, and shared metadata.

Exposes the high-level ``FileReader``/``FileWriter`` classes and common
constants. Also re-exports ``read_image`` from ``IO.reader_tools`` for
direct, single-file metadata/array opening.
"""
from .reader import FileReader
from .writer import FileWriter
from .datasets import (
    BaseMicroscopyDataset, 
    TrainMicroscopyDataset, 
    InferenceMicroscopyDataset,
    build_train_dataset_from_config
)
from .IO_types import OUTPUT_CHOICES, TYPE_MAP, VALID_SUFFIXES, VolumeMetadata

__all__ = [
    "FileReader",
    "FileWriter",
    "BaseMicroscopyDataset",
    "TrainMicroscopyDataset",
    "InferenceMicroscopyDataset",
    "build_train_dataset_from_config",
    "OUTPUT_CHOICES",
    "TYPE_MAP",
    "VALID_SUFFIXES",
    "VolumeMetadata",
]
