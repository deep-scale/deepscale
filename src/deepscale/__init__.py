"""
DeepScale - A Python library for large-scale deep learning training.

This package provides utilities for training massive models like LLMs
across thousands of GPUs, including distributed training, model parallelism,
and optimization techniques for extreme scale.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .scaling import ModelScaler, DataScaler
from .utils import get_device_info, calculate_model_size
from .parallel import ModelParallel, DataParallel
from .distributed import DistributedTrainer, LLMTrainer, MemoryOptimizer, ModelSharding, ZeroOptimizer

__all__ = [
    "ModelScaler",
    "DataScaler", 
    "get_device_info",
    "calculate_model_size",
    "ModelParallel",
    "DataParallel",
    "DistributedTrainer",
    "LLMTrainer",
    "MemoryOptimizer",
    "ModelSharding",
    "ZeroOptimizer",
]
