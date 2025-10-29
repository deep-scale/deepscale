"""
Scaling utilities for deep learning models.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union
import numpy as np


class ModelScaler:
    """Utility class for scaling deep learning models."""
    
    def __init__(self, model: nn.Module):
        """
        Initialize the ModelScaler.
        
        Args:
            model: PyTorch model to scale
        """
        self.model = model
        self.original_state = None
    
    def scale_parameters(self, scale_factor: float) -> None:
        """
        Scale model parameters by a given factor.
        
        Args:
            scale_factor: Factor to scale parameters by
        """
        for param in self.model.parameters():
            param.data *= scale_factor
    
    def get_model_size(self) -> int:
        """
        Get the total number of parameters in the model.
        
        Returns:
            Total number of parameters
        """
        return sum(p.numel() for p in self.model.parameters())
    
    def get_model_size_mb(self) -> float:
        """
        Get the model size in megabytes.
        
        Returns:
            Model size in MB
        """
        total_params = self.get_model_size()
        # Assuming float32 (4 bytes per parameter)
        return total_params * 4 / (1024 * 1024)
    
    def save_state(self) -> None:
        """Save the current model state."""
        self.original_state = self.model.state_dict().copy()
    
    def restore_state(self) -> None:
        """Restore the saved model state."""
        if self.original_state is not None:
            self.model.load_state_dict(self.original_state)


class DataScaler:
    """Utility class for scaling data and batch operations."""
    
    def __init__(self, batch_size: int = 32):
        """
        Initialize the DataScaler.
        
        Args:
            batch_size: Default batch size
        """
        self.batch_size = batch_size
    
    def scale_batch_size(self, scale_factor: float) -> int:
        """
        Scale the batch size by a given factor.
        
        Args:
            scale_factor: Factor to scale batch size by
            
        Returns:
            New batch size
        """
        return int(self.batch_size * scale_factor)
    
    def calculate_gradient_accumulation_steps(self, target_batch_size: int) -> int:
        """
        Calculate gradient accumulation steps needed to achieve target batch size.
        
        Args:
            target_batch_size: Desired effective batch size
            
        Returns:
            Number of gradient accumulation steps
        """
        return max(1, target_batch_size // self.batch_size)
    
    def split_batch(self, data: torch.Tensor, num_splits: int) -> list:
        """
        Split a batch into smaller chunks.
        
        Args:
            data: Input tensor
            num_splits: Number of splits
            
        Returns:
            List of split tensors
        """
        chunk_size = data.size(0) // num_splits
        return torch.split(data, chunk_size)
