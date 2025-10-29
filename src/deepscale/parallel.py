"""
Parallel processing utilities for deep learning models.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any


class ModelParallel:
    """Utility class for model parallelism."""
    
    def __init__(self, model: nn.Module, device_ids: Optional[List[int]] = None):
        """
        Initialize ModelParallel.
        
        Args:
            model: PyTorch model
            device_ids: List of device IDs to use
        """
        self.model = model
        self.device_ids = device_ids or list(range(torch.cuda.device_count()))
        self.parallel_model = None
    
    def setup_model_parallel(self) -> nn.Module:
        """
        Set up model parallelism.
        
        Returns:
            Parallelized model
        """
        if len(self.device_ids) > 1:
            self.parallel_model = nn.DataParallel(self.model, device_ids=self.device_ids)
        else:
            self.parallel_model = self.model
        
        return self.parallel_model
    
    def get_device_memory_usage(self) -> Dict[int, float]:
        """
        Get memory usage for each device.
        
        Returns:
            Dictionary mapping device ID to memory usage in GB
        """
        memory_usage = {}
        for device_id in self.device_ids:
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated(device_id) / (1024**3)
                memory_usage[device_id] = memory_allocated
        
        return memory_usage


class DataParallel:
    """Utility class for data parallelism."""
    
    def __init__(self, batch_size: int = 32, num_workers: int = 4):
        """
        Initialize DataParallel.
        
        Args:
            batch_size: Batch size per worker
            num_workers: Number of workers
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def calculate_effective_batch_size(self) -> int:
        """
        Calculate effective batch size across all workers.
        
        Returns:
            Effective batch size
        """
        return self.batch_size * self.num_workers
    
    def split_data_across_workers(self, data: torch.Tensor) -> List[torch.Tensor]:
        """
        Split data across workers.
        
        Args:
            data: Input data tensor
            
        Returns:
            List of data chunks for each worker
        """
        chunk_size = data.size(0) // self.num_workers
        return torch.split(data, chunk_size)
    
    def gather_results(self, results: List[torch.Tensor]) -> torch.Tensor:
        """
        Gather results from all workers.
        
        Args:
            results: List of result tensors from workers
            
        Returns:
            Concatenated results
        """
        return torch.cat(results, dim=0)


class PipelineParallel:
    """Utility class for pipeline parallelism."""
    
    def __init__(self, model: nn.Module, num_stages: int = 2):
        """
        Initialize PipelineParallel.
        
        Args:
            model: PyTorch model
            num_stages: Number of pipeline stages
        """
        self.model = model
        self.num_stages = num_stages
        self.stages = []
    
    def split_model_into_stages(self) -> List[nn.Module]:
        """
        Split model into pipeline stages.
        
        Returns:
            List of model stages
        """
        # This is a simplified implementation
        # In practice, you'd need to carefully split the model based on its architecture
        modules = list(self.model.children())
        stage_size = len(modules) // self.num_stages
        
        for i in range(self.num_stages):
            start_idx = i * stage_size
            end_idx = start_idx + stage_size if i < self.num_stages - 1 else len(modules)
            stage = nn.Sequential(*modules[start_idx:end_idx])
            self.stages.append(stage)
        
        return self.stages
    
    def forward_pipeline(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through pipeline stages.
        
        Args:
            input_data: Input tensor
            
        Returns:
            Output tensor
        """
        if not self.stages:
            self.split_model_into_stages()
        
        x = input_data
        for stage in self.stages:
            x = stage(x)
        
        return x
