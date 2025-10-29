"""
Utility functions for deep learning operations.
"""

import torch
import psutil
import platform
from typing import Dict, Any, Optional


def get_device_info() -> Dict[str, Any]:
    """
    Get information about available devices.
    
    Returns:
        Dictionary containing device information
    """
    info = {
        "platform": platform.system(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cpu_count": psutil.cpu_count(),
        "memory_total_gb": psutil.virtual_memory().total / (1024**3),
    }
    
    if torch.cuda.is_available():
        info.update({
            "cuda_version": torch.version.cuda,
            "cuda_device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device(),
            "device_name": torch.cuda.get_device_name(),
            "cuda_memory_total_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
        })
    
    return info


def calculate_model_size(model: torch.nn.Module) -> Dict[str, Any]:
    """
    Calculate model size and memory requirements.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary containing model size information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate memory usage (assuming float32)
    param_memory_mb = total_params * 4 / (1024 * 1024)
    
    # Estimate forward pass memory (rough approximation)
    forward_memory_mb = param_memory_mb * 2  # Rough estimate
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": total_params - trainable_params,
        "parameter_memory_mb": param_memory_mb,
        "estimated_forward_memory_mb": forward_memory_mb,
        "estimated_total_memory_mb": param_memory_mb + forward_memory_mb,
    }


def optimize_memory_usage(model: torch.nn.Module) -> None:
    """
    Apply memory optimization techniques to the model.
    
    Args:
        model: PyTorch model to optimize
    """
    # Enable memory efficient attention if available
    if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
        # This is a placeholder - actual implementation would depend on model architecture
        pass
    
    # Set model to eval mode to reduce memory usage
    model.eval()
    
    # Enable gradient checkpointing if supported
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()


def get_optimal_batch_size(
    model: torch.nn.Module,
    input_shape: tuple,
    device: torch.device,
    max_memory_gb: float = 8.0
) -> int:
    """
    Find optimal batch size for given model and constraints.
    
    Args:
        model: PyTorch model
        input_shape: Shape of input tensor (batch_size, ...)
        device: Device to run on
        max_memory_gb: Maximum memory usage in GB
        
    Returns:
        Optimal batch size
    """
    model.eval()
    batch_size = 1
    
    while True:
        try:
            # Create dummy input
            dummy_input = torch.randn(batch_size, *input_shape[1:]).to(device)
            
            # Forward pass
            with torch.no_grad():
                _ = model(dummy_input)
            
            # Check memory usage
            if device.type == 'cuda':
                memory_used = torch.cuda.memory_allocated() / (1024**3)
                if memory_used > max_memory_gb:
                    break
            
            batch_size *= 2
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                break
            else:
                raise e
    
    return max(1, batch_size // 2)
