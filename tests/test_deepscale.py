"""
Tests for DeepScale package.
"""

import pytest
import torch
import torch.nn as nn
from deepscale import (
    ModelScaler, 
    DataScaler, 
    DistributedTrainer,
    LLMTrainer,
    MemoryOptimizer,
    ModelSharding,
    ZeroOptimizer,
    get_device_info,
    calculate_model_size
)


class TestModelScaler:
    """Test ModelScaler functionality."""
    
    def test_model_scaler_init(self):
        """Test ModelScaler initialization."""
        model = nn.Linear(10, 5)
        scaler = ModelScaler(model)
        assert scaler.model == model
        assert scaler.original_state is None
    
    def test_get_model_size(self):
        """Test getting model size."""
        model = nn.Linear(10, 5)
        scaler = ModelScaler(model)
        size = scaler.get_model_size()
        assert size == 55  # 10*5 + 5 (bias)
    
    def test_get_model_size_mb(self):
        """Test getting model size in MB."""
        model = nn.Linear(10, 5)
        scaler = ModelScaler(model)
        size_mb = scaler.get_model_size_mb()
        expected = 55 * 4 / (1024 * 1024)  # 55 params * 4 bytes
        assert abs(size_mb - expected) < 1e-6


class TestDataScaler:
    """Test DataScaler functionality."""
    
    def test_data_scaler_init(self):
        """Test DataScaler initialization."""
        scaler = DataScaler(batch_size=32)
        assert scaler.batch_size == 32
    
    def test_scale_batch_size(self):
        """Test batch size scaling."""
        scaler = DataScaler(batch_size=32)
        new_size = scaler.scale_batch_size(2.0)
        assert new_size == 64
    
    def test_calculate_gradient_accumulation_steps(self):
        """Test gradient accumulation calculation."""
        scaler = DataScaler(batch_size=32)
        steps = scaler.calculate_gradient_accumulation_steps(128)
        assert steps == 4


class TestDistributedTrainer:
    """Test DistributedTrainer functionality."""
    
    def test_distributed_trainer_init(self):
        """Test DistributedTrainer initialization."""
        model = nn.Linear(10, 5)
        trainer = DistributedTrainer(model, num_gpus=4)
        assert trainer.num_gpus == 4
        assert trainer.batch_size_per_gpu == 1
        assert trainer.gradient_accumulation_steps == 1
    
    def test_calculate_effective_batch_size(self):
        """Test effective batch size calculation."""
        model = nn.Linear(10, 5)
        trainer = DistributedTrainer(
            model, 
            num_gpus=4, 
            batch_size_per_gpu=8,
            gradient_accumulation_steps=2
        )
        effective_size = trainer.calculate_effective_batch_size()
        assert effective_size == 64  # 4 * 8 * 2


class TestLLMTrainer:
    """Test LLMTrainer functionality."""
    
    def test_llm_trainer_init(self):
        """Test LLMTrainer initialization."""
        trainer = LLMTrainer(
            model_size="7B",
            num_gpus=8,
            sequence_length=2048
        )
        assert trainer.model_size == "7B"
        assert trainer.num_gpus == 8
        assert trainer.sequence_length == 2048
        assert trainer.gradient_checkpointing is True


class TestMemoryOptimizer:
    """Test MemoryOptimizer functionality."""
    
    def test_memory_optimizer_init(self):
        """Test MemoryOptimizer initialization."""
        model = nn.Linear(10, 5)
        optimizer = MemoryOptimizer(model)
        assert optimizer.model == model
        assert optimizer.offload_optimizer is True
        assert optimizer.cpu_offload is True
    
    def test_calculate_memory_savings(self):
        """Test memory savings calculation."""
        model = nn.Linear(10, 5)
        optimizer = MemoryOptimizer(model)
        savings = optimizer.calculate_memory_savings()
        assert savings > 0


class TestModelSharding:
    """Test ModelSharding functionality."""
    
    def test_model_sharding_init(self):
        """Test ModelSharding initialization."""
        model = nn.Sequential(nn.Linear(10, 5), nn.Linear(5, 2))
        sharding = ModelSharding(model, num_shards=2)
        assert sharding.model == model
        assert sharding.num_shards == 2
        assert sharding.sharding_strategy == "tensor_parallel"


class TestZeroOptimizer:
    """Test ZeroOptimizer functionality."""
    
    def test_zero_optimizer_init(self):
        """Test ZeroOptimizer initialization."""
        model = nn.Linear(10, 5)
        zero_opt = ZeroOptimizer(model, stage=2)
        assert zero_opt.model == model
        assert zero_opt.stage == 2
        assert zero_opt.partition_optimizer is True


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_get_device_info(self):
        """Test device info function."""
        info = get_device_info()
        assert "platform" in info
        assert "python_version" in info
        assert "torch_version" in info
        assert "cuda_available" in info
    
    def test_calculate_model_size(self):
        """Test model size calculation."""
        model = nn.Linear(10, 5)
        info = calculate_model_size(model)
        assert "total_parameters" in info
        assert "trainable_parameters" in info
        assert "parameter_memory_mb" in info
        assert info["total_parameters"] == 55


if __name__ == "__main__":
    pytest.main([__file__])
