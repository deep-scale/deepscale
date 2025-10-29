# DeepScale

A Python library for large-scale deep learning training across thousands of GPUs, designed for training massive models like Large Language Models (LLMs) and other billion-parameter architectures.

## Features

- **Massive Scale Training**: Support for training across thousands of GPUs
- **LLM Training**: Specialized utilities for Large Language Model training
- **Distributed Training**: Advanced distributed training strategies
- **Model Parallelism**: Pipeline and tensor parallelism for massive models
- **Memory Optimization**: Techniques for training billion-parameter models
- **Multi-Node Support**: Cross-node communication and synchronization
- **Gradient Scaling**: Efficient gradient accumulation and synchronization

## Installation

```bash
pip install deepscale
```

## Quick Start

### Large-Scale Training

```python
import torch
import torch.nn as nn
from deepscale import DistributedTrainer, ModelParallel, get_device_info

# Create a large model (e.g., transformer-based LLM)
class LargeLanguageModel(nn.Module):
    def __init__(self, vocab_size=50000, d_model=2048, num_layers=24):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead=16),
            num_layers=num_layers
        )
        self.output_proj = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.output_proj(x)

# Initialize large model
model = LargeLanguageModel()
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Set up distributed training across thousands of GPUs
num_gpus = 1000  # Example: 1000 GPUs
device_ids = list(range(num_gpus))

# Initialize distributed trainer
trainer = DistributedTrainer(
    model=model,
    num_gpus=num_gpus,
    batch_size_per_gpu=4,
    gradient_accumulation_steps=8
)

# Calculate effective batch size
effective_batch_size = trainer.calculate_effective_batch_size()
print(f"Effective batch size across {num_gpus} GPUs: {effective_batch_size:,}")

# Set up model parallelism for massive models
model_parallel = ModelParallel(model, device_ids=device_ids)
parallel_model = model_parallel.setup_model_parallel()

# Get cluster information
device_info = get_device_info()
print(f"CUDA devices available: {device_info['cuda_device_count']}")
```

### LLM Training at Scale

```python
from deepscale import LLMTrainer, PipelineParallel, GradientScaling

# Initialize LLM trainer for massive scale
trainer = LLMTrainer(
    model_size="70B",  # 70 billion parameters
    num_gpus=2048,     # 2048 GPUs
    sequence_length=4096,
    batch_size_per_gpu=1,
    gradient_checkpointing=True
)

# Set up pipeline parallelism for the LLM
pipeline_parallel = PipelineParallel(
    model=trainer.model,
    num_stages=64,  # Split across 64 pipeline stages
    micro_batch_size=1
)

# Configure gradient scaling for stability
gradient_scaler = GradientScaling(
    initial_scale=2**16,
    growth_factor=2.0,
    backoff_factor=0.5
)

# Start training
print(f"Training {trainer.model_size} parameter model on {trainer.num_gpus} GPUs")
print(f"Effective batch size: {trainer.effective_batch_size:,}")
print(f"Pipeline stages: {pipeline_parallel.num_stages}")
```

### Memory Optimization for Massive Models

```python
from deepscale import MemoryOptimizer, ModelSharding, ZeroOptimizer

# Optimize memory for billion-parameter models
memory_optimizer = MemoryOptimizer(
    model=trainer.model,
    offload_optimizer=True,
    offload_params=True,
    cpu_offload=True
)

# Shard model across multiple GPUs
model_sharding = ModelSharding(
    model=trainer.model,
    sharding_strategy="tensor_parallel",
    num_shards=8
)

# Zero redundancy optimizer
zero_optimizer = ZeroOptimizer(
    model=trainer.model,
    stage=2,  # ZeRO-2
    partition_optimizer=True,
    partition_gradients=True
)

# Calculate memory savings
memory_savings = memory_optimizer.calculate_memory_savings()
print(f"Memory savings: {memory_savings:.2f} GB")
```

## API Reference

### DistributedTrainer

- `__init__(model, num_gpus, batch_size_per_gpu, gradient_accumulation_steps)`: Initialize distributed trainer
- `calculate_effective_batch_size()`: Calculate effective batch size across all GPUs
- `setup_distributed_training()`: Set up distributed training environment
- `train_step(data, labels)`: Perform one training step
- `get_training_stats()`: Get training statistics

### LLMTrainer

- `__init__(model_size, num_gpus, sequence_length, batch_size_per_gpu)`: Initialize LLM trainer
- `setup_llm_training()`: Set up LLM-specific training configuration
- `train_on_batch(batch)`: Train on a single batch
- `get_model_size_info()`: Get detailed model size information

### PipelineParallel

- `__init__(model, num_stages, micro_batch_size)`: Initialize pipeline parallelism
- `setup_pipeline()`: Set up pipeline stages
- `forward_pipeline(input_data)`: Forward pass through pipeline
- `get_pipeline_efficiency()`: Calculate pipeline efficiency

### MemoryOptimizer

- `__init__(model, offload_optimizer, offload_params, cpu_offload)`: Initialize memory optimizer
- `optimize_memory()`: Apply memory optimizations
- `calculate_memory_savings()`: Calculate memory savings
- `get_memory_usage()`: Get current memory usage

### ModelSharding

- `__init__(model, sharding_strategy, num_shards)`: Initialize model sharding
- `shard_model()`: Shard model across devices
- `get_shard_info()`: Get sharding information

### ZeroOptimizer

- `__init__(model, stage, partition_optimizer, partition_gradients)`: Initialize ZeRO optimizer
- `setup_zero()`: Set up ZeRO optimization
- `get_zero_stats()`: Get ZeRO optimization statistics

## Requirements

- Python 3.8+
- PyTorch 1.9.0+
- NumPy 1.21.0+
- NCCL (for multi-GPU communication)
- CUDA 11.0+ (for GPU training)
- OpenMPI (for multi-node training)

## Development

To install the development dependencies:

```bash
pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```

Format code:

```bash
black src/
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Changelog

### 0.1.0 (2024-01-XX)
- Initial release
- Large-scale distributed training support
- LLM training utilities
- Pipeline and tensor parallelism
- Memory optimization for billion-parameter models
- Multi-node training capabilities
