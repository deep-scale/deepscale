#!/usr/bin/env python3
"""
DeepScale - A Python library for deep learning scaling utilities.

This package provides utilities for scaling deep learning models,
including model parallelism, data parallelism, and optimization techniques.

Installation:
-------------
The package automatically installs PyTorch from CUDA 12.4 wheel index:
    pip install deepscale

PyTorch, torchvision, and torchaudio will be automatically installed from:
    https://download.pytorch.org/whl/cu124
"""

from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop
import subprocess
import sys

# Read the README file for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read version from __init__.py
def get_version():
    with open("src/deepscale/__init__.py", "r") as f:
        for line in f:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip('"').strip("'")
    return "0.1.0"


def install_pytorch_from_cuda_index():
    """Install PyTorch from CUDA 12.4 wheel index."""
    pytorch_index = "https://download.pytorch.org/whl/cu124"
    pytorch_packages = ["torch", "torchvision", "torchaudio"]
    
    print("\n" + "="*70)
    print("Installing PyTorch from CUDA 12.4 wheel index...")
    print("="*70 + "\n")
    
    try:
        # Install PyTorch packages from CUDA wheel index
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--upgrade",
            "--index-url", pytorch_index
        ] + pytorch_packages)
        print("\n✓ PyTorch installed successfully from CUDA 12.4 wheel index\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n⚠ Warning: Failed to install PyTorch from CUDA index: {e}")
        print("Falling back to standard PyPI installation...\n")
        # Fallback to standard PyPI installation
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "--upgrade"
            ] + pytorch_packages)
            print("✓ PyTorch installed from PyPI\n")
            return True
        except subprocess.CalledProcessError:
            print("⚠ Warning: Could not install PyTorch. Please install manually.")
            return False


class CustomInstallCommand(install):
    """Custom installation command that installs PyTorch from CUDA wheel index."""
    
    def run(self):
        """Run the installation."""
        install_pytorch_from_cuda_index()
        # Proceed with normal installation
        install.run(self)


class CustomDevelopCommand(develop):
    """Custom develop command that installs PyTorch from CUDA wheel index."""
    
    def run(self):
        """Run the development installation."""
        install_pytorch_from_cuda_index()
        # Proceed with normal development installation
        develop.run(self)

setup(
    name="deepscale",
    version=get_version(),
    author="Maruf Abbasi",
    author_email="maruf@marufabbasi.com",
    description="A Python library for large-scale deep learning training across thousands of GPUs for LLMs and other massive models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/deep-scale/deepscale",
    project_urls={
        "Bug Reports": "https://github.com/deep-scale/deepscale/issues",
        "Source": "https://github.com/deep-scale/deepscale",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    cmdclass={
        "install": CustomInstallCommand,
        "develop": CustomDevelopCommand,
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "tqdm>=4.62.0",
        "psutil>=5.8.0",
        "rich>=10.0.0",
        "pyyaml>=5.4.0",
        "tensorboard>=2.8.0",
        "nvidia-ml-py3>=7.352.0",
        "deepspeed>=0.9.0",
        # Note: PyTorch packages are automatically installed from CUDA 12.4 wheel index
        # via CustomInstallCommand before these dependencies are processed
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "torchaudio>=0.9.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
        ],
        "perf": [
            # Optional but useful for performance optimization
            "packaging>=21.0",
            "ninja>=1.10.0",
            "triton>=2.0.0",
        ],
    },
    keywords="large-scale training, distributed training, LLM training, multi-GPU, thousands of GPUs, massive models, deep learning scaling",
    include_package_data=True,
    zip_safe=False,
)
