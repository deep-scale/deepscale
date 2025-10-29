#!/usr/bin/env python3
"""
DeepScale - A Python library for deep learning scaling utilities.

This package provides utilities for scaling deep learning models,
including model parallelism, data parallelism, and optimization techniques.
"""

from setuptools import setup, find_packages

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
        "torch>=1.9.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
        ],
    },
    keywords="large-scale training, distributed training, LLM training, multi-GPU, thousands of GPUs, massive models, deep learning scaling",
    include_package_data=True,
    zip_safe=False,
)
