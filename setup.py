#!/usr/bin/env python3
"""
Setup script for Stereo Vision Pixel Matching System
"""

from setuptools import setup, find_packages
import os

# Read README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements file
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="stereo-vision-pixel-matching",
    version="2.0.0",
    author="Munna Chowhan",
    author_email="chowhanm25@gmail.com",
    description="Interactive stereo vision system for pixel correspondence matching",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/chowhanm25/Stereo-Vision-Pixel-Matching",
    project_urls={
        "Bug Reports": "https://github.com/chowhanm25/Stereo-Vision-Pixel-Matching/issues",
        "Source": "https://github.com/chowhanm25/Stereo-Vision-Pixel-Matching",
        "Documentation": "https://github.com/chowhanm25/Stereo-Vision-Pixel-Matching/tree/main/docs",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Education",
        "Topic :: Multimedia :: Graphics :: Capture",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "black>=21.6.0",
            "flake8>=3.9.0",
            "isort>=5.9.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
            "myst-parser>=0.15.0",
        ],
        "gpu": [
            "opencv-contrib-python>=4.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "stereo-match=stereo_matcher.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
    keywords=[
        "computer-vision",
        "stereo-vision", 
        "epipolar-geometry",
        "feature-matching",
        "opencv",
        "image-processing",
        "correlation",
        "sift",
        "ransac",
        "interactive",
        "education",
        "jupyter",
        "colab"
    ],
    zip_safe=False,
)