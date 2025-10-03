# Installation Guide

This guide provides comprehensive instructions for setting up the Stereo Vision Pixel Matching System in different environments.

## 🎯 Quick Start Options

| Method | Difficulty | Setup Time | Best For |
|--------|------------|------------|----------|
| **Google Colab** | ⭐ Easy | 2 minutes | Beginners, quick testing |
| **Local Jupyter** | ⭐⭐ Medium | 5 minutes | Development, customization |
| **Docker** | ⭐⭐⭐ Advanced | 10 minutes | Production, consistent environments |

---

## 🌐 Google Colab Setup (Recommended)

### Step 1: Open the Notebook

1. **Click the Colab badge** in the main README
2. **Or visit directly**: [StereoVision Interactive Notebook](https://colab.research.google.com/github/chowhanm25/Stereo-Vision-Pixel-Matching/blob/main/StereoVision_Interactive.ipynb)

### Step 2: Run Setup Cell

```python
# This cell installs all required packages
!pip install opencv-python numpy ipywidgets ipyevents matplotlib
```

### Step 3: Execute Code Cells

1. **Run all cells** in sequence (Ctrl+F9)
2. **Upload stereo images** when prompted
3. **Start clicking** on the left image for correspondence matching

### Colab Advantages

- ✅ **No local installation** required
- ✅ **Pre-configured environment** with GPU support
- ✅ **Easy file upload** via browser
- ✅ **Instant sharing** with others
- ✅ **Free to use** with Google account

---

## 💻 Local Installation

### Prerequisites

#### System Requirements
- **Operating System**: Windows 10+, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **Python**: Version 3.8 or higher
- **Memory**: Minimum 4GB RAM (8GB recommended)
- **Storage**: At least 1GB free space

#### Required Software
- **Python**: [Download Python](https://python.org/downloads/)
- **Git**: [Download Git](https://git-scm.com/downloads)
- **Jupyter**: Will be installed with dependencies

### Installation Steps

#### 1. Clone the Repository

```bash
git clone https://github.com/chowhanm25/Stereo-Vision-Pixel-Matching.git
cd Stereo-Vision-Pixel-Matching
```

#### 2. Create Virtual Environment

**Windows:**
```bash
python -m venv stereo_env
stereo_env\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv stereo_env
source stereo_env/bin/activate
```

#### 3. Install Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install project dependencies
pip install -r requirements.txt

# Verify OpenCV installation
python -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"
```

#### 4. Install Jupyter Extensions

```bash
# Install Jupyter widgets
jupyter nbextension enable --py widgetsnbextension --sys-prefix

# Install ipyevents for click handling
jupyter nbextension install --py --symlink --user ipyevents
jupyter nbextension enable --py --user ipyevents
```

#### 5. Launch Jupyter Notebook

```bash
# Start Jupyter Lab (recommended)
jupyter lab

# Or start classic Jupyter Notebook
jupyter notebook
```

#### 6. Open the Interactive Notebook

- Navigate to `StereoVision_Interactive.ipynb`
- Run all cells to start the interactive system

---

## 🐳 Docker Installation

### Prerequisites
- **Docker**: [Install Docker](https://docs.docker.com/get-docker/)
- **Docker Compose**: [Install Docker Compose](https://docs.docker.com/compose/install/)

### Quick Docker Setup

#### 1. Create Dockerfile

```dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose Jupyter port
EXPOSE 8888

# Start Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
```

#### 2. Build and Run Container

```bash
# Build Docker image
docker build -t stereo-vision .

# Run container with port mapping
docker run -p 8888:8888 -v $(pwd):/app stereo-vision
```

#### 3. Access Jupyter Interface

- Open browser to `http://localhost:8888`
- Use the token displayed in terminal for authentication
- Navigate to `StereoVision_Interactive.ipynb`

### Docker Compose (Alternative)

```yaml
# docker-compose.yml
version: '3.8'

services:
  stereo-vision:
    build: .
    ports:
      - "8888:8888"
    volumes:
      - .:/app
      - jupyter_data:/home/jovyan/.jupyter
    environment:
      - JUPYTER_ENABLE_LAB=yes

volumes:
  jupyter_data:
```

```bash
# Start with Docker Compose
docker-compose up --build
```

---

## ⚙️ Configuration Options

### Environment Variables

Create a `.env` file for customization:

```env
# Display settings
DISPLAY_SCALE=0.5
WINDOW_SIZE=15
SEARCH_RANGE=50

# SIFT parameters
SIFT_FEATURES=5000
SIFT_CONTRAST_THRESHOLD=0.04
SIFT_EDGE_THRESHOLD=10

# FLANN parameters
FLANN_TREES=5
FLANN_CHECKS=50

# RANSAC parameters
RANSAC_THRESHOLD=3.0
RANSAC_CONFIDENCE=0.99
RANSAC_MAX_ITERS=2000
```

### Custom Configuration

```python
# Load configuration from file
from configparser import ConfigParser

config = ConfigParser()
config.read('config.ini')

matcher = StereoMatcher(
    window_size=config.getint('matching', 'window_size'),
    search_range=config.getint('matching', 'search_range'),
    display_scale=config.getfloat('display', 'scale')
)
```

---

## 🔧 Troubleshooting

### Common Issues

#### OpenCV Installation Problems

**Issue**: `ImportError: No module named 'cv2'`

**Solutions:**
```bash
# Try different OpenCV packages
pip uninstall opencv-python opencv-contrib-python
pip install opencv-python

# For additional features
pip install opencv-contrib-python

# If still failing, try system package manager
# Ubuntu/Debian:
sudo apt-get install python3-opencv

# macOS with Homebrew:
brew install opencv
```

#### Jupyter Widgets Not Working

**Issue**: Interactive widgets not displaying or responding

**Solutions:**
```bash
# Reinstall widgets
pip install --upgrade ipywidgets ipyevents

# Enable extensions
jupyter nbextension enable --py widgetsnbextension --sys-prefix
jupyter nbextension install --py --symlink --user ipyevents
jupyter nbextension enable --py --user ipyevents

# Restart Jupyter
jupyter lab --ip=0.0.0.0
```

#### Memory Issues with Large Images

**Issue**: System runs out of memory with high-resolution images

**Solutions:**
```python
# Reduce display scale
matcher = StereoMatcher(display_scale=0.3)  # Smaller images

# Reduce search range
matcher = StereoMatcher(search_range=30)    # Faster processing

# Resize images before processing
img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
```

#### Feature Detection Failures

**Issue**: Not enough features detected for fundamental matrix

**Solutions:**
```python
# Adjust SIFT parameters
sift = cv2.SIFT_create(
    nfeatures=10000,           # More features
    contrastThreshold=0.02,    # Lower threshold
    edgeThreshold=15           # Higher edge threshold
)

# Try different feature detector
orb = cv2.ORB_create(nfeatures=5000)
```

### Performance Optimization

#### For Large Images
```python
# Process at reduced resolution
matcher = StereoMatcher(display_scale=0.25)

# Use smaller correlation windows
matcher = StereoMatcher(window_size=11)

# Limit feature detection
sift = cv2.SIFT_create(nfeatures=1000)
```

#### For Real-time Applications
```python
# Cache feature detection results
self.features_cached = True

# Use faster but less accurate methods
fast = cv2.FastFeatureDetector_create()
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
```

### Debugging Tools

#### Feature Visualization
```python
def visualize_features(img, keypoints):
    """Draw detected keypoints on image"""
    img_with_keypoints = cv2.drawKeypoints(
        img, keypoints, None, 
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    return img_with_keypoints
```

#### Match Visualization
```python
def visualize_matches(img1, kp1, img2, kp2, matches):
    """Draw feature matches between images"""
    match_img = cv2.drawMatches(
        img1, kp1, img2, kp2, matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    return match_img
```

---

## 📊 Testing Installation

### Basic Functionality Test

```python
# Test OpenCV installation
import cv2
print(f"OpenCV version: {cv2.__version__}")

# Test NumPy
import numpy as np
print(f"NumPy version: {np.__version__}")

# Test widgets (for Jupyter)
import ipywidgets
print(f"IPywidgets version: {ipywidgets.__version__}")

# Test SIFT availability
sift = cv2.SIFT_create()
print("SIFT detector created successfully")
```

### Sample Image Test

```python
# Create sample stereo pair for testing
left_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
right_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

# Test basic functionality
matcher = StereoMatcher()
matcher.img1 = left_img
matcher.img2 = right_img
matcher.gray1 = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
matcher.gray2 = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

print("Basic functionality test passed!")
```

---

## 🔍 Platform-Specific Notes

### Windows

- **Use PowerShell or Command Prompt** for commands
- **Install Visual Studio Build Tools** if compilation errors occur
- **Consider Windows Subsystem for Linux (WSL)** for Linux-like experience

### macOS

- **Use Homebrew** for easier package management
- **Install Xcode Command Line Tools**: `xcode-select --install`
- **Consider using conda** for scientific computing packages

### Linux

- **Install system dependencies**:
  ```bash
  # Ubuntu/Debian
  sudo apt-get update
  sudo apt-get install python3-dev python3-pip
  sudo apt-get install libopencv-dev python3-opencv
  
  # CentOS/RHEL
  sudo yum install python3-devel python3-pip
  sudo yum install opencv-python3
  ```

- **For GUI support**:
  ```bash
  sudo apt-get install python3-tk
  ```

---

## 🔌 Advanced Setup

### Development Environment

For contributors and developers:

```bash
# Clone with development branch
git clone -b develop https://github.com/chowhanm25/Stereo-Vision-Pixel-Matching.git

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests to verify setup
python -m pytest tests/
```

### GPU Acceleration (Optional)

For CUDA-enabled OpenCV:

```bash
# Install CUDA toolkit first
# Then install OpenCV with CUDA support
pip uninstall opencv-python
pip install opencv-contrib-python

# Verify CUDA support
python -c "import cv2; print('CUDA devices:', cv2.cuda.getCudaEnabledDeviceCount())"
```

### Performance Monitoring

```python
# Install monitoring tools
pip install memory-profiler psutil

# Profile memory usage
@profile
def your_function():
    # Your code here
    pass

# Run with profiler
python -m memory_profiler your_script.py
```

---

## 📊 Verification

### Installation Verification Script

```python
#!/usr/bin/env python3
"""
Installation verification script for Stereo Vision Pixel Matching
"""

import sys
import importlib

def check_package(package_name, min_version=None):
    """Check if package is installed and meets version requirements"""
    try:
        module = importlib.import_module(package_name)
        version = getattr(module, '__version__', 'Unknown')
        print(f"✅ {package_name}: {version}")
        return True
    except ImportError:
        print(f"❌ {package_name}: Not installed")
        return False

def main():
    print("Verifying Stereo Vision Pixel Matching installation...")
    print("=" * 50)
    
    # Check Python version
    python_version = sys.version_info
    if python_version >= (3, 8):
        print(f"✅ Python: {python_version.major}.{python_version.minor}.{python_version.micro}")
    else:
        print(f"❌ Python: {python_version.major}.{python_version.minor}.{python_version.micro} (3.8+ required)")
    
    # Check required packages
    packages = [
        'cv2',
        'numpy',
        'matplotlib',
        'ipywidgets',
        'ipyevents'
    ]
    
    all_good = True
    for package in packages:
        if not check_package(package):
            all_good = False
    
    print("=" * 50)
    if all_good:
        print("🎉 Installation verification successful!")
        print("🚀 Ready to run stereo vision matching!")
    else:
        print("⚠️ Some packages are missing. Please install them using:")
        print("   pip install -r requirements.txt")

if __name__ == "__main__":
    main()
```

Save as `verify_installation.py` and run:
```bash
python verify_installation.py
```

---

## 🔄 Updating

### Update to Latest Version

```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade

# Clear Jupyter cache if needed
jupyter lab clean
```

### Version Management

```bash
# Check current version
git describe --tags

# Switch to specific version
git checkout v1.0.0

# Return to latest
git checkout main
```

---

## 📞 Support

### Getting Help

1. **Check this installation guide** thoroughly
2. **Review [troubleshooting section](#troubleshooting)**
3. **Search [existing issues](https://github.com/chowhanm25/Stereo-Vision-Pixel-Matching/issues)**
4. **Create a new issue** with detailed information
5. **Contact maintainer** directly if needed

### When Reporting Issues

Please include:
- **Operating system** and version
- **Python version**: `python --version`
- **OpenCV version**: `python -c "import cv2; print(cv2.__version__)"`
- **Error messages** (full traceback)
- **Steps to reproduce** the issue
- **Expected vs actual behavior**

### Useful Debugging Commands

```bash
# Check Python path
python -c "import sys; print(sys.path)"

# Check installed packages
pip list

# Check OpenCV build info
python -c "import cv2; print(cv2.getBuildInformation())"

# System information
python -c "import platform; print(platform.platform())"
```

---

## 🚀 Next Steps

After successful installation:

1. **Read the [Algorithm Details](ALGORITHM_DETAILS.md)** for technical background
2. **Try the interactive notebook** with your own stereo images
3. **Experiment with parameters** for different scenarios
4. **Contribute improvements** or report issues
5. **Share your results** with the community

**Happy stereo vision matching!** 🎉