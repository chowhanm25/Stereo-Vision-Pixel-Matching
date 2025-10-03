# API Reference

Comprehensive documentation for all classes, methods, and functions in the Stereo Vision Pixel Matching System.

## 🔧 Core Classes

### `StereoMatcher`

Main class implementing interactive stereo vision correspondence matching.

#### Constructor

```python
StereoMatcher(window_size=15, search_range=50, display_scale=1.0)
```

**Parameters:**
- `window_size` (int): Size of correlation window for ZNCC matching. Must be odd and >= 5.
- `search_range` (int): Search range along epipolar line in pixels. Must be >= 10.
- `display_scale` (float): Image display scaling factor. Range: [0.1, 2.0].

**Raises:**
- `ValueError`: If parameters are outside valid ranges.

**Example:**
```python
# Standard configuration
matcher = StereoMatcher()

# High-accuracy configuration
matcher = StereoMatcher(window_size=21, search_range=100)

# Fast processing configuration
matcher = StereoMatcher(window_size=11, search_range=30, display_scale=0.3)
```

#### Methods

##### `upload_images(self)`

Uploads stereo image pair from user with validation and preprocessing.

**Returns:** None

**Raises:**
- `RuntimeError`: If image upload, decoding, or processing fails.

**Side Effects:**
- Sets `self.img1`, `self.img2` (color images)
- Sets `self.gray1`, `self.gray2` (grayscale versions)
- Applies display scaling if configured

**Example:**
```python
matcher = StereoMatcher()
matcher.upload_images()  # Prompts for file upload
```

##### `detect_and_match_features(self)`

Detects SIFT features and performs FLANN-based matching with Lowe's ratio test.

**Returns:** None

**Raises:**
- `RuntimeError`: If insufficient features are detected or matching fails.

**Side Effects:**
- Sets `self.pts1`, `self.pts2` with matched keypoints
- Prints feature detection statistics

**Algorithm Details:**
1. SIFT feature detection with optimized parameters
2. FLANN k-NN matching (k=2)
3. Lowe's ratio test (threshold=0.7)
4. Validation of minimum match count

**Example:**
```python
matcher.detect_and_match_features()
# Output: 🔍 Detected 1247 features in left image
#         🔍 Detected 1156 features in right image
#         ✅ Found 342 reliable matches
```

##### `calculate_fundamental_matrix(self)`

Computes fundamental matrix using RANSAC for robust estimation.

**Returns:** None

**Raises:**
- `RuntimeError`: If fundamental matrix computation fails.

**Side Effects:**
- Sets `self.F` (3x3 fundamental matrix)
- Filters `self.pts1`, `self.pts2` to inliers only
- Prints computation statistics

**Algorithm Parameters:**
- Method: `cv2.FM_RANSAC`
- Threshold: 3.0 pixels
- Confidence: 99%
- Max iterations: 2000

**Example:**
```python
matcher.calculate_fundamental_matrix()
# Output: ✅ Fundamental matrix computed successfully
#         📈 Inlier ratio: 89.2% (305/342)
#         📐 F-matrix condition number: 1.23e+03
```

##### `compute_zncc(self, patch1, patch2)`

Computes Zero Normalized Cross Correlation between image patches.

**Parameters:**
- `patch1` (np.ndarray): First image patch
- `patch2` (np.ndarray): Second image patch

**Returns:**
- `float`: Correlation score in range [-1, 1], or -1 if computation fails

**Mathematical Formula:**
```
ZNCC = Σ((I1 - μ1)(I2 - μ2)) / √(Σ(I1 - μ1)² × Σ(I2 - μ2)²)
```

Where μ1, μ2 are the mean intensities of the patches.

**Example:**
```python
patch1 = gray_img[y1-7:y1+8, x1-7:x1+8]
patch2 = gray_img[y2-7:y2+8, x2-7:x2+8]
correlation = matcher.compute_zncc(patch1, patch2)
print(f"Correlation: {correlation:.3f}")
```

##### `find_correspondence(self, x, y)`

Finds best pixel correspondence along epipolar line using ZNCC.

**Parameters:**
- `x` (int): X coordinate in left image
- `y` (int): Y coordinate in left image

**Returns:**
- `tuple`: (best_x, best_y, epipolar_line_coeffs) or (None, None, line_coeffs)

**Algorithm Steps:**
1. Validate input coordinates
2. Compute epipolar line using fundamental matrix
3. Extract template patch from left image
4. Search along epipolar line in right image
5. Compute ZNCC for each candidate patch
6. Return best match and statistics

**Example:**
```python
best_x, best_y, line = matcher.find_correspondence(320, 240)
if best_x is not None:
    print(f"Best match at ({best_x}, {best_y})")
```

##### `draw_epipolar_line(self, img, line_coeffs)`

Draws epipolar line on image with proper bounds checking.

**Parameters:**
- `img` (np.ndarray): Input image
- `line_coeffs` (tuple): Line coefficients (a, b, c) from ax + by + c = 0

**Returns:**
- `np.ndarray`: Image copy with epipolar line drawn in green

**Features:**
- Automatic bounds clipping
- Handles vertical and horizontal lines
- 2-pixel line width for visibility

**Example:**
```python
line_coeffs = (0.001, -0.002, 0.5)
img_with_line = matcher.draw_epipolar_line(right_image, line_coeffs)
```

##### `interactive_matching(self)`

Sets up interactive display with click event handling.

**Returns:** None

**Features:**
- Side-by-side image display using IPython widgets
- Click event binding to left image
- Real-time visualization updates
- Error handling for click events

**UI Elements:**
- Red dot: Clicked pixel in left image
- White outline: Enhances dot visibility
- Green line: Epipolar line in right image
- Red cross: Best matching pixel
- Blue circle: Additional match highlighting

##### `run(self)`

Main execution pipeline orchestrating the complete workflow.

**Returns:** None

**Workflow:**
1. Display system information and version
2. Upload stereo image pair
3. Detect and match features
4. Compute fundamental matrix
5. Initialize interactive matching interface

**Error Handling:**
- Catches and reports all system errors
- Provides user-friendly error messages
- Suggests troubleshooting steps

---

## 🔢 Utility Functions

### Image Processing Utilities

```python
def validate_image(img):
    """
    Validate image format and properties
    
    Args:
        img: Input image array
    Returns:
        bool: True if valid, False otherwise
    """
    if img is None:
        return False
    if not isinstance(img, np.ndarray):
        return False
    if img.size == 0:
        return False
    if len(img.shape) not in [2, 3]:
        return False
    return True

def resize_image_proportional(img, max_dimension=800):
    """
    Resize image while maintaining aspect ratio
    
    Args:
        img: Input image
        max_dimension: Maximum width or height
    Returns:
        Resized image
    """
    height, width = img.shape[:2]
    if max(height, width) <= max_dimension:
        return img
    
    if height > width:
        scale = max_dimension / height
    else:
        scale = max_dimension / width
    
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
```

### Mathematical Utilities

```python
def normalize_fundamental_matrix(F):
    """
    Normalize fundamental matrix for numerical stability
    
    Args:
        F: 3x3 fundamental matrix
    Returns:
        Normalized fundamental matrix
    """
    if F is None or F[2,2] == 0:
        return F
    return F / F[2,2]

def compute_epipolar_distance(p1, p2, F):
    """
    Compute epipolar distance for point correspondence
    
    Args:
        p1: Point in left image [x, y, 1]
        p2: Point in right image [x, y, 1] 
        F: Fundamental matrix
    Returns:
        Epipolar distance in pixels
    """
    line = F.dot(p1)
    distance = abs(p2.dot(line)) / np.linalg.norm(line[:2])
    return distance
```

---

## 📈 Performance Characteristics

### Computational Complexity

| Operation | Time Complexity | Space Complexity | Typical Runtime |
|-----------|----------------|------------------|------------------|
| SIFT Detection | O(n × m × s) | O(k) | 0.5-2.0 sec |
| Feature Matching | O(k₁ × log k₂) | O(k₁ + k₂) | 0.1-0.5 sec |
| Fundamental Matrix | O(n × i) | O(n) | 0.1-0.3 sec |
| ZNCC Search | O(r × w²) | O(w²) | 0.1-0.5 sec |

Where:
- n, m: Image dimensions
- s: Number of scales in SIFT
- k, k₁, k₂: Number of keypoints/features
- i: RANSAC iterations
- r: Search range
- w: Window size

### Memory Usage

| Component | Memory Usage | Notes |
|-----------|--------------|-------|
| Image Storage | 4 × H × W × C | Color + grayscale versions |
| SIFT Features | 128 × K bytes | K = number of keypoints |
| Fundamental Matrix | 72 bytes | 3×3 double precision |
| Correlation Window | W² bytes | Temporary patch storage |

### Optimization Recommendations

1. **For large images (>2MP)**:
   ```python
   matcher = StereoMatcher(display_scale=0.3, search_range=30)
   ```

2. **For real-time applications**:
   ```python
   matcher = StereoMatcher(window_size=11, search_range=20)
   ```

3. **For high accuracy**:
   ```python
   matcher = StereoMatcher(window_size=21, search_range=100)
   ```

---

## 🔍 Error Codes and Handling

### Common Error Codes

| Error Code | Description | Resolution |
|------------|-------------|------------|
| `IMG_UPLOAD_FAILED` | Image upload or decoding failed | Check image format and size |
| `INSUFFICIENT_FEATURES` | Too few features detected | Use images with more texture |
| `FUNDAMENTAL_MATRIX_FAILED` | F-matrix computation failed | Check feature matches quality |
| `INVALID_COORDINATES` | Click coordinates out of bounds | Click within image boundaries |
| `ZNCC_COMPUTATION_ERROR` | Correlation computation failed | Check patch sizes and validity |

### Error Handling Best Practices

```python
try:
    matcher = StereoMatcher(window_size=15)
    matcher.run()
except ValueError as e:
    print(f"Parameter error: {e}")
except RuntimeError as e:
    print(f"Runtime error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
    # Log full traceback for debugging
    import traceback
    traceback.print_exc()
```

---

## 🔬 Integration Examples

### Custom Parameter Configuration

```python
# Configuration for different scenarios
configs = {
    'high_accuracy': {
        'window_size': 21,
        'search_range': 100,
        'display_scale': 0.8
    },
    'fast_processing': {
        'window_size': 11,
        'search_range': 30,
        'display_scale': 0.4
    },
    'large_images': {
        'window_size': 15,
        'search_range': 50,
        'display_scale': 0.2
    }
}

# Select configuration
config = configs['high_accuracy']
matcher = StereoMatcher(**config)
```

### Batch Processing Extension

```python
def process_stereo_sequence(image_pairs, matcher_config):
    """
    Process multiple stereo pairs in batch
    
    Args:
        image_pairs: List of (left_img, right_img) tuples
        matcher_config: StereoMatcher configuration
    Returns:
        List of processing results
    """
    results = []
    
    for i, (left, right) in enumerate(image_pairs):
        print(f"Processing pair {i+1}/{len(image_pairs)}")
        
        matcher = StereoMatcher(**matcher_config)
        # Set images directly instead of uploading
        matcher.img1 = left
        matcher.img2 = right
        matcher.gray1 = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        matcher.gray2 = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        
        # Process without interactive interface
        matcher.detect_and_match_features()
        matcher.calculate_fundamental_matrix()
        
        results.append({
            'pair_id': i,
            'features_left': len(matcher.pts1),
            'fundamental_matrix': matcher.F,
            'inlier_ratio': len(matcher.pts1) / len(matcher.pts2)
        })
    
    return results
```

---

## 📊 Debugging and Profiling

### Debug Mode

```python
class StereoMatcherDebug(StereoMatcher):
    """Extended version with debugging capabilities"""
    
    def __init__(self, *args, debug=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.debug = debug
        self.stats = {}
    
    def visualize_features(self, img, keypoints, title="Features"):
        """Visualize detected features on image"""
        if not self.debug:
            return
            
        img_features = cv2.drawKeypoints(
            img, keypoints, None, 
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        
        plt.figure(figsize=(10, 6))
        plt.imshow(cv2.cvtColor(img_features, cv2.COLOR_BGR2RGB))
        plt.title(f"{title}: {len(keypoints)} keypoints")
        plt.axis('off')
        plt.show()
    
    def plot_zncc_scores(self, scores, search_positions):
        """Plot ZNCC scores along epipolar line"""
        if not self.debug:
            return
            
        plt.figure(figsize=(12, 4))
        plt.plot(search_positions, scores, 'b-', linewidth=2)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        plt.xlabel('Position along epipolar line')
        plt.ylabel('ZNCC Score')
        plt.title('Correlation Scores Along Epipolar Line')
        plt.grid(True, alpha=0.3)
        
        # Mark best match
        best_idx = np.argmax(scores)
        plt.plot(search_positions[best_idx], scores[best_idx], 
                'ro', markersize=8, label=f'Best match: {scores[best_idx]:.3f}')
        plt.legend()
        plt.show()
```

### Performance Profiling

```python
import time
import cProfile
import pstats

def profile_stereo_matching():
    """Profile the complete stereo matching pipeline"""
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run stereo matching
    matcher = StereoMatcher()
    # ... run matching operations ...
    
    profiler.disable()
    
    # Analyze results
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)  # Top 10 functions
```

---

## 🔗 External Dependencies

### Required Packages

| Package | Version | Purpose |
|---------|---------|----------|
| `opencv-python` | ≥4.5.0 | Computer vision algorithms |
| `numpy` | ≥1.21.0 | Numerical computations |
| `ipywidgets` | ≥7.6.0 | Interactive Jupyter widgets |
| `ipyevents` | ≥0.9.0 | Event handling in notebooks |
| `matplotlib` | ≥3.5.0 | Plotting and visualization |

### Optional Packages

| Package | Purpose |
|---------|---------|
| `opencv-contrib-python` | Additional OpenCV features |
| `scipy` | Advanced scientific computing |
| `scikit-image` | Alternative image processing |
| `plotly` | Interactive visualizations |

---

This API reference provides complete documentation for integrating and extending the Stereo Vision Pixel Matching System. For additional examples and tutorials, see the [Examples Documentation](EXAMPLES.md).