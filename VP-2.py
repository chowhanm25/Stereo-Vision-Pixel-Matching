#!/usr/bin/env python3
"""
Stereo Vision Pixel Matching System

Advanced computer vision application for interactive stereo correspondence matching.
Implements epipolar geometry visualization and ZNCC-based pixel matching.

Author: Munna Chowhan
Email: chowhanm25@gmail.com
Date: October 2025
Version: 2.0

Usage:
    matcher = StereoMatcher(window_size=15, search_range=50, display_scale=0.5)
    matcher.run()
"""

import cv2
import numpy as np
from google.colab import files
from IPython.display import clear_output, display
from ipywidgets import Image as WImage, HBox
from ipyevents import Event
import warnings
warnings.filterwarnings('ignore')


class StereoMatcher:
    """
    Interactive stereo vision system for pixel correspondence matching.
    
    This class implements a complete stereo vision pipeline including:
    - SIFT feature detection and FLANN matching
    - Robust fundamental matrix computation using RANSAC
    - Interactive pixel selection with click events
    - ZNCC-based correspondence search along epipolar lines
    - Real-time visualization of epipolar geometry
    
    Attributes:
        window_size (int): Size of correlation window for ZNCC matching
        search_range (int): Search range along epipolar line in pixels
        display_scale (float): Image display scaling factor
        img1, img2 (np.ndarray): Original color stereo images
        gray1, gray2 (np.ndarray): Grayscale versions for processing
        F (np.ndarray): Fundamental matrix (3x3)
        pts1, pts2 (np.ndarray): Matched keypoints after RANSAC filtering
    """
    
    def __init__(self, window_size=15, search_range=50, display_scale=1.0):
        """
        Initialize StereoMatcher with configurable parameters.
        
        Args:
            window_size (int): Size of correlation window (must be odd, >= 5)
            search_range (int): Search range along epipolar line (>= 10)
            display_scale (float): Image scaling factor (0.1 to 2.0)
            
        Raises:
            ValueError: If parameters are out of valid ranges
        """
        # Validate parameters
        if window_size < 5 or window_size % 2 == 0:
            raise ValueError("Window size must be odd and >= 5")
        if search_range < 10:
            raise ValueError("Search range must be >= 10")
        if not (0.1 <= display_scale <= 2.0):
            raise ValueError("Display scale must be between 0.1 and 2.0")
            
        self.window_size = window_size
        self.search_range = search_range
        self.display_scale = display_scale
        
        # Initialize attributes
        self.img1 = None
        self.img2 = None
        self.gray1 = None
        self.gray2 = None
        self.F = None
        self.pts1 = None
        self.pts2 = None
        
        print(f"🎯 StereoMatcher initialized:")
        print(f"   Window size: {window_size}px")
        print(f"   Search range: {search_range}px")
        print(f"   Display scale: {display_scale:.1f}x")
    
    def upload_images(self):
        """
        Upload stereo image pair from user with validation.
        
        Raises:
            RuntimeError: If image upload or processing fails
        """
        try:
            # Upload LEFT image
            print("🖼️ Upload LEFT stereo image:")
            up1 = files.upload()
            clear_output()
            
            if not up1:
                raise RuntimeError("No left image uploaded")
                
            fn1 = next(iter(up1))
            img1 = cv2.imdecode(np.frombuffer(up1[fn1], np.uint8), cv2.IMREAD_COLOR)
            
            if img1 is None:
                raise RuntimeError("Failed to decode left image")
            
            # Upload RIGHT image
            print("🖼️ Upload RIGHT stereo image:")
            up2 = files.upload()
            clear_output()
            
            if not up2:
                raise RuntimeError("No right image uploaded")
                
            fn2 = next(iter(up2))
            img2 = cv2.imdecode(np.frombuffer(up2[fn2], np.uint8), cv2.IMREAD_COLOR)
            
            if img2 is None:
                raise RuntimeError("Failed to decode right image")
            
            # Resize images for display if needed
            if self.display_scale != 1.0:
                img1 = cv2.resize(img1, (0,0), fx=self.display_scale, 
                                 fy=self.display_scale, interpolation=cv2.INTER_AREA)
                img2 = cv2.resize(img2, (0,0), fx=self.display_scale, 
                                 fy=self.display_scale, interpolation=cv2.INTER_AREA)
            
            # Store images and convert to grayscale
            self.img1 = img1
            self.gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            self.img2 = img2
            self.gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            
            print(f"✅ Left image loaded: {self.img1.shape}")
            print(f"✅ Right image loaded: {self.img2.shape}")
            
        except Exception as e:
            raise RuntimeError(f"Image upload failed: {str(e)}")
    
    def detect_and_match_features(self):
        """
        Detect SIFT features and match using FLANN with Lowe's ratio test.
        
        Raises:
            RuntimeError: If feature detection or matching fails
        """
        try:
            # Initialize SIFT detector
            sift = cv2.SIFT_create(
                nfeatures=5000,
                contrastThreshold=0.04,
                edgeThreshold=10
            )
            
            # Detect features in both images
            kp1, desc1 = sift.detectAndCompute(self.gray1, None)
            kp2, desc2 = sift.detectAndCompute(self.gray2, None)
            
            if len(kp1) < 8 or len(kp2) < 8:
                raise RuntimeError("Insufficient features detected (need at least 8 in each image)")
            
            print(f"🔍 Detected {len(kp1)} features in left image")
            print(f"🔍 Detected {len(kp2)} features in right image")
            
            # FLANN-based matching
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            raw_matches = flann.knnMatch(desc1, desc2, k=2)
            
            # Apply Lowe's ratio test
            good_matches = []
            for match_pair in raw_matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)
            
            if len(good_matches) < 8:
                raise RuntimeError(f"Insufficient good matches found: {len(good_matches)} (need at least 8)")
            
            # Extract matching points
            self.pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
            self.pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
            
            print(f"✅ Found {len(good_matches)} reliable matches")
            
        except Exception as e:
            raise RuntimeError(f"Feature detection/matching failed: {str(e)}")
    
    def calculate_fundamental_matrix(self):
        """
        Compute fundamental matrix using RANSAC for robust estimation.
        
        Raises:
            RuntimeError: If fundamental matrix computation fails
        """
        try:
            # Compute fundamental matrix with RANSAC
            F, mask = cv2.findFundamentalMat(
                self.pts1, self.pts2, 
                cv2.FM_RANSAC,
                ransacReprojThreshold=3.0,
                confidence=0.99,
                maxIters=2000
            )
            
            if F is None:
                raise RuntimeError("Failed to compute fundamental matrix")
            
            # Normalize matrix for numerical stability
            self.F = F / F[2,2]
            
            # Filter points to keep only inliers
            inliers = mask.ravel() == 1
            self.pts1 = self.pts1[inliers]
            self.pts2 = self.pts2[inliers]
            
            inlier_ratio = np.sum(inliers) / len(inliers) * 100
            
            print(f"✅ Fundamental matrix computed successfully")
            print(f"📈 Inlier ratio: {inlier_ratio:.1f}% ({np.sum(inliers)}/{len(inliers)})")
            print(f"📐 F-matrix condition number: {np.linalg.cond(self.F):.2e}")
            
        except Exception as e:
            raise RuntimeError(f"Fundamental matrix computation failed: {str(e)}")
    
    def compute_zncc(self, patch1, patch2):
        """
        Compute Zero Normalized Cross Correlation between two patches.
        
        ZNCC is robust to illumination changes and provides correlation
        scores in the range [-1, 1], where 1 indicates perfect correlation.
        
        Args:
            patch1, patch2 (np.ndarray): Image patches for comparison
            
        Returns:
            float: ZNCC correlation score, -1 if computation fails
        """
        try:
            if patch1.shape != patch2.shape or patch1.size == 0:
                return -1
            
            # Convert to float for numerical precision
            A = patch1.astype(np.float32)
            B = patch2.astype(np.float32)
            
            # Compute means
            mean_A = A.mean()
            mean_B = B.mean()
            
            # Zero-mean patches
            A_centered = A - mean_A
            B_centered = B - mean_B
            
            # Compute correlation
            numerator = (A_centered * B_centered).sum()
            
            # Compute standard deviations
            std_A = np.sqrt((A_centered * A_centered).sum())
            std_B = np.sqrt((B_centered * B_centered).sum())
            
            denominator = std_A * std_B
            
            # Return correlation or -1 for invalid patches
            return numerator / denominator if denominator > 1e-5 else -1
            
        except Exception:
            return -1
    
    def find_correspondence(self, x, y):
        """
        Find best pixel correspondence along epipolar line using ZNCC.
        
        Args:
            x, y (int): Pixel coordinates in left image
            
        Returns:
            tuple: (best_x, best_y, epipolar_line_coeffs) or (None, None, line)
        """
        try:
            # Validate input coordinates
            if not (0 <= x < self.gray1.shape[1] and 0 <= y < self.gray1.shape[0]):
                print(f"⚠️ Click coordinates ({x}, {y}) are out of bounds")
                return None, None, None
            
            # Compute epipolar line in right image
            point_homogeneous = np.array([x, y, 1.0], dtype=np.float32)
            epipolar_line = self.F.dot(point_homogeneous)
            
            # Normalize line equation for numerical stability
            a, b, c = epipolar_line
            norm = np.hypot(a, b)
            if norm > 1e-6:
                a, b, c = a/norm, b/norm, c/norm
            else:
                print("⚠️ Degenerate epipolar line")
                return None, None, (a, b, c)
            
            # Extract patch from left image
            hw = self.window_size // 2
            y_min = max(0, y - hw)
            y_max = min(self.gray1.shape[0], y + hw + 1)
            x_min = max(0, x - hw)
            x_max = min(self.gray1.shape[1], x + hw + 1)
            
            left_patch = self.gray1[y_min:y_max, x_min:x_max]
            
            if left_patch.size == 0:
                print("⚠️ Empty left patch extracted")
                return None, None, (a, b, c)
            
            # Search along epipolar line in right image
            best_score = -1
            best_x, best_y = None, None
            height, width = self.gray2.shape
            
            candidates_tested = 0
            valid_patches = 0
            
            for dx in range(-self.search_range, self.search_range + 1):
                x_candidate = x + dx
                
                # Check horizontal bounds
                if not (hw <= x_candidate < width - hw):
                    continue
                
                # Compute y coordinate on epipolar line
                if abs(b) > 1e-6:
                    y_candidate = int(-(a * x_candidate + c) / b)
                else:
                    continue  # Skip vertical lines
                
                # Check vertical bounds
                if not (hw <= y_candidate < height - hw):
                    continue
                
                candidates_tested += 1
                
                # Extract patch from right image
                right_patch = self.gray2[
                    y_candidate - hw:y_candidate + hw + 1,
                    x_candidate - hw:x_candidate + hw + 1
                ]
                
                # Ensure patches have the same size
                if right_patch.shape == left_patch.shape:
                    valid_patches += 1
                    score = self.compute_zncc(left_patch, right_patch)
                    
                    if score > best_score:
                        best_score = score
                        best_x, best_y = x_candidate, y_candidate
            
            # Print search statistics
            print(f"🔍 Searched {candidates_tested} candidates, {valid_patches} valid patches")
            if best_x is not None:
                print(f"🎯 Best match: ({best_x}, {best_y}) with ZNCC = {best_score:.3f}")
            else:
                print("❌ No valid correspondence found")
            
            return best_x, best_y, (a, b, c)
            
        except Exception as e:
            print(f"❌ Error in correspondence search: {str(e)}")
            return None, None, None
    
    def draw_epipolar_line(self, img, line_coeffs):
        """
        Draw epipolar line on image.
        
        Args:
            img (np.ndarray): Input image
            line_coeffs (tuple): Epipolar line coefficients (a, b, c)
            
        Returns:
            np.ndarray: Image with epipolar line drawn
        """
        if line_coeffs is None:
            return img.copy()
            
        a, b, c = line_coeffs
        height, width = img.shape[:2]
        
        # Calculate line endpoints
        if abs(b) > 1e-6:
            x0, y0 = 0, int(-c / b)
            x1, y1 = width - 1, int(-(a * (width - 1) + c) / b)
        else:
            # Vertical line
            x0, y0 = int(-c / a), 0
            x1, y1 = int(-c / a), height - 1
        
        # Clip line to image bounds
        x0 = max(0, min(width - 1, x0))
        x1 = max(0, min(width - 1, x1))
        y0 = max(0, min(height - 1, y0))
        y1 = max(0, min(height - 1, y1))
        
        # Draw line
        output = img.copy()
        cv2.line(output, (x0, y0), (x1, y1), (0, 255, 0), 2)
        
        return output
    
    def interactive_matching(self):
        """
        Setup interactive display with click event handling.
        
        Creates side-by-side image display where clicking on the left
        image triggers correspondence search and visualization.
        """
        def to_png_bytes(img):
            """Convert OpenCV image to PNG bytes for widget display"""
            if img is None:
                return b''
            _, buffer = cv2.imencode('.png', img)
            return buffer.tobytes()
        
        # Create interactive image widgets
        widget_left = WImage(value=to_png_bytes(self.img1), format='png')
        widget_right = WImage(value=to_png_bytes(self.img2), format='png')
        
        # Display images side by side
        display(HBox([widget_left, widget_right]))
        
        def on_click(event):
            """
            Handle click events on left image.
            
            Args:
                event: Click event with relativeX and relativeY coordinates
            """
            try:
                x = int(event['relativeX'])
                y = int(event['relativeY'])
                
                print(f"\n🔍 Clicked at ({x}, {y}) in left image")
                
                # Reset images to original state
                widget_left.value = to_png_bytes(self.img1)
                widget_right.value = to_png_bytes(self.img2)
                
                # Mark clicked point on left image
                left_marked = self.img1.copy()
                cv2.circle(left_marked, (x, y), 5, (0, 0, 255), -1)  # Red dot
                cv2.circle(left_marked, (x, y), 8, (255, 255, 255), 2)  # White outline
                widget_left.value = to_png_bytes(left_marked)
                
                # Find correspondence
                best_x, best_y, epipolar_line = self.find_correspondence(x, y)
                
                # Draw epipolar line on right image
                right_with_line = self.draw_epipolar_line(self.img2, epipolar_line)
                
                # Mark best matching point if found
                if best_x is not None and best_y is not None:
                    cv2.drawMarker(right_with_line, (best_x, best_y), (0, 0, 255),
                                  markerType=cv2.MARKER_CROSS, markerSize=20, thickness=3)
                    
                    # Add circle around match for better visibility
                    cv2.circle(right_with_line, (best_x, best_y), 12, (255, 0, 0), 2)
                
                # Update right image display
                widget_right.value = to_png_bytes(right_with_line)
                
            except Exception as e:
                print(f"❌ Error handling click: {str(e)}")
        
        # Setup click event listener
        click_event = Event(source=widget_left, watched_events=['click'])
        click_event.on_dom_event(on_click)
        
        print("✨ Interactive matching ready!")
        print("👆 Click anywhere on the LEFT image to find correspondence")
        print("   • Red dot: Selected pixel")
        print("   • Green line: Epipolar line in right image")
        print("   • Red cross: Best matching pixel")
    
    def run(self):
        """
        Main execution pipeline for stereo vision matching.
        
        Orchestrates the complete workflow from image upload to interactive matching.
        """
        try:
            print("🎯 Stereo Vision Pixel Matching System v2.0")
            print("=" * 55)
            print("🎓 Advanced Computer Vision Assignment")
            print("🔬 Interactive Epipolar Geometry Visualization")
            print("=" * 55)
            
            # Step 1: Image upload
            print("\n📁 Step 1: Upload stereo image pair")
            self.upload_images()
            
            # Step 2: Feature detection and matching
            print("\n🔍 Step 2: Feature detection and matching")
            self.detect_and_match_features()
            
            # Step 3: Fundamental matrix computation
            print("\n🧮 Step 3: Fundamental matrix computation")
            self.calculate_fundamental_matrix()
            
            # Step 4: Interactive matching interface
            print("\n🎯 Step 4: Interactive correspondence matching")
            print("-" * 40)
            self.interactive_matching()
            
        except Exception as e:
            print(f"❌ System error: {str(e)}")
            print("📞 Please check your images and try again")


# Example usage and demonstration
if __name__ == "__main__":
    print("🚀 Starting Stereo Vision Pixel Matching Demo")
    print("\nInitializing with optimal parameters...")
    
    # Create matcher with balanced parameters
    matcher = StereoMatcher(
        window_size=15,        # Good balance of accuracy and speed
        search_range=50,       # Reasonable search range
        display_scale=0.5      # Fit images on screen
    )
    
    # Run the complete system
    matcher.run()
    
    print("\n🎉 System ready! Upload your stereo images and start clicking!")
    print("🔍 Pro tip: Click on textured regions for better matching results")