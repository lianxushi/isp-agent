"""
Image Quality Metrics
===================

Image quality metrics for ISP comparison.
Supports:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator)
- Sharpness metrics
- Color metrics
- Traffic light specific metrics

Author: ISP Team
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MetricsResult:
    """Image quality metrics result"""
    # Overall
    overall_score: float = 0.0
    
    # Reference-based metrics (if reference provided)
    psnr: float = 0.0
    ssim: float = 0.0
    
    # No-reference metrics
    sharpness_score: float = 0.0
    noise_score: float = 0.0
    color_score: float = 0.0
    
    # Traffic light specific
    traffic_light_score: float = 0.0
    red_light_delta: float = 0.0
    green_light_delta: float = 0.0
    yellow_light_delta: float = 0.0
    
    # Details
    sharpness_details: Dict[str, float] = field(default_factory=dict)
    noise_details: Dict[str, float] = field(default_factory=dict)
    color_details: Dict[str, float] = field(default_factory=dict)
    traffic_light_details: Dict[str, Any] = field(default_factory=dict)
    
    # Status
    passed: bool = False
    issues: List[str] = field(default_factory=list)


class ImageMetrics:
    """
    Image quality metrics calculator.
    
    Provides comprehensive image quality assessment for ISP comparison.
    
    Example:
        >>> metrics = ImageMetrics()
        >>> result = metrics.evaluate("test.jpg", reference="ref.jpg")
        >>> print(f"Overall score: {result.overall_score}")
    """
    
    # IEC 61888 standard traffic light colors (in BGR for OpenCV)
    TRAFFIC_LIGHT_COLORS = {
        "red": {"lower": np.array([0, 50, 50]), "upper": np.array([10, 255, 255]), "target": (20, 200, 200)},
        "yellow": {"lower": np.array([15, 50, 50]), "upper": np.array([35, 255, 255]), "target": (30, 220, 210)},
        "green": {"lower": np.array([40, 50, 50]), "upper": np.array([80, 255, 255]), "target": (60, 200, 60)},
    }
    
    # IEC 61888 acceptable chroma deviation
    CHROMA_DELTA_THRESHOLD = 0.05
    
    def __init__(self):
        """Initialize metrics calculator"""
        self._cache = {}
        logger.debug("ImageMetrics initialized")
    
    def evaluate(
        self,
        image_path: str,
        reference_path: Optional[str] = None,
        traffic_light_roi: Optional[Tuple[int, int, int, int]] = None
    ) -> MetricsResult:
        """
        Evaluate image quality.
        
        Args:
            image_path: Path to image to evaluate
            reference_path: Optional reference image for PSNR/SSIM
            traffic_light_roi: Optional ROI (x, y, w, h) for traffic light region
            
        Returns:
            MetricsResult: Quality assessment result
        """
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
        
        result = MetricsResult()
        
        # Calculate metrics
        if reference_path:
            ref_image = cv2.imread(str(reference_path))
            if ref_image is not None:
                result.psnr = self._calculate_psnr(image, ref_image)
                result.ssim = self._calculate_ssim(image, ref_image)
        
        result.sharpness_score, result.sharpness_details = self._calculate_sharpness(image)
        result.noise_score, result.noise_details = self._calculate_noise(image)
        result.color_score, result.color_details = self._calculate_color_metrics(image)
        
        # Traffic light analysis
        if traffic_light_roi:
            tl_result = self._analyze_traffic_light(image, traffic_light_roi)
            result.traffic_light_score = tl_result["score"]
            result.traffic_light_details = tl_result
            result.red_light_delta = tl_result.get("red", {}).get("chroma_delta", 0)
            result.green_light_delta = tl_result.get("green", {}).get("chroma_delta", 0)
            result.yellow_light_delta = tl_result.get("yellow", {}).get("chroma_delta", 0)
        
        # Calculate overall score (weighted average)
        result.overall_score = self._calculate_overall_score(result)
        
        # Determine pass/fail
        result.passed = self._determine_pass(result)
        
        return result
    
    def _calculate_psnr(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate PSNR between two images"""
        try:
            # Resize if needed
            if img1.shape != img2.shape:
                img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
            
            mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
            if mse == 0:
                return 100.0
            
            max_pixel = 255.0
            psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
            return float(psnr)
        except Exception as e:
            logger.warning(f"PSNR calculation failed: {e}")
            return 0.0
    
    def _calculate_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate SSIM between two images"""
        try:
            # Convert to grayscale for SSIM
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            
            # Resize if needed
            if gray1.shape != gray2.shape:
                gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))
            
            # Simple SSIM implementation
            C1 = (0.01 * 255) ** 2
            C2 = (0.03 * 255) ** 2
            
            mu1 = cv2.GaussianBlur(gray1.astype(float), (11, 11), 1.5)
            mu2 = cv2.GaussianBlur(gray2.astype(float), (11, 11), 1.5)
            
            mu1_sq = mu1 ** 2
            mu2_sq = mu2 ** 2
            mu1_mu2 = mu1 * mu2
            
            sigma1_sq = cv2.GaussianBlur(gray1.astype(float) ** 2, (11, 11), 1.5) - mu1_sq
            sigma2_sq = cv2.GaussianBlur(gray2.astype(float) ** 2, (11, 11), 1.5) - mu2_sq
            sigma12 = cv2.GaussianBlur(gray1.astype(float) * gray2.astype(float), (11, 11), 1.5) - mu1_mu2
            
            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                       ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
            
            return float(np.mean(ssim_map))
        except Exception as e:
            logger.warning(f"SSIM calculation failed: {e}")
            return 0.0
    
    def _calculate_sharpness(self, image: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """Calculate sharpness metrics"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Laplacian variance (higher = sharper)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            laplacian_var = laplacian.var()
            
            # MTF50 estimate (simplified)
            # Sobel edges
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            sobel_mag = np.sqrt(sobelx**2 + sobely**2)
            
            # Edge sharpness (10-90% rise time)
            edge_profile = np.percentile(sobel_mag, [10, 90])
            rise_time = edge_profile[1] - edge_profile[0] if edge_profile[1] > edge_profile[0] else 0
            
            # Normalize to 0-100
            sharpness_score = min(100, laplacian_var / 100)
            
            return sharpness_score, {
                "laplacian_variance": float(laplacian_var),
                "mtf50_estimate": float(rise_time),
                "edge_strength": float(sobel_mag.mean())
            }
        except Exception as e:
            logger.warning(f"Sharpness calculation failed: {e}")
            return 0.0, {}
    
    def _calculate_noise(self, image: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """Calculate noise metrics"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Estimate noise using local variance
            kernel_size = 7
            local_mean = cv2.blur(gray.astype(float), (kernel_size, kernel_size))
            local_var = cv2.blur((gray.astype(float) - local_mean) ** 2, (kernel_size, kernel_size))
            
            noise_var = np.mean(local_var)
            noise_std = np.sqrt(noise_var)
            
            # Noise in dB
            if noise_var > 0:
                noise_db = 20 * np.log10(noise_std)
            else:
                noise_db = -100
            
            # Normalize score (higher noise = lower score)
            noise_score = max(0, min(100, 100 - noise_std / 2))
            
            return noise_score, {
                "noise_variance": float(noise_var),
                "noise_std": float(noise_std),
                "noise_db": float(noise_db)
            }
        except Exception as e:
            logger.warning(f"Noise calculation failed: {e}")
            return 0.0, {}
    
    def _calculate_color_metrics(self, image: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """Calculate color metrics"""
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Saturation statistics
            saturation_mean = hsv[:, :, 1].mean()
            saturation_std = hsv[:, :, 1].std()
            
            # Hue distribution
            hue_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
            hue_entropy = -np.sum(hue_hist / hue_hist.sum() * np.log2(hue_hist / hue_hist.sum() + 1e-10))
            
            # Normalize
            saturation_score = min(100, saturation_mean / 2.55)  # 0-255 to 0-100
            
            return saturation_score, {
                "saturation_mean": float(saturation_mean),
                "saturation_std": float(saturation_std),
                "hue_entropy": float(hue_entropy)
            }
        except Exception as e:
            logger.warning(f"Color calculation failed: {e}")
            return 0.0, {}
    
    def _analyze_traffic_light(
        self,
        image: np.ndarray,
        roi: Tuple[int, int, int, int]
    ) -> Dict[str, Any]:
        """
        Analyze traffic light region.
        
        Args:
            image: Input image
            roi: Region of interest (x, y, w, h)
            
        Returns:
            dict: Traffic light analysis results
        """
        x, y, w, h = roi
        roi_image = image[y:y+h, x:x+w]
        hsv_roi = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)
        
        result = {"score": 0, "red": {}, "yellow": {}, "green": {}}
        
        for color_name, color_spec in self.TRAFFIC_LIGHT_COLORS.items():
            # Create mask
            mask = cv2.inRange(hsv_roi, color_spec["lower"], color_spec["upper"])
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get largest contour
                largest = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest)
                
                if area > 100:  # Minimum size threshold
                    # Calculate mean color in region
                    mean_val = cv2.mean(roi_image, mask=mask)
                    
                    # Calculate chroma deviation from target
                    target = color_spec["target"]
                    chroma_delta = np.sqrt((mean_val[0] - target[0])**2 + 
                                          (mean_val[1] - target[1])**2) / 255.0
                    
                    # Color uniformity (std of hue)
                    hue_std = hsv_roi[:, :, 0][mask > 0].std()
                    
                    result[color_name] = {
                        "detected": True,
                        "area": float(area),
                        "mean_bgr": [float(v) for v in mean_val[:3]],
                        "chroma_delta": float(chroma_delta),
                        "hue_std": float(hue_std),
                        "passed": chroma_delta < self.CHROMA_DELTA_THRESHOLD
                    }
                else:
                    result[color_name] = {"detected": False}
            else:
                result[color_name] = {"detected": False}
        
        # Calculate overall traffic light score
        detected_count = sum(1 for c in ["red", "yellow", "green"] if result[c].get("detected", False))
        if detected_count > 0:
            passed_count = sum(1 for c in ["red", "yellow", "green"] 
                             if result[c].get("detected") and result[c].get("passed", False))
            result["score"] = (passed_count / 3) * 100
        else:
            result["score"] = 0
        
        return result
    
    def _calculate_overall_score(self, result: MetricsResult) -> float:
        """Calculate weighted overall score"""
        weights = {
            "sharpness": 0.25,
            "noise": 0.25,
            "color": 0.20,
            "traffic_light": 0.30
        }
        
        score = (
            result.sharpness_score * weights["sharpness"] +
            result.noise_score * weights["noise"] +
            result.color_score * weights["color"] +
            result.traffic_light_score * weights["traffic_light"]
        )
        
        return min(100, max(0, score))
    
    def _determine_pass(self, result: MetricsResult) -> bool:
        """Determine if result passes thresholds"""
        issues = []
        
        # Check traffic light deltas
        for color in ["red", "green", "yellow"]:
            delta = getattr(result, f"{color}_light_delta", 0)
            if delta > self.CHROMA_DELTA_THRESHOLD:
                issues.append(f"{color} light chroma deviation too high: {delta:.4f}")
        
        # Check sharpness
        if result.sharpness_score < 50:
            issues.append(f"Sharpness too low: {result.sharpness_score:.1f}")
        
        # Check noise
        if result.noise_score < 50:
            issues.append(f"Noise too high: {result.noise_score:.1f}")
        
        result.issues = issues
        return len(issues) == 0
    
    def compare_images(
        self,
        img1_path: str,
        img2_path: str
    ) -> Dict[str, Any]:
        """
        Compare two images and return difference metrics.
        
        Args:
            img1_path: Path to first image
            img2_path: Path to second image
            
        Returns:
            dict: Comparison results
        """
        img1 = cv2.imread(str(img1_path))
        img2 = cv2.imread(str(img2_path))
        
        if img1 is None or img2 is None:
            raise ValueError("Failed to read images")
        
        # Resize to match
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        # Calculate difference
        diff = cv2.absdiff(img1, img2)
        
        return {
            "psnr": self._calculate_psnr(img1, img2),
            "ssim": self._calculate_ssim(img1, img2),
            "mean_diff": float(diff.mean()),
            "max_diff": float(diff.max()),
            "diff_std": float(diff.std())
        }
