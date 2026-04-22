"""
Contour Evaluator
=================

Evaluates contour/sharpness quality for ISP comparison.
Implements:
- Edge sharpness (MTF50 estimation, 10%-90% rise time)
- Edge completeness (continuity, breakage detection)
- Contour accuracy (geometric distortion)

Author: ISP Team
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SharpnessResult:
    """Edge sharpness analysis result"""
    mtf50_estimate: float = 0.0  # MTF50 frequency (cycles/pixel)
    rise_time_10_90: float = 0.0  # 10%-90% rise time in pixels
    edge_strength: float = 0.0   # Mean edge gradient magnitude
    laplacian_variance: float = 0.0
    score: float = 0.0  # 0-100
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "mtf50_estimate": self.mtf50_estimate,
            "rise_time_10_90": self.rise_time_10_90,
            "edge_strength": self.edge_strength,
            "laplacian_variance": self.laplacian_variance,
            "score": self.score
        }


@dataclass
class CompletenessResult:
    """Edge completeness analysis result"""
    edge_continuity: float = 0.0  # 0-1, higher is better
    breakage_count: int = 0
    breakage_locations: List[Tuple[int, int]] = field(default_factory=list)
    broken_edge_ratio: float = 0.0  # Ratio of broken edges to total
    score: float = 0.0  # 0-100
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "edge_continuity": self.edge_continuity,
            "breakage_count": self.breakage_count,
            "breakage_locations": self.breakage_locations,
            "broken_edge_ratio": self.broken_edge_ratio,
            "score": self.score
        }


@dataclass
class AccuracyResult:
    """Contour geometric accuracy result"""
    geometric_distortion: float = 0.0  # Distortion ratio
    contour_deviation_px: float = 0.0  # Mean pixel deviation from reference
    aspect_ratio_error: float = 0.0     # Aspect ratio error
    score: float = 0.0  # 0-100
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "geometric_distortion": self.geometric_distortion,
            "contour_deviation_px": self.contour_deviation_px,
            "aspect_ratio_error": self.aspect_ratio_error,
            "score": self.score
        }


@dataclass
class ContourEvaluationResult:
    """Complete contour evaluation result"""
    sharpness: SharpnessResult = field(default_factory=SharpnessResult)
    completeness: CompletenessResult = field(default_factory=CompletenessResult)
    accuracy: AccuracyResult = field(default_factory=AccuracyResult)
    
    overall_score: float = 0.0
    overall_passed: bool = False
    issues: List[Dict[str, Any]] = field(default_factory=list)
    
    # Edge data for visualization
    edge_image: Optional[np.ndarray] = None  # Edge map visualization
    mtf_curve: Optional[np.ndarray] = None    # MTF curve data
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "sharpness": self.sharpness.to_dict(),
            "completeness": self.completeness.to_dict(),
            "accuracy": self.accuracy.to_dict(),
            "overall_score": self.overall_score,
            "overall_passed": self.overall_passed,
            "issues": self.issues
        }


class ContourEvaluator:
    """
    Contour/sharpness evaluator for ISP comparison.
    
    Evaluates edge quality through three dimensions:
    1. Sharpness: MTF50 estimation, rise time, edge strength
    2. Completeness: Edge continuity, breakage detection
    3. Accuracy: Geometric distortion, contour deviation
    
    Example:
        >>> evaluator = ContourEvaluator()
        >>> result = evaluator.evaluate(image, roi=(x, y, w, h))
        >>> print(f"Overall score: {result.overall_score}")
        >>> print(f"Sharpness MTF50: {result.sharpness.mtf50_estimate:.3f}")
    """
    
    # Sharpness thresholds
    MIN_SHARPNESS_SCORE = 50.0
    MIN_LAPLACIAN_VARIANCE = 100.0
    
    # Completeness thresholds
    MAX_EDGE_BREAKAGE = 5
    MIN_EDGE_CONTINUITY = 0.85
    
    # Accuracy thresholds
    MAX_GEOMETRIC_DISTORTION = 0.05  # 5%
    MAX_CONTOUR_DEVIATION_PX = 3.0
    
    def __init__(self):
        """Initialize contour evaluator"""
        logger.debug("ContourEvaluator initialized")
    
    def evaluate(
        self,
        image: np.ndarray,
        roi: Optional[Tuple[int, int, int, int]] = None,
        reference_edges: Optional[np.ndarray] = None
    ) -> ContourEvaluationResult:
        """
        Evaluate contour quality.
        
        Args:
            image: Input image (BGR or grayscale)
            roi: Optional region of interest (x, y, w, h)
            reference_edges: Optional reference edge map for accuracy comparison
            
        Returns:
            ContourEvaluationResult: Evaluation result
        """
        result = ContourEvaluationResult()
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Extract ROI
        if roi:
            x, y, w, h = roi
            gray = gray[y:y+h, x:x+w]
        
        # 1. Sharpness analysis
        result.sharpness = self._analyze_sharpness(gray)
        
        # 2. Completeness analysis
        result.completeness = self._analyze_completeness(gray)
        
        # 3. Accuracy analysis
        result.accuracy = self._analyze_accuracy(gray, reference_edges)
        
        # Calculate overall score
        result.overall_score = self._calculate_overall_score(result)
        result.overall_passed = self._check_pass(result)
        
        return result
    
    def _analyze_sharpness(self, gray: np.ndarray) -> SharpnessResult:
        """Analyze edge sharpness using multiple methods"""
        result = SharpnessResult()
        
        # Laplacian variance (higher = sharper)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        result.laplacian_variance = float(laplacian.var())
        
        # Sobel edge detection for MTF estimation
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = np.sqrt(sobelx**2 + sobely**2)
        result.edge_strength = float(sobel_mag.mean())
        
        # MTF50 estimation using edge spread function
        # Find strong edges
        threshold = np.percentile(sobel_mag, 95)
        edge_mask = (sobel_mag > threshold).astype(np.uint8)
        
        # Get edge profiles perpendicular to edge direction
        mtf_data = self._compute_mtf_from_edges(gray, edge_mask)
        result.mtf50_estimate = mtf_data["mtf50"]
        
        # 10%-90% rise time
        result.rise_time_10_90 = mtf_data["rise_time"]
        
        # Score calculation
        # Based on laplacian variance (normalized)
        laplacian_score = min(100, result.laplacian_variance / 10)
        
        # Based on edge strength
        strength_score = min(100, result.edge_strength * 2)
        
        # Based on MTF50 (normalized to 0.5 cycles/pixel max)
        mtf_score = min(100, result.mtf50_estimate * 200)
        
        # Weighted average
        result.score = laplacian_score * 0.4 + strength_score * 0.3 + mtf_score * 0.3
        
        return result
    
    def _compute_mtf_from_edges(
        self,
        gray: np.ndarray,
        edge_mask: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute MTF50 from edge spread function.
        
        Returns:
            dict with mtf50 and rise_time
        """
        # Find edge pixels
        edge_points = np.where(edge_mask > 0)
        
        if len(edge_points[0]) < 100:
            # Not enough edge points
            return {"mtf50": 0.0, "rise_time": 10.0}
        
        # Sample edge profiles
        profiles = []
        
        # For each edge point, sample perpendicular to edge direction
        for i in range(min(1000, len(edge_points[0]))):
            y, x = edge_points[0][i], edge_points[1][i]
            
            # Get local gradient direction
            if y > 0 and y < gray.shape[0] - 1 and x > 0 and x < gray.shape[1] - 1:
                gy = gray[y+1, x] - gray[y-1, x]
                gx = gray[y, x+1] - gray[y, x-1]
                
                if abs(gy) + abs(gx) > 0:
                    # Sample profile perpendicular to edge
                    dx = -gy
                    dy = gx
                    length = int(np.sqrt(dx**2 + dy**2))
                    if length > 0:
                        dx = dx / length * 3  # 3 pixels on each side
                        dy = dy / length * 3
                        
                        # Collect profile
                        profile = []
                        for t in range(-5, 6):
                            px = int(x + t * dx)
                            py = int(y + t * dy)
                            if 0 <= px < gray.shape[1] and 0 <= py < gray.shape[0]:
                                profile.append(gray[py, px])
                            else:
                                profile.append(gray[y, x])
                        
                        if len(profile) == 11:
                            profiles.append(profile)
        
        if not profiles:
            return {"mtf50": 0.0, "rise_time": 10.0}
        
        # Average all profiles
        avg_profile = np.mean(profiles, axis=0)
        
        # Normalize
        profile_min = avg_profile.min()
        profile_max = avg_profile.max()
        if profile_max > profile_min:
            norm_profile = (avg_profile - profile_min) / (profile_max - profile_min)
        else:
            norm_profile = np.zeros_like(avg_profile)
        
        # Find 10% and 90% points
        threshold_10 = 0.1
        threshold_90 = 0.9
        
        idx_10 = np.where(norm_profile >= threshold_10)[0]
        idx_90 = np.where(norm_profile >= threshold_90)[0]
        
        rise_time = 10.0
        if len(idx_10) > 0 and len(idx_90) > 0:
            rise_time = (idx_90[0] - idx_10[0]) * 0.5  # Convert to pixel units (approx)
        
        # MTF50: frequency where MTF drops to 50%
        # Approximate using the edge profile width
        # MTF50 ≈ 0.5 / (rise_time in pixels)
        mtf50 = 0.5 / max(rise_time, 0.1)
        
        return {"mtf50": float(mtf50), "rise_time": float(rise_time)}
    
    def _analyze_completeness(self, gray: np.ndarray) -> CompletenessResult:
        """Analyze edge completeness (continuity, breakage)"""
        result = CompletenessResult()
        
        # Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            result.breakage_count = 0
            result.edge_continuity = 0.0
            result.score = 0.0
            return result
        
        # Analyze each contour for breaks
        total_length = 0
        broken_length = 0
        all_breakage_points = []
        
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            total_length += perimeter
            
            # Approximate contour to detect corners/breaks
            epsilon = 0.02 * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Detect breaks by analyzing curvature
            if len(approx) > 2:
                # Calculate curvature at each point
                for i in range(len(approx)):
                    p1 = approx[i % len(approx)][0]
                    p2 = approx[(i + 1) % len(approx)][0]
                    p3 = approx[(i + 2) % len(approx)][0]
                    
                    # Angle at p2
                    v1 = p1 - p2
                    v2 = p3 - p2
                    
                    angle = self._angle_between(v1, v2)
                    
                    # Sharp angle indicates break/corner
                    if angle > 150:  # degrees
                        broken_length += np.linalg.norm(v2)
                        all_breakage_points.append((int(p2[0]), int(p2[1])))
        
        # Calculate metrics
        if total_length > 0:
            result.broken_edge_ratio = broken_length / total_length
            result.edge_continuity = max(0, 1 - result.broken_edge_ratio * 2)
        
        result.breakage_count = len(all_breakage_points)
        result.breakage_locations = all_breakage_points[:10]  # Limit to first 10
        
        # Score: fewer breaks = higher score
        breakage_score = max(0, 100 - result.breakage_count * 10)
        continuity_score = result.edge_continuity * 100
        
        result.score = breakage_score * 0.5 + continuity_score * 0.5
        
        return result
    
    @staticmethod
    def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate angle between two vectors in degrees"""
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
        cos_angle = np.clip(cos_angle, -1, 1)
        return np.degrees(np.arccos(cos_angle))
    
    def _analyze_accuracy(
        self,
        gray: np.ndarray,
        reference_edges: Optional[np.ndarray] = None
    ) -> AccuracyResult:
        """Analyze contour geometric accuracy"""
        result = AccuracyResult()
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Find dominant lines using Hough transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
        
        if lines is None or len(lines) == 0:
            result.score = 50.0  # Neutral
            return result
        
        # Calculate geometric properties
        angles = []
        lengths = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            angle = np.arctan2(y2-y1, x2-x1)
            
            angles.append(angle)
            lengths.append(length)
        
        # Check for rectangular structures (0°, 90°, 180°, 270°)
        expected_angles = [0, np.pi/2, np.pi, 3*np.pi/2]
        angle_errors = []
        
        for angle in angles:
            min_error = min(abs(angle - ea) % (2*np.pi) for ea in expected_angles)
            angle_errors.append(min_error)
        
        # Geometric distortion = mean angular deviation from expected
        result.geometric_distortion = float(np.mean(angle_errors) / (np.pi/4))
        
        # Aspect ratio error (if we have rectangles)
        if len(lines) >= 4:
            vertical_lines = [l for l, a in zip(lengths, angles) if abs(abs(a) - np.pi/2) < 0.1]
            horizontal_lines = [l for l, a in zip(lengths, angles) if abs(a) < 0.1 or abs(a) > 2.8]
            
            if vertical_lines and horizontal_lines:
                avg_v = np.mean(vertical_lines)
                avg_h = np.mean(horizontal_lines)
                if avg_h > 0:
                    aspect_ratio = avg_v / avg_h
                    result.aspect_ratio_error = abs(aspect_ratio - 1.0)  # Error from perfect square
        
        # Compare with reference edges if provided
        if reference_edges is not None:
            # Calculate pixel-wise deviation
            diff = cv2.absdiff(edges, reference_edges)
            result.contour_deviation_px = float(diff.mean())
        
        # Score calculation
        distortion_score = max(0, 100 - result.geometric_distortion * 200)
        deviation_score = max(0, 100 - result.contour_deviation_px * 20)
        aspect_score = max(0, 100 - result.aspect_ratio_error * 100)
        
        result.score = distortion_score * 0.4 + deviation_score * 0.3 + aspect_score * 0.3
        
        return result
    
    def _calculate_overall_score(self, result: ContourEvaluationResult) -> float:
        """Calculate weighted overall score"""
        weights = {
            "sharpness": 0.45,
            "completeness": 0.30,
            "accuracy": 0.25
        }
        
        score = (
            result.sharpness.score * weights["sharpness"] +
            result.completeness.score * weights["completeness"] +
            result.accuracy.score * weights["accuracy"]
        )
        
        return min(100, max(0, score))
    
    def _check_pass(self, result: ContourEvaluationResult) -> bool:
        """Check if result passes all thresholds"""
        issues = []
        
        # Sharpness check
        if result.sharpness.score < self.MIN_SHARPNESS_SCORE:
            issues.append({
                "type": "sharpness",
                "severity": "high",
                "detail": f"Sharpness score {result.sharpness.score:.1f} below {self.MIN_SHARPNESS_SCORE}"
            })
        
        if result.sharpness.laplacian_variance < self.MIN_LAPLACIAN_VARIANCE:
            issues.append({
                "type": "sharpness",
                "severity": "medium",
                "detail": f"Laplacian variance {result.sharpness.laplacian_variance:.1f} below {self.MIN_LAPLACIAN_VARIANCE}"
            })
        
        # Completeness check
        if result.completeness.breakage_count > self.MAX_EDGE_BREAKAGE:
            issues.append({
                "type": "edge_breakage",
                "severity": "medium",
                "detail": f"Edge breakage count {result.completeness.breakage_count} exceeds {self.MAX_EDGE_BREAKAGE}",
                "locations": result.completeness.breakage_locations
            })
        
        if result.completeness.edge_continuity < self.MIN_EDGE_CONTINUITY:
            issues.append({
                "type": "edge_continuity",
                "severity": "medium",
                "detail": f"Edge continuity {result.completeness.edge_continuity:.2f} below {self.MIN_EDGE_CONTINUITY}"
            })
        
        # Accuracy check
        if result.accuracy.geometric_distortion > self.MAX_GEOMETRIC_DISTORTION:
            issues.append({
                "type": "geometric_distortion",
                "severity": "low",
                "detail": f"Geometric distortion {result.accuracy.geometric_distortion:.3f} exceeds {self.MAX_GEOMETRIC_DISTORTION}"
            })
        
        if result.accuracy.contour_deviation_px > self.MAX_CONTOUR_DEVIATION_PX:
            issues.append({
                "type": "contour_deviation",
                "severity": "low",
                "detail": f"Contour deviation {result.accuracy.contour_deviation_px:.1f}px exceeds {self.MAX_CONTOUR_DEVIATION_PX}px"
            })
        
        result.issues = issues
        return len(issues) == 0
    
    def visualize_edges(self, gray: np.ndarray) -> np.ndarray:
        """
        Generate edge visualization for report.
        
        Args:
            gray: Grayscale image
            
        Returns:
            Visualization image (BGR)
        """
        # Canny edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Colorize edges
        edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # Ensure gray is 3-channel for addWeighted
        if len(gray.shape) == 2:
            gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        else:
            gray_3ch = gray
        
        # Overlay on original
        result = cv2.addWeighted(gray_3ch, 0.7, edges_color, 0.3, 0)
        
        return result
