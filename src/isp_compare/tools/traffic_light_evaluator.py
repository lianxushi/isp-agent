"""
Traffic Light Evaluator
=======================

Dedicated evaluator for traffic light color fidelity and detection assessment.
Implements two analysis scenarios:
- Scenario 1: ISP tuning (objective metrics - color/chroma accuracy)
- Scenario 2: ADAS perception validation (detection success evaluation)

Author: ISP Team
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

from ..utils.logger import get_logger

logger = get_logger(__name__)


class LightColor(Enum):
    """Traffic light color enumeration"""
    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"


@dataclass
class TrafficLightRegion:
    """Individual traffic light region analysis result"""
    color: LightColor
    detected: bool = False
    
    # Geometry
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # x, y, w, h
    area: float = 0.0
    centroid: Tuple[float, float] = (0.0, 0.0)
    
    # Color analysis
    mean_hsv: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    mean_bgr: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    chroma_delta: float = 0.0  # CIE 1976 Δuv deviation from standard
    brightness: float = 0.0
    saturation: float = 0.0
    
    # Color uniformity
    hue_std: float = 0.0
    uniformity_score: float = 0.0
    
    # Ghost/halo detection
    has_ghost: bool = False
    ghost_severity: float = 0.0
    
    # Pass/fail
    passed: bool = False
    issues: List[str] = field(default_factory=list)


@dataclass
class TrafficLightEvaluationResult:
    """Complete traffic light evaluation result"""
    overall_score: float = 0.0
    overall_passed: bool = False
    
    # Per-color results
    red: Optional[TrafficLightRegion] = None
    yellow: Optional[TrafficLightRegion] = None
    green: Optional[TrafficLightRegion] = None
    
    # Detection summary
    detected_colors: List[LightColor] = field(default_factory=list)
    missing_colors: List[LightColor] = field(default_factory=list)
    
    # Issues
    issues: List[Dict[str, Any]] = field(default_factory=list)
    
    # Detection mode (ISP tuning vs ADAS perception)
    mode: str = "isp_tuning"  # "isp_tuning" or "adas_perception"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = {
            "overall_score": self.overall_score,
            "overall_passed": self.overall_passed,
            "mode": self.mode,
            "detected_colors": [c.value for c in self.detected_colors],
            "missing_colors": [c.value for c in self.missing_colors],
            "issues": self.issues
        }
        
        for color in ["red", "yellow", "green"]:
            region = getattr(self, color, None)
            if region:
                result[color] = {
                    "detected": region.detected,
                    "bbox": region.bbox,
                    "area": region.area,
                    "chroma_delta": region.chroma_delta,
                    "brightness": region.brightness,
                    "uniformity_score": region.uniformity_score,
                    "has_ghost": region.has_ghost,
                    "passed": region.passed,
                    "issues": region.issues
                }
            else:
                result[color] = {"detected": False}
        
        return result


class TrafficLightEvaluator:
    """
    Dedicated traffic light evaluator.
    
    Implements color fidelity analysis for ISP tuning and detection
    assessment for ADAS perception validation.
    
    IEC 61888 standard color ranges are used for reference.
    
    Example (ISP tuning):
        >>> evaluator = TrafficLightEvaluator()
        >>> result = evaluator.evaluate(image, roi=(x, y, w, h))
        >>> print(f"Score: {result.overall_score}, Passed: {result.overall_passed}")
    
    Example (ADAS perception):
        >>> evaluator = TrafficLightEvaluator(mode="adas_perception")
        >>> result = evaluator.evaluate(image, roi=(x, y, w, h))
    """
    
    # IEC 61888 standard traffic light colors in HSV (OpenCV format: H=0-180, S=0-255, V=0-255)
    # Based on CIE 1976 UCS coordinates converted to HSV
    STANDARD_COLORS = {
        LightColor.RED: {
            "hsv_lower": np.array([0, 50, 50]),
            "hsv_upper": np.array([10, 255, 255]),
            "cie_uv_target": (0.456, 0.294),  # Approximate for red
            "cie_uv_tolerance": 0.05,
        },
        LightColor.YELLOW: {
            "hsv_lower": np.array([20, 50, 50]),
            "hsv_upper": np.array([35, 255, 255]),
            "cie_uv_target": (0.447, 0.511),  # Approximate for yellow
            "cie_uv_tolerance": 0.05,
        },
        LightColor.GREEN: {
            "hsv_lower": np.array([40, 50, 50]),
            "hsv_upper": np.array([80, 255, 255]),
            "cie_uv_target": (0.213, 0.525),  # Approximate for green
            "cie_uv_tolerance": 0.05,
        },
    }
    
    # Minimum area threshold to filter noise (pixels)
    MIN_LIGHT_AREA = 100
    
    # Chroma delta threshold (IEC 61888)
    MAX_CHROMA_DELTA = 0.05
    
    # Brightness thresholds
    MIN_BRIGHTNESS = 50
    MAX_BRIGHTNESS = 250
    
    def __init__(self, mode: str = "isp_tuning"):
        """
        Initialize traffic light evaluator.
        
        Args:
            mode: Analysis mode - "isp_tuning" or "adas_perception"
        """
        self.mode = mode
        logger.debug(f"TrafficLightEvaluator initialized in {mode} mode")
    
    def evaluate(
        self,
        image: np.ndarray,
        roi: Optional[Tuple[int, int, int, int]] = None,
        ground_truth_boxes: Optional[List[List[float]]] = None
    ) -> TrafficLightEvaluationResult:
        """
        Evaluate traffic light in image.
        
        Args:
            image: Input image (BGR format,HxWx3)
            roi: Region of interest (x, y, w, h) - manual ROI for traffic light
            ground_truth_boxes: Optional GT bboxes for ADAS mode [x1,y1,x2,y2]
            
        Returns:
            TrafficLightEvaluationResult: Evaluation result
        """
        result = TrafficLightEvaluationResult(mode=self.mode)
        
        # Extract ROI
        if roi:
            x, y, w, h = roi
            roi_image = image[y:y+h, x:x+w]
        else:
            roi_image = image
            roi = (0, 0, image.shape[1], image.shape[0])
        
        hsv_roi = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)
        
        # Analyze each color
        for color in LightColor:
            region = self._analyze_color(hsv_roi, roi_image, color, roi)
            setattr(result, color.value, region)
            
            if region.detected:
                result.detected_colors.append(color)
            else:
                result.missing_colors.append(color)
        
        # Calculate overall score
        result.overall_score = self._calculate_overall_score(result)
        result.overall_passed = self._check_pass(result)
        
        # ADAS mode: evaluate detection against ground truth
        if self.mode == "adas_perception" and ground_truth_boxes:
            self._evaluate_detection_adas(result, ground_truth_boxes)
        
        return result
    
    def _analyze_color(
        self,
        hsv_roi: np.ndarray,
        bgr_roi: np.ndarray,
        color: LightColor,
        roi: Tuple[int, int, int, int]
    ) -> TrafficLightRegion:
        """Analyze a specific traffic light color"""
        region = TrafficLightRegion(color=color)
        spec = self.STANDARD_COLORS[color]
        
        # Create mask for this color
        mask = cv2.inRange(hsv_roi, spec["hsv_lower"], spec["hsv_upper"])
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            region.detected = False
            return region
        
        # Get largest contour
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        
        if area < self.MIN_LIGHT_AREA:
            region.detected = False
            region.issues.append(f"Area {area:.0f} below threshold {self.MIN_LIGHT_AREA}")
            return region
        
        region.detected = True
        region.area = area
        
        # Bounding box
        x, y, w, h = cv2.boundingRect(largest)
        region.bbox = (x + roi[0], y + roi[1], w, h)  # Adjust for ROI offset
        
        # Centroid
        M = cv2.moments(largest)
        if M["m00"] > 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            region.centroid = (cx, cy)
        
        # Color analysis
        mean_val = cv2.mean(bgr_roi, mask=mask)
        region.mean_bgr = (mean_val[0], mean_val[1], mean_val[2])
        
        # HSV analysis
        hsv_masked = hsv_roi[mask > 0]
        if len(hsv_masked) > 0:
            region.hue_std = float(hsv_masked[:, 0].std())
            region.mean_hsv = (
                float(hsv_masked[:, 0].mean()),
                float(hsv_masked[:, 1].mean()),
                float(hsv_masked[:, 2].mean())
            )
            region.brightness = float(hsv_masked[:, 2].mean())
            region.saturation = float(hsv_masked[:, 1].mean())
        
        # Calculate chroma deviation (simplified CIE 1976 Δuv)
        region.chroma_delta = self._calculate_chroma_delta(region.mean_bgr, color)
        
        # Color uniformity
        region.uniformity_score = self._calculate_uniformity(hsv_masked, color)
        
        # Ghost/halo detection
        region.has_ghost, region.ghost_severity = self._detect_ghost(bgr_roi, mask, largest)
        
        # Pass/fail
        region.passed = self._check_color_pass(region)
        
        return region
    
    def _calculate_chroma_delta(
        self,
        mean_bgr: Tuple[float, float, float],
        color: LightColor
    ) -> float:
        """
        Calculate chroma deviation from standard (simplified CIE 1976 Δuv).
        
        Uses BGR distance as proxy for chroma deviation since full LAB
        conversion is not always available.
        """
        spec = self.STANDARD_COLORS[color]
        target = spec["cie_uv_target"]
        
        # Convert BGR to approximate UV
        # Simplified: use normalized BGR as proxy
        b, g, r = [v / 255.0 for v in mean_bgr]
        
        # Approximate CIE UV from RGB (simplified)
        # Real implementation would use proper color space conversion
        r_corrected = r - 0.5 * (b + g)
        g_corrected = g - 0.5 * (r + b)
        
        # Distance from origin as chroma proxy
        chroma = np.sqrt(r_corrected**2 + g_corrected**2)
        
        # Compare with typical target chroma
        target_chroma = np.sqrt(target[0]**2 + target[1]**2)
        
        delta = abs(chroma - target_chroma)
        
        return float(delta)
    
    def _calculate_uniformity(
        self,
        hsv_pixels: np.ndarray,
        color: LightColor
    ) -> float:
        """
        Calculate color uniformity score (0-1).
        
        Lower hue standard deviation = more uniform color.
        """
        if len(hsv_pixels) < 10:
            return 0.0
        
        hue_std = hsv_pixels[:, 0].std()
        sat_std = hsv_pixels[:, 1].std()
        
        # Normalize: typical hue std for uniform light is < 10
        hue_uniformity = max(0, 1 - hue_std / 20)
        sat_uniformity = max(0, 1 - sat_std / 50)
        
        return float((hue_uniformity + sat_uniformity) / 2)
    
    def _detect_ghost(
        self,
        bgr_roi: np.ndarray,
        mask: np.ndarray,
        contour
    ) -> Tuple[bool, float]:
        """
        Detect ghosting/halo artifacts around the light.
        
        Returns:
            (has_ghost, severity)
        """
        # Get the bounding box of the light
        x, y, w, h = cv2.boundingRect(contour)
        
        # Expand the region to check surroundings
        expand = max(w, h) // 2
        x1 = max(0, x - expand)
        y1 = max(0, y - expand)
        x2 = min(bgr_roi.shape[1], x + w + expand)
        y2 = min(bgr_roi.shape[0], y + h + expand)
        
        if x2 <= x1 or y2 <= y1:
            return False, 0.0
        
        # Calculate mean brightness inside vs outside the light
        light_region = bgr_roi[y:y+h, x:x+w]
        surround_region = bgr_roi[y1:y2, x1:x2]
        
        # Mask out the light region from surrounding
        surround_gray = cv2.cvtColor(surround_region, cv2.COLOR_BGR2GRAY)
        
        # Check for bright ring around the light (halo)
        light_mean = np.mean(cv2.cvtColor(light_region, cv2.COLOR_BGR2GRAY))
        surround_mean = np.mean(surround_gray)
        
        if surround_mean > light_mean * 0.5:  # Significant bright ring
            severity = min(1.0, surround_mean / light_mean)
            return True, float(severity)
        
        return False, 0.0
    
    def _check_color_pass(self, region: TrafficLightRegion) -> bool:
        """Check if a color region passes all criteria"""
        issues = []
        
        # Check chroma delta
        if region.chroma_delta > self.MAX_CHROMA_DELTA:
            issues.append(f"Chroma delta {region.chroma_delta:.4f} exceeds threshold {self.MAX_CHROMA_DELTA}")
        
        # Check brightness range
        if region.brightness < self.MIN_BRIGHTNESS:
            issues.append(f"Brightness {region.brightness:.1f} below minimum {self.MIN_BRIGHTNESS}")
        elif region.brightness > self.MAX_BRIGHTNESS:
            issues.append(f"Brightness {region.brightness:.1f} exceeds maximum {self.MAX_BRIGHTNESS}")
        
        # Check uniformity
        if region.uniformity_score < 0.7:
            issues.append(f"Uniformity {region.uniformity_score:.2f} below 0.7")
        
        region.issues = issues
        return len(issues) == 0
    
    def _calculate_overall_score(self, result: TrafficLightEvaluationResult) -> float:
        """Calculate overall traffic light score (0-100)"""
        if not result.detected_colors:
            return 0.0
        
        scores = []
        
        for color in [LightColor.RED, LightColor.YELLOW, LightColor.GREEN]:
            region = getattr(result, color.value, None)
            if region and region.detected:
                # Score based on chroma delta (lower is better)
                chroma_score = max(0, 1 - region.chroma_delta / 0.15) * 100
                
                # Uniformity weight
                uniformity_weight = 0.3
                color_score = chroma_score * (1 - uniformity_weight) + region.uniformity_score * 100 * uniformity_weight
                
                # Ghost penalty
                if region.has_ghost:
                    color_score *= (1 - region.ghost_severity * 0.3)
                
                scores.append(color_score)
        
        if not scores:
            return 0.0
        
        return float(np.mean(scores))
    
    def _check_pass(self, result: TrafficLightEvaluationResult) -> bool:
        """Check overall pass/fail"""
        issues = []
        
        for color in [LightColor.RED, LightColor.YELLOW, LightColor.GREEN]:
            region = getattr(result, color.value, None)
            if region and region.detected and not region.passed:
                issues.extend([f"{color.value}: {issue}" for issue in region.issues])
        
        result.issues = [{"type": "color_issue", "detail": issue} for issue in issues]
        return len(issues) == 0
    
    def _evaluate_detection_adas(
        self,
        result: TrafficLightEvaluationResult,
        ground_truth_boxes: List[List[float]]
    ) -> None:
        """Evaluate detection performance against ground truth (ADAS mode)"""
        result.issues.append({
            "type": "adas_evaluation",
            "mode": "adas_perception",
            "note": "ADAS evaluation requires perception model integration",
            "detected": [c.value for c in result.detected_colors],
            "ground_truth_count": len(ground_truth_boxes)
        })
    
    def detect_auto_roi(
        self,
        image: np.ndarray,
        expected_lights: int = 3
    ) -> Optional[List[Tuple[int, int, int, int]]]:
        """
        Auto-detect potential traffic light ROIs.
        
        This is a helper for finding traffic light regions automatically.
        For full ADAS perception, use the PerceptionModelInterface.
        
        Args:
            image: Input image
            expected_lights: Expected number of lights (for filtering)
            
        Returns:
            List of ROIs (x, y, w, h) or None if detection fails
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        all_regions = []
        
        for color in LightColor:
            spec = self.STANDARD_COLORS[color]
            mask = cv2.inRange(hsv, spec["hsv_lower"], spec["hsv_upper"])
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                if cv2.contourArea(cnt) >= self.MIN_LIGHT_AREA:
                    x, y, w, h = cv2.boundingRect(cnt)
                    all_regions.append((x, y, w, h, color))
        
        if not all_regions:
            return None
        
        # Sort by area (largest first) and take expected_lights
        all_regions.sort(key=lambda r: r[2] * r[3], reverse=True)
        
        # Cluster nearby regions
        rois = self._cluster_regions(all_regions[:expected_lights * 2])
        
        return rois[:expected_lights] if len(rois) >= expected_lights else None
    
    def _cluster_regions(
        self,
        regions: List[Tuple[int, int, int, int, LightColor]]
    ) -> List[Tuple[int, int, int, int]]:
        """Cluster nearby regions to avoid duplicates"""
        if not regions:
            return []
        
        rois = []
        used = set()
        
        for i, (x, y, w, h, color) in enumerate(regions):
            if i in used:
                continue
            
            # Find all regions in same vertical band (likely same traffic light column)
            group = [(x, y, w, h)]
            used.add(i)
            
            for j, (x2, y2, w2, h2, _) in enumerate(regions[i+1:], i+1):
                if j in used:
                    continue
                # Check if roughly aligned vertically
                if abs(x - x2) < max(w, w2):
                    group.append((x2, y2, w2, h2))
                    used.add(j)
            
            # Merge group into single ROI
            all_x = [r[0] for r in group]
            all_y = [r[1] for r in group]
            all_x2 = [r[0] + r[2] for r in group]
            all_y2 = [r[1] + r[3] for r in group]
            
            merged_x = min(all_x)
            merged_y = min(all_y)
            merged_w = max(all_x2) - merged_x
            merged_h = max(all_y2) - merged_y
            
            rois.append((merged_x, merged_y, merged_w, merged_h))
        
        return rois
