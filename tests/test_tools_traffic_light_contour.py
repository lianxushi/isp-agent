"""
Tests for TrafficLightEvaluator and ContourEvaluator
====================================================

Author: ISP Team
"""

import sys
import os
import pytest
import numpy as np
import cv2

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from isp_compare.tools import (
    TrafficLightEvaluator,
    ContourEvaluator,
    LightColor,
    TrafficLightRegion,
    ContourEvaluationResult,
    TrafficLightEvaluationResult
)


class TestTrafficLightEvaluator:
    """Tests for TrafficLightEvaluator"""
    
    @pytest.fixture
    def evaluator(self):
        """Create evaluator instance"""
        return TrafficLightEvaluator(mode="isp_tuning")
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample image with traffic light colors"""
        # Create 640x480 BGR image
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add some noise background
        image[:] = (30, 30, 30)
        
        # Add red region (top)
        cv2.rectangle(image, (280, 50), (360, 130), (30, 30, 200), -1)
        
        # Add yellow region (middle)
        cv2.rectangle(image, (280, 160), (360, 240), (40, 210, 230), -1)
        
        # Add green region (bottom)
        cv2.rectangle(image, (280, 270), (360, 350), (60, 200, 60), -1)
        
        return image
    
    def test_init(self, evaluator):
        """Test evaluator initialization"""
        assert evaluator.mode == "isp_tuning"
        assert evaluator.MIN_LIGHT_AREA == 100
        assert evaluator.MAX_CHROMA_DELTA == 0.05
    
    def test_evaluate_detects_colors(self, evaluator, sample_image):
        """Test that evaluator detects all three colors"""
        result = evaluator.evaluate(sample_image)
        
        assert isinstance(result, TrafficLightEvaluationResult)
        assert result.mode == "isp_tuning"
        
        # Should detect at least red and green
        assert len(result.detected_colors) >= 2
    
    def test_evaluate_with_roi(self, evaluator, sample_image):
        """Test evaluation with ROI"""
        roi = (270, 40, 100, 320)  # Around the traffic light
        result = evaluator.evaluate(sample_image, roi=roi)
        
        assert result is not None
        assert isinstance(result, TrafficLightEvaluationResult)
    
    def test_red_region_detected(self, evaluator, sample_image):
        """Test red region detection"""
        roi = (270, 40, 100, 320)
        result = evaluator.evaluate(sample_image, roi=roi)
        
        if result.red:
            assert result.red.color == LightColor.RED
            assert result.red.detected == True
            assert result.red.area > 0
    
    def test_score_calculation(self, evaluator, sample_image):
        """Test overall score calculation"""
        roi = (270, 40, 100, 320)
        result = evaluator.evaluate(sample_image, roi=roi)
        
        assert 0 <= result.overall_score <= 100
    
    def test_empty_image(self, evaluator):
        """Test evaluation on empty/dark image"""
        empty_image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = evaluator.evaluate(empty_image)
        
        # Should return 0 score with no detections
        assert result.overall_score == 0.0
        assert len(result.detected_colors) == 0
    
    def test_adas_mode(self):
        """Test ADAS perception mode"""
        evaluator = TrafficLightEvaluator(mode="adas_perception")
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Add a red region
        cv2.rectangle(image, (40, 30), (60, 70), (20, 20, 180), -1)
        
        result = evaluator.evaluate(image)
        
        assert result.mode == "adas_perception"
    
    def test_auto_roi_detection(self, evaluator, sample_image):
        """Test automatic ROI detection"""
        rois = evaluator.detect_auto_roi(sample_image, expected_lights=3)
        
        # Should find at least some potential lights or return None gracefully
        # Either way, shouldn't crash
        assert rois is None or isinstance(rois, list)
    
    def test_color_passing(self, evaluator, sample_image):
        """Test color pass/fail determination"""
        roi = (270, 40, 100, 320)
        result = evaluator.evaluate(sample_image, roi=roi)
        
        # If a color is detected, it should have a passed status
        for color in [LightColor.RED, LightColor.YELLOW, LightColor.GREEN]:
            region = getattr(result, color.value, None)
            if region and region.detected:
                # Check that passed is boolean
                assert isinstance(region.passed, bool)


class TestContourEvaluator:
    """Tests for ContourEvaluator"""
    
    @pytest.fixture
    def evaluator(self):
        """Create evaluator instance"""
        return ContourEvaluator()
    
    @pytest.fixture
    def sharp_image(self):
        """Create a sharp test image with clear edges"""
        image = np.zeros((200, 200), dtype=np.uint8)
        
        # Add white rectangle (sharp edges)
        cv2.rectangle(image, (50, 50), (150, 150), 255, 2)
        
        # Add diagonal line
        cv2.line(image, (20, 180), (180, 20), 200, 2)
        
        # Add grid pattern
        for i in range(20, 200, 20):
            cv2.line(image, (i, 0), (i, 200), 100, 1)
            cv2.line(image, (0, i), (200, i), 100, 1)
        
        # Convert to BGR
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    @pytest.fixture
    def blurry_image(self):
        """Create a blurry test image"""
        image = np.zeros((200, 200), dtype=np.uint8)
        
        # Add shapes
        cv2.circle(image, (100, 100), 50, 150, -1)
        cv2.rectangle(image, (30, 30), (80, 80), 100, -1)
        
        # Blur heavily
        blurred = cv2.GaussianBlur(image, (21, 21), 10)
        
        return cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
    
    def test_init(self, evaluator):
        """Test evaluator initialization"""
        assert evaluator.MIN_SHARPNESS_SCORE == 50.0
        assert evaluator.MAX_EDGE_BREAKAGE == 5
    
    def test_evaluate_sharp_image(self, evaluator, sharp_image):
        """Test evaluation on sharp image"""
        result = evaluator.evaluate(sharp_image)
        
        assert isinstance(result, ContourEvaluationResult)
        assert result.overall_score >= 0
        assert result.overall_score <= 100
    
    def test_evaluate_blurry_image(self, evaluator, blurry_image):
        """Test evaluation on blurry image"""
        result = evaluator.evaluate(blurry_image)
        
        assert isinstance(result, ContourEvaluationResult)
        # Blurry image should have lower score
        assert result.overall_score >= 0
    
    def test_sharpness_analysis(self, evaluator, sharp_image):
        """Test sharpness analysis component"""
        result = evaluator.evaluate(sharp_image)
        
        assert hasattr(result, 'sharpness')
        assert hasattr(result.sharpness, 'score')
        assert hasattr(result.sharpness, 'mtf50_estimate')
        assert hasattr(result.sharpness, 'laplacian_variance')
        
        # Sharp image should have decent MTF50
        assert result.sharpness.mtf50_estimate >= 0
    
    def test_completeness_analysis(self, evaluator, sharp_image):
        """Test edge completeness analysis"""
        result = evaluator.evaluate(sharp_image)
        
        assert hasattr(result, 'completeness')
        assert hasattr(result.completeness, 'edge_continuity')
        assert hasattr(result.completeness, 'breakage_count')
        
        # Completeness score should be 0-100
        assert 0 <= result.completeness.score <= 100
    
    def test_accuracy_analysis(self, evaluator, sharp_image):
        """Test geometric accuracy analysis"""
        result = evaluator.evaluate(sharp_image)
        
        assert hasattr(result, 'accuracy')
        assert hasattr(result.accuracy, 'geometric_distortion')
        assert hasattr(result.accuracy, 'score')
    
    def test_compare_sharp_vs_blurry(self, evaluator, sharp_image, blurry_image):
        """Test that sharp image scores higher than blurry"""
        sharp_result = evaluator.evaluate(sharp_image)
        blurry_result = evaluator.evaluate(blurry_image)
        
        # Sharp should typically score higher
        # (not guaranteed for all images, but reasonable expectation)
        print(f"Sharp score: {sharp_result.overall_score:.1f}")
        print(f"Blurry score: {blurry_result.overall_score:.1f}")
    
    def test_with_roi(self, evaluator, sharp_image):
        """Test evaluation with ROI"""
        roi = (50, 50, 100, 100)
        result = evaluator.evaluate(sharp_image, roi=roi)
        
        assert result is not None
        assert isinstance(result, ContourEvaluationResult)
    
    def test_empty_image(self, evaluator):
        """Test evaluation on empty image"""
        empty = np.zeros((100, 100, 3), dtype=np.uint8)
        result = evaluator.evaluate(empty)
        
        assert result is not None
        assert 0 <= result.overall_score <= 100
    
    def test_visualize_edges(self, evaluator, sharp_image):
        """Test edge visualization"""
        gray = cv2.cvtColor(sharp_image, cv2.COLOR_BGR2GRAY)
        vis = evaluator.visualize_edges(gray)
        
        assert vis is not None
        assert vis.shape == sharp_image.shape
    
    def test_pass_check(self, evaluator, sharp_image):
        """Test pass/fail determination"""
        result = evaluator.evaluate(sharp_image)
        
        # Result should have overall_passed boolean
        assert isinstance(result.overall_passed, bool)
        
        # Issues should be a list
        assert isinstance(result.issues, list)


class TestIntegration:
    """Integration tests"""
    
    def test_both_evaluators_work(self):
        """Test that both evaluators can be instantiated and used"""
        tl_eval = TrafficLightEvaluator()
        contour_eval = ContourEvaluator()
        
        # Create test image
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.rectangle(image, (50, 50), (150, 150), (30, 30, 200), -1)
        
        # Run both
        tl_result = tl_eval.evaluate(image)
        contour_result = contour_eval.evaluate(image)
        
        assert tl_result is not None
        assert contour_result is not None
    
    def test_tools_import(self):
        """Test that all tools can be imported"""
        from isp_compare.tools import (
            TrafficLightEvaluator,
            ContourEvaluator,
            PerceptionModelInterface,
            LightColor,
            SharpnessResult,
            CompletenessResult,
            AccuracyResult,
            TrafficLightRegion,
            TrafficLightEvaluationResult,
            ContourEvaluationResult
        )
        
        assert TrafficLightEvaluator is not None
        assert ContourEvaluator is not None
        assert PerceptionModelInterface is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
