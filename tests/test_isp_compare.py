"""
Tests for ISP Compare Module
==========================

Author: ISP Team
"""

import os
import sys
import pytest
import tempfile
import numpy as np
import cv2
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from isp_compare.core.comp12_parser import Comp12Parser, Comp12Config, Comp12ParseError
from isp_compare.core.cmodel_wrapper import CModelISP, CModelResult, CModelError
from isp_compare.core.metrics import ImageMetrics, MetricsResult
from isp_compare.core.comparison import ISPComparator, ComparisonConfig, ComparisonResult
from isp_compare.tools.perception_model import PerceptionModelInterface, DetectionResult
from isp_compare.reports.pdf_generator import PDFReportGenerator


class TestComp12Parser:
    """Tests for Comp12Parser"""
    
    def test_parser_initialization(self):
        """Test parser initialization"""
        parser = Comp12Parser()
        assert parser.config.width == 3840
        assert parser.config.height == 2160
        assert parser.config.pattern == "RGGB"
    
    def test_parser_with_custom_config(self):
        """Test parser with custom config"""
        config = Comp12Config(width=1920, height=1080, pattern="BGGR")
        parser = Comp12Parser(config)
        assert parser.config.width == 1920
        assert parser.config.height == 1080
        assert parser.config.pattern == "BGGR"
    
    def test_parser_unsupported_pattern(self):
        """Test parser rejects unsupported pattern"""
        config = Comp12Config(pattern="UNSUPPORTED")
        with pytest.raises(Comp12ParseError):
            Comp12Parser(config)
    
    def test_parser_unsupported_resolution(self):
        """Test parser rejects unsupported resolution"""
        config = Comp12Config(width=1024, height=768)
        with pytest.raises(Comp12ParseError):
            Comp12Parser(config)
    
    def test_parse_nonexistent_file(self):
        """Test parser handles nonexistent file"""
        parser = Comp12Parser()
        with pytest.raises(FileNotFoundError):
            parser.parse("/nonexistent/file.raw")
    
    def test_parse_valid_file(self, tmp_path):
        """Test parsing valid Comp12 file"""
        # Create test file
        test_file = tmp_path / "test.raw"
        width, height = 3840, 2160
        expected_pixels = width * height
        
        # Create random 12-bit data
        data = np.random.randint(0, 4096, expected_pixels, dtype=np.uint16)
        data.tofile(str(test_file))
        
        # Parse
        parser = Comp12Parser(Comp12Config(width=width, height=height))
        raw16 = parser.parse(str(test_file))
        
        assert raw16.shape == (height, width)
        assert raw16.dtype == np.uint16
        assert raw16.min() >= 0
        assert raw16.max() < 4096
    
    def test_save_for_cmodel(self, tmp_path):
        """Test saving for CModel"""
        parser = Comp12Parser()
        raw16 = np.random.randint(0, 4095, (2160, 3840), dtype=np.uint16)
        
        output_file = tmp_path / "output.raw"
        parser.save_for_cmodel(raw16, str(output_file))
        
        assert output_file.exists()
        
        # Verify saved data
        loaded = np.fromfile(str(output_file), dtype=np.uint16)
        assert len(loaded) == 2160 * 3840
        assert np.array_equal(loaded.reshape((2160, 3840)), raw16)
    
    def test_get_bayer_channel_map(self):
        """Test Bayer pattern channel mapping"""
        parser = Comp12Parser(Comp12Config(pattern="RGGB"))
        channel_map = parser.get_bayer_channel_map()
        
        assert (0, 0) in channel_map  # R at (0,0)
        assert (0, 1) in channel_map  # Gr at (0,1)
    
    def test_validate_file(self, tmp_path):
        """Test file validation"""
        test_file = tmp_path / "test.raw"
        expected_pixels = 3840 * 2160
        
        # Create valid file
        data = np.random.randint(0, 4096, expected_pixels, dtype=np.uint16)
        data.tofile(str(test_file))
        
        assert Comp12Parser.validate_file(str(test_file), expected_pixels)
        assert Comp12Parser.validate_file(str(test_file), expected_pixels + 1) is False


class TestCModelWrapper:
    """Tests for CModel wrapper"""
    
    def test_cmodel_nonexistent_path(self):
        """Test CModel with nonexistent executable"""
        with pytest.raises(CModelError):
            CModelISP("/nonexistent/cmodel")
    
    def test_cmodel_result_dataclass(self):
        """Test CModelResult dataclass"""
        result = CModelResult(
            success=True,
            output_path="/tmp/output.jpg",
            time_ms=150.5
        )
        
        assert result.success is True
        assert result.output_path == "/tmp/output.jpg"
        assert result.time_ms == 150.5
        assert result.error == ""


class TestImageMetrics:
    """Tests for ImageMetrics"""
    
    def test_metrics_initialization(self):
        """Test metrics initialization"""
        metrics = ImageMetrics()
        assert metrics is not None
    
    def test_calculate_sharpness(self, tmp_path):
        """Test sharpness calculation"""
        metrics = ImageMetrics()
        
        # Create test image (sharp edge)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[40:60, 40:60] = 255  # White square
        
        score, details = metrics._calculate_sharpness(img)
        
        assert 0 <= score <= 100
        assert "laplacian_variance" in details
    
    def test_calculate_noise(self, tmp_path):
        """Test noise calculation"""
        metrics = ImageMetrics()
        
        # Create test image with noise
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        score, details = metrics._calculate_noise(img)
        
        assert 0 <= score <= 100
        assert "noise_std" in details
    
    def test_calculate_color_metrics(self, tmp_path):
        """Test color metrics calculation"""
        metrics = ImageMetrics()
        
        # Create colorful test image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, :33] = [255, 0, 0]    # Red
        img[:, 33:66] = [0, 255, 0]  # Green
        img[:, 66:] = [0, 0, 255]    # Blue
        
        score, details = metrics._calculate_color_metrics(img)
        
        assert "saturation_mean" in details
    
    def test_calculate_psnr_identical(self):
        """Test PSNR for identical images"""
        metrics = ImageMetrics()
        
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        psnr = metrics._calculate_psnr(img, img)
        
        assert psnr == 100.0  # Perfect match
    
    def test_calculate_psnr_different(self):
        """Test PSNR for different images"""
        metrics = ImageMetrics()
        
        img1 = np.zeros((100, 100, 3), dtype=np.uint8)
        img2 = np.ones((100, 100, 3), dtype=np.uint8) * 255
        
        psnr = metrics._calculate_psnr(img1, img2)
        
        assert psnr < 100.0
    
    def test_evaluate_invalid_image(self):
        """Test evaluation with invalid image"""
        metrics = ImageMetrics()
        
        with pytest.raises(ValueError):
            metrics.evaluate("/nonexistent/image.jpg")
    
    def test_compare_images(self, tmp_path):
        """Test image comparison"""
        metrics = ImageMetrics()
        
        img1 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        # Save to temp files
        path1 = tmp_path / "img1.jpg"
        path2 = tmp_path / "img2.jpg"
        cv2.imwrite(str(path1), img1)
        cv2.imwrite(str(path2), img2)
        
        result = metrics.compare_images(str(path1), str(path2))
        
        assert "psnr" in result
        assert "ssim" in result


class TestISPComparator:
    """Tests for ISP Comparator"""
    
    def test_comparison_config_defaults(self):
        """Test ComparisonConfig defaults"""
        config = ComparisonConfig()
        
        assert config.comp12_width == 3840
        assert config.comp12_height == 2160
        assert config.comp12_pattern == "RGGB"
        assert config.cmodel_threads == 8
    
    def test_comparison_result_dataclass(self):
        """Test ComparisonResult dataclass"""
        from datetime import datetime
        
        result = ComparisonResult(
            report_id="TEST_001",
            timestamp=datetime.now().isoformat()
        )
        
        assert result.report_id == "TEST_001"
        assert result.overall_status == "unknown"


class TestPerceptionModel:
    """Tests for Perception Model Interface"""
    
    def test_perception_init_without_model(self):
        """Test perception model initialization without model"""
        model = PerceptionModelInterface()
        
        assert not model.is_available()
        assert model.model_type == "yolov8"
    
    def test_perception_init_with_model(self):
        """Test perception model initialization with model path"""
        model = PerceptionModelInterface(
            model_path="/nonexistent/model.pt",
            model_type="yolov8"
        )
        
        assert not model.is_available()  # Should not load nonexistent
    
    def test_get_supported_classes(self):
        """Test getting supported classes"""
        model = PerceptionModelInterface()
        classes = model.get_supported_classes()
        
        assert "traffic_light" in classes
        assert "red_light" in classes
        assert "yellow_light" in classes
        assert "green_light" in classes
    
    def test_detect_returns_empty_without_model(self):
        """Test detection returns empty without model"""
        model = PerceptionModelInterface()
        
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        detections = model.detect_traffic_lights(img)
        
        assert detections == []
    
    def test_evaluate_detections_no_gt(self):
        """Test evaluation without ground truth"""
        model = PerceptionModelInterface()
        
        detection = DetectionResult(bbox=[10, 10, 50, 50], class_name="red_light", confidence=0.9)
        
        result = model.evaluate_detections([detection], None)
        
        assert result.detected is True
        assert result.confidence == 0.9
    
    def test_calculate_iou(self):
        """Test IoU calculation"""
        box1 = np.array([0, 0, 50, 50])
        box2 = np.array([25, 25, 75, 75])
        
        iou = PerceptionModelInterface._calculate_iou(box1, box2)
        
        assert 0 < iou < 1  # Should have some overlap


class TestPDFReportGenerator:
    """Tests for PDF Report Generator"""
    
    def test_generator_initialization(self):
        """Test generator initialization"""
        generator = PDFReportGenerator()
        assert generator is not None
    
    def test_format_status(self):
        """Test status formatting"""
        generator = PDFReportGenerator()
        
        assert "better" in generator._format_status("a_improved")
        assert "better" in generator._format_status("b_improved")
        assert "Error" in generator._format_status("error")


def test_import_module():
    """Test that module can be imported"""
    from isp_compare import Comp12Parser, CModelISP, ISPComparator, ImageMetrics
    assert Comp12Parser is not None
    assert CModelISP is not None
    assert ISPComparator is not None
    assert ImageMetrics is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
