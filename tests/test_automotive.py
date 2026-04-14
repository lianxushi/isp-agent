#!/usr/bin/env python3
"""
车载场景深度分析功能测试
Phase 3: ISO 16505合规检查、场景自适应分析、夜间成像专项
"""
import pytest
import numpy as np
import cv2
import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tools.automotive_analyzer import AutomotiveQualityAnalyzer


class TestISO16505Compliance:
    """ISO 16505合规检查测试"""
    
    @pytest.fixture
    def analyzer(self):
        return AutomotiveQualityAnalyzer()
    
    @pytest.fixture
    def test_image(self):
        """读取测试图像"""
        test_path = os.path.join(os.path.dirname(__file__), '..', 'test_color.jpg')
        if os.path.exists(test_path):
            return cv2.imread(test_path)
        # 创建测试图像
        return np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    
    def test_iso_check_resolution_pass(self, analyzer, test_image):
        """测试分辨率合规检查通过"""
        # 1280x960以上应该通过
        result = analyzer.check_iso_16505_compliance(test_image, {'fps': 30, 'fov': 120})
        
        assert 'compliant' in result
        assert 'checks' in result
        assert 'resolution' in result['checks']
        assert result['checks']['resolution']['status'] == 'pass'
    
    def test_iso_check_low_resolution_fail(self, analyzer):
        """测试低分辨率不合规"""
        # 创建低分辨率图像 (640x480)
        low_res = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        result = analyzer.check_iso_16505_compliance(low_res, {'fps': 30, 'fov': 120})
        
        assert result['checks']['resolution']['status'] == 'fail'
    
    def test_iso_check_fps_pass(self, analyzer, test_image):
        """测试帧率合规检查"""
        result = analyzer.check_iso_16505_compliance(test_image, {'fps': 30, 'fov': 120})
        
        assert result['checks']['framerate']['status'] == 'pass'
        assert result['checks']['framerate']['actual_fps'] == 30
    
    def test_iso_check_fps_fail(self, analyzer, test_image):
        """测试低帧率不合规"""
        result = analyzer.check_iso_16505_compliance(test_image, {'fps': 15, 'fov': 120})
        
        assert result['checks']['framerate']['status'] == 'fail'
    
    def test_iso_check_fov_pass(self, analyzer, test_image):
        """测试FOV合规检查"""
        result = analyzer.check_iso_16505_compliance(test_image, {'fps': 30, 'fov': 100})
        
        assert result['checks']['horizontal_fov']['status'] == 'pass'
    
    def test_iso_check_fov_fail(self, analyzer, test_image):
        """测试窄FOV不合规"""
        result = analyzer.check_iso_16505_compliance(test_image, {'fps': 30, 'fov': 50})
        
        assert result['checks']['horizontal_fov']['status'] == 'fail'
    
    def test_iso_check_dynamic_range(self, analyzer, test_image):
        """测试动态范围检查"""
        result = analyzer.check_iso_16505_compliance(test_image, {'fps': 30, 'fov': 120})
        
        assert 'dynamic_range' in result['checks']
        assert 'actual_db' in result['checks']['dynamic_range']
        assert 'status' in result['checks']['dynamic_range']
    
    def test_iso_check_overall_compliance(self, analyzer, test_image):
        """测试综合合规判定"""
        result = analyzer.check_iso_16505_compliance(
            test_image, 
            {'fps': 30, 'fov': 120}
        )
        
        assert 'overall_level' in result
        assert result['compliant'] == True or result['compliant'] == False
        assert 'summary' in result
        assert result['summary']['total_checks'] == 4


class TestADASSceneAnalysis:
    """场景自适应分析测试"""
    
    @pytest.fixture
    def analyzer(self):
        return AutomotiveQualityAnalyzer()
    
    @pytest.fixture
    def test_image(self):
        """读取测试图像"""
        test_path = os.path.join(os.path.dirname(__file__), '..', 'test_color.jpg')
        if os.path.exists(test_path):
            return cv2.imread(test_path)
        return np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    
    def test_forward_adas_scene(self, analyzer, test_image):
        """测试前视ADAS场景分析"""
        result = analyzer.analyze_adas_scene(test_image, 'forward_adas')
        
        assert result['scene_type'] == 'forward_adas'
        assert 'metrics' in result
        assert 'scores' in result
        assert 'lane_clarity' in result['metrics']
        assert 'edge_sharpness' in result['metrics']
        assert 'overall' in result['scores']
    
    def test_surround_view_scene(self, analyzer, test_image):
        """测试环视场景分析"""
        result = analyzer.analyze_adas_scene(test_image, 'surround_view')
        
        assert result['scene_type'] == 'surround_view'
        assert 'metrics' in result
        assert 'distortion_quality' in result['metrics']
        assert 'view_uniformity' in result['metrics']
    
    def test_dms_scene(self, analyzer, test_image):
        """测试DMS场景分析"""
        result = analyzer.analyze_adas_scene(test_image, 'dms')
        
        assert result['scene_type'] == 'dms'
        assert 'metrics' in result
        assert 'face_clarity' in result['metrics']
        assert 'ir_response' in result['metrics']
        assert 'illumination_evenness' in result['metrics']
    
    def test_cabin_scene(self, analyzer, test_image):
        """测试座舱场景分析"""
        result = analyzer.analyze_adas_scene(test_image, 'cabin')
        
        assert result['scene_type'] == 'cabin'
        assert 'metrics' in result
        assert 'color_accuracy' in result['metrics']
        assert 'skin_tone_fidelity' in result['metrics']
        assert 'white_balance' in result['metrics']
    
    def test_unknown_scene(self, analyzer, test_image):
        """测试未知场景类型"""
        result = analyzer.analyze_adas_scene(test_image, 'unknown_scene')
        
        assert 'error' in result


class TestLowLightAnalysis:
    """夜间成像专项测试"""
    
    @pytest.fixture
    def analyzer(self):
        return AutomotiveQualityAnalyzer()
    
    def test_low_light_normal_image(self, analyzer):
        """测试正常亮度图像"""
        # 创建正常亮度图像
        normal_img = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)
        
        result = analyzer.analyze_low_light(normal_img)
        
        assert 'overall_score' in result
        assert 'metrics' in result
        assert 'brightness' in result['metrics']
        assert 'noise_level' in result['metrics']
        assert 'shadow_gain' in result['metrics']
    
    def test_low_light_dark_image(self, analyzer):
        """测试低光图像"""
        # 创建低光图像
        dark_img = np.random.randint(5, 30, (480, 640, 3), dtype=np.uint8)
        
        result = analyzer.analyze_low_light(dark_img)
        
        assert result['analysis']['low_light_detected'] == True
        assert result['metrics']['brightness']['level'] in ['偏暗', '低光']
    
    def test_low_light_noise_analysis(self, analyzer):
        """测试噪声分析"""
        # 高噪声图像
        noisy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        noisy_img = noisy_img.astype(np.float32) + np.random.normal(0, 50, noisy_img.shape)
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
        
        result = analyzer.analyze_low_light(noisy_img)
        
        assert 'noise_level' in result['metrics']
    
    def test_low_light_dynamic_range(self, analyzer):
        """测试动态范围分析"""
        img = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
        
        result = analyzer.analyze_low_light(img)
        
        assert 'dynamic_range_db' in result['metrics']
        assert result['metrics']['dynamic_range_db']['value'] >= 0


class TestNumPyImplementation:
    """纯NumPy实现验证"""
    
    @pytest.fixture
    def analyzer(self):
        return AutomotiveQualityAnalyzer()
    
    def test_numpy_grayscale_conversion(self, analyzer):
        """测试灰度转换NumPy实现"""
        # 创建测试图像 (BGR格式，如cv2读取)
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # 手动灰度转换 (BGR顺序)
        gray_manual = (0.114 * img[:,:,0] + 0.587 * img[:,:,1] + 0.299 * img[:,:,2])
        
        # cv2灰度转换
        gray_cv2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        # 应该接近
        assert np.allclose(gray_manual, gray_cv2, atol=2)
    
    def test_numpy_laplacian_variance(self, analyzer):
        """测试NumPy实现的Laplacian方差"""
        img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        
        # 调用内部方法
        score = analyzer._evaluate_edge_sharpness(img)
        
        assert 0 <= score <= 100
    
    def test_numpy_hsv_conversion(self, analyzer):
        """测试NumPy HSV转换"""
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # 调用内部方法
        hsv = analyzer._rgb_to_hsv(img)
        
        assert hsv.shape == img.shape
        assert hsv.dtype == np.float32


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-k', 'automotive'])
