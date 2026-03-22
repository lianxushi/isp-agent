#!/usr/bin/env python3
"""
ImageAnalyzer 单元测试
"""
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools.image_analyzer import ImageAnalyzer, AnalysisResult


class TestImageAnalyzer(unittest.TestCase):
    """ImageAnalyzer 测试类"""
    
    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        cls.test_dir = Path(__file__).parent.parent
        cls.test_image = cls.test_dir / 'test_image.jpg'
        cls.test_config = {
            'tools': {
                'image': {
                    'max_size_mb': 50
                }
            }
        }
    
    def test_01_analyzer_init(self):
        """测试分析器初始化"""
        analyzer = ImageAnalyzer()
        self.assertIsNotNone(analyzer)
        self.assertEqual(analyzer.max_size, 50 * 1024 * 1024)
        
        # 带配置初始化
        analyzer = ImageAnalyzer(self.test_config)
        self.assertEqual(analyzer.max_size, 50 * 1024 * 1024)
    
    def test_02_validate_file_not_found(self):
        """测试文件不存在异常"""
        from src.tools.image_analyzer import ImageValidationError
        
        analyzer = ImageAnalyzer()
        
        with self.assertRaises(ImageValidationError) as context:
            analyzer._validate('/nonexistent/path/image.jpg')
        
        self.assertIn('文件不存在', str(context.exception))
    
    def test_03_validate_unsupported_format(self):
        """测试不支持格式异常"""
        from src.tools.image_analyzer import ImageValidationError
        
        analyzer = ImageAnalyzer()
        
        # 创建一个临时测试文件
        test_file = self.test_dir / 'test.txt'
        test_file.write_text('test')
        
        try:
            with self.assertRaises(ImageValidationError) as context:
                analyzer._validate(str(test_file))
            
            self.assertIn('不支持的图像格式', str(context.exception))
        finally:
            if test_file.exists():
                test_file.unlink()
    
    def test_04_validate_file_too_large(self):
        """测试文件过大异常"""
        from src.tools.image_analyzer import ImageValidationError
        
        analyzer = ImageAnalyzer({'tools': {'image': {'max_size_mb': 0.001}}})
        
        if self.test_image.exists():
            with self.assertRaises(ImageValidationError) as context:
                analyzer._validate(str(self.test_image))
            
            self.assertIn('文件过大', str(context.exception))
    
    def test_05_get_basic_info(self):
        """测试获取基本信息"""
        analyzer = ImageAnalyzer()
        
        if not self.test_image.exists():
            self.skipTest("测试图像不存在")
        
        result = analyzer._get_basic_info(str(self.test_image))
        
        self.assertIsInstance(result, AnalysisResult)
        self.assertEqual(result.file_name, 'test_image.jpg')
        self.assertGreater(result.width, 0)
        self.assertGreater(result.height, 0)
        self.assertGreater(result.size_kb, 0)
    
    def test_06_analyze_histogram(self):
        """测试直方图分析"""
        analyzer = ImageAnalyzer()
        
        # 创建测试图像
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, :] = [128, 128, 128]  # 灰色
        
        result = analyzer._analyze_histogram(img)
        
        self.assertIn('gray', result)
        self.assertIn('blue', result)
        self.assertIn('green', result)
        self.assertIn('red', result)
        self.assertEqual(len(result['gray']), 256)
    
    def test_07_analyze_dynamic_range(self):
        """测试动态范围分析"""
        analyzer = ImageAnalyzer()
        
        # 创建测试图像
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[50:, :] = 255  # 下半部分白色
        
        result = analyzer._analyze_dynamic_range(img)
        
        self.assertIn('min', result)
        self.assertIn('max', result)
        self.assertIn('range', result)
        self.assertIn('useful_range', result)
        self.assertEqual(result['min'], 0)
        self.assertEqual(result['max'], 255)
    
    def test_08_analyze_noise(self):
        """测试噪声分析"""
        analyzer = ImageAnalyzer()
        
        # 创建测试图像
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        noise_level = analyzer._analyze_noise(img)
        
        self.assertIsInstance(noise_level, float)
        self.assertGreaterEqual(noise_level, 0)
    
    def test_09_analyze_brightness_contrast(self):
        """测试亮度和对比度分析"""
        analyzer = ImageAnalyzer()
        
        # 创建测试图像
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        
        brightness, contrast = analyzer._analyze_brightness_contrast(img)
        
        self.assertIsInstance(brightness, float)
        self.assertIsInstance(contrast, float)
        self.assertEqual(brightness, 128.0)
        self.assertEqual(contrast, 0.0)
    
    def test_10_analyze_color(self):
        """测试色彩分析"""
        analyzer = ImageAnalyzer()
        
        # 创建测试图像 - 红色
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, :, 2] = 255  # R通道
        
        result = analyzer._analyze_color(img)
        
        self.assertIn('B_mean', result)
        self.assertIn('G_mean', result)
        self.assertIn('R_mean', result)
        self.assertIn('white_balance', result)
        self.assertIn('saturation', result)
        self.assertEqual(result['R_mean'], 255.0)
        self.assertEqual(result['B_mean'], 0.0)
    
    def test_11_analysis_result_to_dict(self):
        """测试结果转字典"""
        result = AnalysisResult(
            file_path='/test/path.jpg',
            file_name='path.jpg',
            width=1920,
            height=1080,
            format='JPEG',
            size_bytes=1024000,
            size_kb=1000.0
        )
        
        result_dict = result.to_dict()
        
        self.assertIsInstance(result_dict, dict)
        self.assertEqual(result_dict['width'], 1920)
        self.assertEqual(result_dict['height'], 1080)
    
    def test_12_analysis_result_to_json(self):
        """测试结果转JSON"""
        result = AnalysisResult(
            file_path='/test/path.jpg',
            file_name='path.jpg',
            width=1920,
            height=1080,
            format='JPEG',
            size_bytes=1024000,
            size_kb=1000.0
        )
        
        result_json = result.to_json()
        
        self.assertIsInstance(result_json, str)
        self.assertIn('1920', result_json)
    
    def test_13_full_analyze(self):
        """测试完整分析流程"""
        analyzer = ImageAnalyzer()
        
        if not self.test_image.exists():
            self.skipTest("测试图像不存在")
        
        result = analyzer.analyze(str(self.test_image))
        
        self.assertIsInstance(result, AnalysisResult)
        self.assertGreater(result.width, 0)
        self.assertGreater(result.height, 0)
        self.assertIsNotNone(result.histogram)
        self.assertIsNotNone(result.dynamic_range)
        self.assertIsNotNone(result.noise_level)
        self.assertIsNotNone(result.brightness)
        self.assertIsNotNone(result.contrast)
        self.assertIsNotNone(result.color_analysis)
    
    def test_14_parse_exif_nonexistent(self):
        """测试EXIF解析 - 不存在的文件"""
        analyzer = ImageAnalyzer()
        
        result = analyzer._parse_exif('/nonexistent/image.jpg')
        
        self.assertIsNone(result)
    
    def test_15_process_invalid_method(self):
        """测试无效处理方法"""
        from src.tools.image_analyzer import ImageProcessingError
        
        analyzer = ImageAnalyzer()
        
        if not self.test_image.exists():
            self.skipTest("测试图像不存在")
        
        with self.assertRaises(ImageProcessingError) as context:
            analyzer.process(str(self.test_image), 'invalid_method')
        
        self.assertIn('不支持的处理方法', str(context.exception))


class TestImageAnalyzerProcess(unittest.TestCase):
    """图像处理功能测试"""
    
    @classmethod
    def setUpClass(cls):
        cls.test_dir = Path(__file__).parent.parent
        cls.test_image = cls.test_dir / 'test_image.jpg'
    
    def test_process_denoise(self):
        """测试降噪处理"""
        analyzer = ImageAnalyzer()
        
        if not self.test_image.exists():
            self.skipTest("测试图像不存在")
        
        result = analyzer.process(str(self.test_image), 'denoise')
        
        self.assertIn('降噪完成', result)
        self.assertIn('_denoised', result)
    
    def test_process_sharpen(self):
        """测试锐化处理"""
        analyzer = ImageAnalyzer()
        
        if not self.test_image.exists():
            self.skipTest("测试图像不存在")
        
        result = analyzer.process(str(self.test_image), 'sharpen')
        
        self.assertIn('锐化完成', result)
        self.assertIn('_sharpened', result)
    
    def test_process_color(self):
        """测试色彩调整"""
        analyzer = ImageAnalyzer()
        
        if not self.test_image.exists():
            self.skipTest("测试图像不存在")
        
        result = analyzer.process(str(self.test_image), 'color', {'temperature': 10})
        
        self.assertIn('色彩调整完成', result)


def run_tests():
    """运行所有测试"""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == '__main__':
    run_tests()
