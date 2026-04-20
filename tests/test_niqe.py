#!/usr/bin/env python3
"""
NIQE 单元测试
测试 NIQE 特征提取和质量评估功能
"""
import os
import sys
import unittest
from pathlib import Path
import numpy as np
import cv2

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools.niqe import (
    NIQE,
    compute_niqe_features,
    assess_quality_niqe,
    _gamma,
    _kurtosis,
    _skewness,
    _estimate_ggd_param,
    _estimate_agd_param,
    _compute_mscn,
    _compute_pairwise_products,
    _fit_mvg_mean_cov,
    _mvg_negative_log_likelihood,
)


class TestNIQEHelperFunctions(unittest.TestCase):
    """NIQE 辅助函数测试"""

    def test_gamma_function_basic(self):
        """测试 Gamma 函数基本计算"""
        self.assertAlmostEqual(_gamma(1.0), 1.0, places=5)
        self.assertAlmostEqual(_gamma(2.0), 1.0, places=5)
        self.assertAlmostEqual(_gamma(3.0), 2.0, places=5)
        self.assertAlmostEqual(_gamma(4.0), 6.0, places=5)

    def test_gamma_function_edge_cases(self):
        """测试 Gamma 函数边界情况"""
        self.assertAlmostEqual(_gamma(0.5), 1.77245, places=3)
        self.assertEqual(_gamma(0.0), float('inf'))
        self.assertEqual(_gamma(-1.0), float('inf'))

    def test_kurtosis_gaussian(self):
        """测试峰度计算 - 高斯分布应接近0"""
        np.random.seed(42)
        data = np.random.randn(10000)
        k = _kurtosis(data, fisher=True)
        self.assertAlmostEqual(k, 0.0, places=1)

    def test_kurtosis_insufficient_data(self):
        """测试峰度计算 - 数据不足时返回0"""
        k = _kurtosis(np.array([1, 2, 3]), fisher=True)
        self.assertEqual(k, 0.0)

    def test_kurtosis_nan_handling(self):
        """测试峰度计算 - NaN数据处理"""
        data = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        k = _kurtosis(data, fisher=True)
        self.assertIsInstance(k, float)
        self.assertFalse(np.isnan(k))

    def test_skewness_gaussian(self):
        """测试偏度计算 - 高斯分布应接近0"""
        np.random.seed(42)
        data = np.random.randn(10000)
        s = _skewness(data)
        self.assertAlmostEqual(s, 0.0, places=1)

    def test_skewness_insufficient_data(self):
        """测试偏度计算 - 数据不足时返回0"""
        s = _skewness(np.array([1, 2]))
        self.assertEqual(s, 0.0)

    def test_estimate_ggd_param(self):
        """测试 GGD 参数估计"""
        np.random.seed(42)
        data = np.random.randn(10000)
        alpha, beta = _estimate_ggd_param(data)
        self.assertIsInstance(alpha, float)
        self.assertIsInstance(beta, float)
        self.assertGreater(alpha, 0)

    def test_estimate_agd_param(self):
        """测试 AGD 参数估计"""
        np.random.seed(42)
        data = np.random.randn(10000)
        alpha_l, alpha_r, skew = _estimate_agd_param(data)
        self.assertIsInstance(alpha_l, float)
        self.assertIsInstance(alpha_r, float)
        self.assertIsInstance(skew, float)


class TestNIQEMSCCalculations(unittest.TestCase):
    """MSCN 计算测试"""

    def test_compute_mscn_shape(self):
        """测试 MSCN 输出形状"""
        img = np.random.rand(100, 100).astype(np.float32)
        mscn = _compute_mscn(img, kernel_size=7)
        self.assertEqual(mscn.shape, img.shape)

    def test_compute_mscn_mean_near_zero(self):
        """测试 MSCN 均值接近0"""
        np.random.seed(42)
        img = np.random.rand(100, 100).astype(np.float32) * 255
        mscn = _compute_mscn(img, kernel_size=7)
        self.assertAlmostEqual(np.mean(mscn), 0.0, places=1)

    def test_compute_mscn_std_near_one(self):
        """测试 MSCN 标准差接近1"""
        np.random.seed(42)
        img = np.random.rand(100, 100).astype(np.float32) * 255
        mscn = _compute_mscn(img, kernel_size=7)
        self.assertAlmostEqual(np.std(mscn), 1.0, places=1)

    def test_compute_pairwise_products(self):
        """测试邻居乘积计算"""
        mscn = np.random.randn(100, 100).astype(np.float32)
        products = _compute_pairwise_products(mscn)
        self.assertIn('H', products)
        self.assertIn('V', products)
        self.assertIn('D1', products)
        self.assertIn('D2', products)
        # H应该比mscn少一列
        self.assertEqual(products['H'].shape[1], mscn.shape[1] - 1)


class TestNIQEMVGModel(unittest.TestCase):
    """MVG 模型测试"""

    def test_fit_mvg_mean_cov(self):
        """测试 MVG 均值和协方差拟合"""
        np.random.seed(42)
        features = np.random.randn(100, 10)
        mean, cov = _fit_mvg_mean_cov(features)
        self.assertEqual(mean.shape, (10,))
        self.assertEqual(cov.shape, (10, 10))

    def test_fit_mvg_single_sample(self):
        """测试单样本 MVG 拟合"""
        features = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mean, cov = _fit_mvg_mean_cov(features)
        self.assertEqual(len(mean), 5)
        self.assertEqual(cov.shape, (5, 5))

    def test_mvg_nll_positive(self):
        """测试 MVG NLL 为正值"""
        features = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mean = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        cov = np.eye(5) * 0.5
        nll = _mvg_negative_log_likelihood(features, mean, cov)
        self.assertIsInstance(nll, float)
        self.assertGreater(nll, 0)


class TestNIQEClass(unittest.TestCase):
    """NIQE 类测试"""

    def setUp(self):
        """创建测试图像"""
        self.niqe = NIQE()
        # 彩色图像
        self.test_img_color = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        # 灰度图像
        self.test_img_gray = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        # 标准图像
        self.std_img_path = Path(__file__).parent.parent / 'test_gray.jpg'

    def test_01_niqe_init(self):
        """测试 NIQE 初始化"""
        niqe = NIQE()
        self.assertIsNotNone(niqe._scales)

    def test_02_extract_features_single_scale(self):
        """测试单尺度特征提取"""
        gray = self.test_img_gray.astype(np.float32)
        features = self.niqe._extract_features_single_scale(gray, kernel_size=7)
        self.assertIsInstance(features, list)
        self.assertGreater(len(features), 0)

    def test_03_extract_features_multi_scale(self):
        """测试多尺度特征提取"""
        gray = self.test_img_gray.astype(np.float32)
        features = self.niqe._extract_features_multi_scale(gray)
        self.assertIsInstance(features, list)
        self.assertGreaterEqual(len(features), 30)

    def test_04_extract_features_color_image(self):
        """测试彩色图像特征提取"""
        result = self.niqe.extract_features(self.test_img_color)
        self.assertIn('features', result)
        self.assertIn('num_features', result)
        self.assertIn('mscn_analysis', result)

    def test_05_extract_features_gray_image(self):
        """测试灰度图像特征提取"""
        result = self.niqe.extract_features(self.test_img_gray)
        self.assertIn('features', result)
        self.assertIn('num_features', result)
        self.assertIn('mscn_analysis', result)

    def test_06_assess_quality_structure(self):
        """测试质量评估返回结构"""
        result = self.niqe.assess(self.test_img_color)
        required_fields = ['niqe_score', 'quality_level', 'mscn_analysis', 'naturalness']
        for field in required_fields:
            self.assertIn(field, result)

    def test_07_assess_quality_field_types(self):
        """测试质量评估字段类型"""
        result = self.niqe.assess(self.test_img_color)
        self.assertIsInstance(result['niqe_score'], float)
        self.assertIsInstance(result['quality_level'], str)

    def test_08_assess_quality_score_range(self):
        """测试质量分数范围"""
        result = self.niqe.assess(self.test_img_color)
        self.assertGreaterEqual(result['niqe_score'], 0)
        self.assertLessEqual(result['niqe_score'], 100)

    def test_09_assess_quality_level_values(self):
        """测试质量等级值"""
        result = self.niqe.assess(self.test_img_color)
        valid_levels = ['excellent', 'good', 'fair', 'poor', 'bad']
        self.assertIn(result['quality_level'], valid_levels)

    def test_10_mscn_analysis_fields(self):
        """测试 MSCN 分析字段"""
        result = self.niqe.assess(self.test_img_color)
        mscn = result['mscn_analysis']
        self.assertIn('mean', mscn)
        self.assertIn('std', mscn)
        self.assertIn('kurtosis', mscn)
        self.assertIn('skewness', mscn)

    def test_11_naturalness_fields(self):
        """测试自然度分析字段"""
        result = self.niqe.assess(self.test_img_color)
        nat = result['naturalness']
        self.assertIn('naturalness_score', nat)
        self.assertIn('is_natural', nat)

    def test_12_real_image_assessment(self):
        """测试真实图像评估"""
        if self.std_img_path.exists():
            img = cv2.imread(str(self.std_img_path))
            result = self.niqe.assess(img)
            self.assertIn('niqe_score', result)


class TestNIQEConvenienceFunctions(unittest.TestCase):
    """NIQE 便捷函数测试"""

    def setUp(self):
        self.std_img_path = Path(__file__).parent.parent / 'test_gray.jpg'

    def test_01_compute_niqe_features_returns_list(self):
        """测试 compute_niqe_features 返回列表"""
        if not self.std_img_path.exists():
            self.skipTest("Test image not found")
        img = cv2.imread(str(self.std_img_path))
        features = compute_niqe_features(img)
        self.assertIsInstance(features, list)
        self.assertGreater(len(features), 0)

    def test_02_assess_quality_niqe_returns_dict(self):
        """测试 assess_quality_niqe 返回字典"""
        if not self.std_img_path.exists():
            self.skipTest("Test image not found")
        img = cv2.imread(str(self.std_img_path))
        result = assess_quality_niqe(img)
        self.assertIsInstance(result, dict)


class TestNIQEEdgeCases(unittest.TestCase):
    """NIQE 边界情况测试"""

    def test_4channel_rgba_input(self):
        """测试 4通道 RGBA 输入"""
        niqe = NIQE()
        img = np.random.randint(0, 256, (100, 100, 4), dtype=np.uint8)
        result = niqe.assess(img)
        self.assertIn('niqe_score', result)

    def test_all_zeros_image(self):
        """测试全零图像"""
        niqe = NIQE()
        img = np.zeros((100, 100), dtype=np.uint8)
        result = niqe.assess(img)
        self.assertIn('niqe_score', result)

    def test_all_same_value_image(self):
        """测试相同值图像"""
        niqe = NIQE()
        img = np.ones((100, 100), dtype=np.uint8) * 128
        result = niqe.assess(img)
        self.assertIn('niqe_score', result)

    def test_extreme_brightness_image(self):
        """测试极端亮度图像"""
        niqe = NIQE()
        img = np.ones((100, 100), dtype=np.uint8) * 255
        result = niqe.assess(img)
        self.assertIn('niqe_score', result)

    def test_color_assess_quality(self):
        """测试彩色图像评估"""
        niqe = NIQE()
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        result = niqe.assess(img)
        self.assertIn('niqe_score', result)

    def test_gray_assess_quality(self):
        """测试灰度图像评估"""
        niqe = NIQE()
        img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        result = niqe.assess(img)
        self.assertIn('niqe_score', result)

    def test_float_input_converted(self):
        """测试浮点输入转换"""
        niqe = NIQE()
        img = np.random.rand(100, 100).astype(np.float32) * 255
        result = niqe.assess(img)
        self.assertIn('niqe_score', result)

    def test_small_image_64x64(self):
        """测试小图像"""
        niqe = NIQE()
        img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        result = niqe.assess(img)
        self.assertIn('niqe_score', result)

    def test_large_image_480x640(self):
        """测试大图像"""
        niqe = NIQE()
        img = np.random.randint(0, 256, (480, 640), dtype=np.uint8)
        result = niqe.assess(img)
        self.assertIn('niqe_score', result)

    def test_tiny_image_32x32(self):
        """测试极小图像"""
        niqe = NIQE()
        img = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
        result = niqe.assess(img)
        self.assertIn('niqe_score', result)

    def test_single_row_image(self):
        """测试单行图像"""
        niqe = NIQE()
        img = np.random.randint(0, 256, (1, 100), dtype=np.uint8)
        result = niqe.assess(img)
        self.assertIn('niqe_score', result)

    def test_single_column_image(self):
        """测试单列图像"""
        niqe = NIQE()
        img = np.random.randint(0, 256, (100, 1), dtype=np.uint8)
        result = niqe.assess(img)
        self.assertIn('niqe_score', result)

    def test_gradient_image(self):
        """测试梯度图像"""
        niqe = NIQE()
        img = np.tile(np.arange(256, dtype=np.uint8), (100, 1))
        result = niqe.assess(img)
        self.assertIn('niqe_score', result)

    def test_high_contrast_image(self):
        """测试高对比度图像"""
        niqe = NIQE()
        img = np.zeros((100, 100), dtype=np.uint8)
        img[:, :50] = 255
        result = niqe.assess(img)
        self.assertIn('niqe_score', result)

    def test_uniform_image(self):
        """测试均匀图像"""
        niqe = NIQE()
        img = np.full((100, 100), 128, dtype=np.uint8)
        result = niqe.assess(img)
        self.assertIn('niqe_score', result)

    def test_non_square_image(self):
        """测试非正方图像"""
        niqe = NIQE()
        img = np.random.randint(0, 256, (200, 100), dtype=np.uint8)
        result = niqe.assess(img)
        self.assertIn('niqe_score', result)

    def test_tall_image(self):
        """测试高图像"""
        niqe = NIQE()
        img = np.random.randint(0, 256, (200, 100), dtype=np.uint8)
        result = niqe.assess(img)
        self.assertIn('niqe_score', result)

    def test_wide_image(self):
        """测试宽图像"""
        niqe = NIQE()
        img = np.random.randint(0, 256, (100, 200), dtype=np.uint8)
        result = niqe.assess(img)
        self.assertIn('niqe_score', result)

    def test_mscn_with_single_value(self):
        """测试单一值图像的MSCN"""
        img = np.ones((100, 100), dtype=np.float32) * 128
        mscn = _compute_mscn(img, kernel_size=7)
        self.assertEqual(mscn.shape, img.shape)

    def test_mscn_with_uniform_image(self):
        """测试均匀图像的MSCN"""
        img = np.ones((100, 100), dtype=np.float32) * 128
        mscn = _compute_mscn(img, kernel_size=7)
        self.assertEqual(mscn.shape, img.shape)

    def test_assess_with_invalid_input(self):
        """测试无效输入"""
        niqe = NIQE()
        with self.assertRaises(Exception):
            niqe.assess(None)


class TestNIQEWithRealImage(unittest.TestCase):
    """NIQE 真实图像测试"""

    def setUp(self):
        self.std_img_path = Path(__file__).parent.parent / 'test_gray.jpg'
        self.color_img_path = Path(__file__).parent.parent / 'test_color.jpg'

    @unittest.skipUnless(Path(__file__).parent.parent.joinpath('test_gray.jpg').exists(), "Test image not found")
    def test_gray_image_scoring(self):
        """测试灰度图像评分"""
        if not self.std_img_path.exists():
            self.skipTest("Test image not found")
        img = cv2.imread(str(self.std_img_path), cv2.IMREAD_GRAYSCALE)
        niqe = NIQE()
        result = niqe.assess(img)
        self.assertGreater(result['niqe_score'], 0)
        self.assertLessEqual(result['niqe_score'], 100)

    @unittest.skipUnless(Path(__file__).parent.parent.joinpath('test_color.jpg').exists(), "Test image not found")
    def test_color_image_scoring(self):
        """测试彩色图像评分"""
        if not self.color_img_path.exists():
            self.skipTest("Test image not found")
        img = cv2.imread(str(self.color_img_path))
        niqe = NIQE()
        result = niqe.assess(img)
        self.assertGreater(result['niqe_score'], 0)
        self.assertLessEqual(result['niqe_score'], 100)

    @unittest.skipUnless(Path(__file__).parent.parent.joinpath('test_gray.jpg').exists(), "Test image not found")
    def test_feature_count_reasonable(self):
        """测试特征数量合理"""
        if not self.std_img_path.exists():
            self.skipTest("Test image not found")
        img = cv2.imread(str(self.std_img_path))
        niqe = NIQE()
        result = niqe.extract_features(img)
        # NIQE特征应该在30-50维之间
        self.assertGreater(result['num_features'], 20)
        self.assertLess(result['num_features'], 60)


if __name__ == '__main__':
    unittest.main()
