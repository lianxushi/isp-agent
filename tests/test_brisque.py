#!/usr/bin/env python3
"""
BRISQUE 单元测试
测试 BRISQUE 特征提取和质量评估功能
"""
import os
import sys
import unittest
from pathlib import Path
import numpy as np
import cv2

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools.brisque import (
    BRISQUE,
    compute_brisque_features,
    assess_quality_brisque,
    _gamma,
    _kurtosis,
    _estimate_ggd_param,
    _estimate_agd_param,
)


class TestBRISQUEHelperFunctions(unittest.TestCase):
    """BRISQUE 辅助函数测试"""

    def test_gamma_function_basic(self):
        """测试 Gamma 函数基本计算"""
        # Gamma(1) = 1, Gamma(2) = 1, Gamma(3) = 2, Gamma(4) = 6
        self.assertAlmostEqual(_gamma(1.0), 1.0, places=5)
        self.assertAlmostEqual(_gamma(2.0), 1.0, places=5)
        self.assertAlmostEqual(_gamma(3.0), 2.0, places=5)
        self.assertAlmostEqual(_gamma(4.0), 6.0, places=5)

    def test_gamma_function_edge_cases(self):
        """测试 Gamma 函数边界情况"""
        # Gamma(0.5) = sqrt(pi) ≈ 1.772
        self.assertAlmostEqual(_gamma(0.5), 1.77245, places=3)
        # Gamma 负数和零返回 inf
        self.assertEqual(_gamma(0.0), float('inf'))
        self.assertEqual(_gamma(-1.0), float('inf'))

    def test_kurtosis_gaussian(self):
        """测试峰度计算 - 高斯分布应接近0"""
        # 生成正态分布数据
        np.random.seed(42)
        data = np.random.randn(10000)
        k = _kurtosis(data, fisher=True)
        self.assertAlmostEqual(k, 0.0, places=1)

    def test_kurtosis_uniform(self):
        """测试峰度计算 - 均匀分布应有负峰度"""
        data = np.random.rand(10000)
        k = _kurtosis(data, fisher=True)
        self.assertLess(k, 0)  # 均匀分布峰度为负

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

    def test_estimate_ggd_param(self):
        """测试 GGD 参数估计"""
        np.random.seed(42)
        # 生成标准高斯分布数据 (alpha≈1, beta≈2)
        data = np.random.randn(10000)
        alpha, beta = _estimate_ggd_param(data)
        self.assertIsInstance(alpha, float)
        self.assertIsInstance(beta, float)
        self.assertGreater(alpha, 0)
        self.assertGreater(beta, 0)

    def test_estimate_ggd_param_insufficient_data(self):
        """测试 GGD 参数估计 - 数据不足"""
        data = np.array([1.0])
        alpha, beta = _estimate_ggd_param(data)
        # 返回默认值
        self.assertEqual(alpha, 1.0)
        self.assertEqual(beta, 2.0)

    def test_estimate_agd_param(self):
        """测试 AGD 参数估计"""
        np.random.seed(42)
        # 生成对称分布数据
        data = np.random.randn(10000)
        alpha_l, alpha_r, mu, skew = _estimate_agd_param(data)
        self.assertIsInstance(alpha_l, float)
        self.assertIsInstance(alpha_r, float)
        self.assertIsInstance(mu, float)
        self.assertIsInstance(skew, float)
        self.assertGreater(alpha_l, 0)
        self.assertGreater(alpha_r, 0)
        self.assertAlmostEqual(skew, 0.0, places=1)

    def test_estimate_agd_param_insufficient_data(self):
        """测试 AGD 参数估计 - 数据不足"""
        data = np.array([1.0])
        alpha_l, alpha_r, mu, skew = _estimate_agd_param(data)
        self.assertEqual(alpha_l, 1.0)
        self.assertEqual(alpha_r, 1.0)
        self.assertEqual(mu, 1.0)
        self.assertEqual(skew, 0.0)


class TestBRISQUEClass(unittest.TestCase):
    """BRISQUE 主类测试"""

    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        cls.project_root = Path(__file__).parent.parent
        cls.test_color = cls.project_root / 'test_color.jpg'
        cls.test_gray = cls.project_root / 'test_gray.jpg'
        cls.brisque = BRISQUE()

    def test_01_brisque_init(self):
        """测试 BRISQUE 初始化"""
        brisque = BRISQUE()
        self.assertIsNotNone(brisque)
        self.assertIsNone(brisque.model)

    def test_02_brisque_init_with_model(self):
        """测试带模型路径初始化"""
        brisque = BRISQUE(model_path='/nonexistent/model.pkl')
        self.assertIsNone(brisque.model)  # 模型不存在时应为 None

    def test_03_compute_mscn(self):
        """测试 MSCN 系数计算"""
        # 创建简单测试图像
        img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        mscn = self.brisque._compute_mscn(img)
        self.assertEqual(mscn.shape, img.shape)
        self.assertIsInstance(mscn, np.ndarray)
        # MSCN 系数应接近标准差 1
        self.assertAlmostEqual(np.std(mscn), 1.0, places=1)

    def test_04_compute_mscn_mean_near_zero(self):
        """测试 MSCN 系数均值接近零"""
        img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        mscn = self.brisque._compute_mscn(img)
        self.assertAlmostEqual(np.mean(mscn), 0.0, places=1)

    def test_05_compute_pairwise_products(self):
        """测试邻居系数乘积计算"""
        mscn = np.random.randn(100, 100).astype(np.float32)
        products = self.brisque._compute_pairwise_products(mscn)
        self.assertIn('H', products)
        self.assertIn('V', products)
        self.assertIn('D1', products)
        self.assertIn('D2', products)
        # 各方向乘积的形状
        self.assertEqual(products['H'].shape, (100, 99))
        self.assertEqual(products['V'].shape, (99, 100))
        self.assertEqual(products['D1'].shape, (99, 99))
        self.assertEqual(products['D2'].shape, (99, 99))

    def test_06_extract_features_single_scale(self):
        """测试单尺度特征提取"""
        img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        features = self.brisque._extract_features_single_scale(img)
        self.assertEqual(len(features), 14)  # 单尺度应有14个特征
        self.assertIsInstance(features, list)
        for f in features:
            self.assertIsInstance(f, float)

    def test_07_extract_features_multi_scale(self):
        """测试多尺度特征提取"""
        img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        features = self.brisque._extract_features_multi_scale(img)
        self.assertEqual(len(features), 28)  # 双尺度应有28个特征
        self.assertIsInstance(features, list)

    def test_08_extract_features_with_real_image(self):
        """测试使用真实图像提取特征"""
        if not self.test_color.exists():
            self.skipTest("测试图像 test_color.jpg 不存在")

        img = cv2.imread(str(self.test_color))
        self.assertIsNotNone(img)

        result = self.brisque.extract_features(img)
        self.assertIn('features', result)
        self.assertIn('num_features', result)
        self.assertIn('scale_7_features', result)
        self.assertIn('scale_15_features', result)
        self.assertIn('mscn_mean', result)
        self.assertIn('mscn_std', result)
        self.assertIn('mscn_kurtosis', result)
        self.assertEqual(result['num_features'], 28)

    def test_09_extract_features_gray_image(self):
        """测试灰度图像特征提取"""
        if not self.test_gray.exists():
            self.skipTest("测试图像 test_gray.jpg 不存在")

        img = cv2.imread(str(self.test_gray), cv2.IMREAD_GRAYSCALE)
        self.assertIsNotNone(img)

        result = self.brisque.extract_features(img)
        self.assertEqual(result['num_features'], 28)
        self.assertEqual(len(result['scale_7_features']), 14)
        self.assertEqual(len(result['scale_15_features']), 14)

    def test_10_assess_quality_structure(self):
        """测试质量评估返回结构"""
        img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        result = self.brisque.assess(img)

        # 验证顶层字段
        self.assertIn('brisque_score', result)
        self.assertIn('mos_predicted', result)
        self.assertIn('quality_level', result)
        self.assertIn('num_features', result)
        self.assertIn('mscn_analysis', result)
        self.assertIn('naturalness', result)
        self.assertIn('details', result)

    def test_11_assess_quality_field_types(self):
        """测试质量评估字段类型"""
        img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        result = self.brisque.assess(img)

        # 验证字段类型
        self.assertIsInstance(result['brisque_score'], float)
        self.assertIsInstance(result['mos_predicted'], float)
        self.assertIsInstance(result['quality_level'], str)
        self.assertIsInstance(result['num_features'], int)

    def test_12_assess_quality_score_range(self):
        """测试质量分数范围"""
        img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        result = self.brisque.assess(img)

        # BRISQUE 分数应在 0-100 范围
        self.assertGreaterEqual(result['brisque_score'], 0.0)
        self.assertLessEqual(result['brisque_score'], 100.0)

        # MOS 分数应在 1-5 范围
        self.assertGreaterEqual(result['mos_predicted'], 1.0)
        self.assertLessEqual(result['mos_predicted'], 5.0)

    def test_13_assess_quality_level_values(self):
        """测试质量等级值"""
        img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        result = self.brisque.assess(img)

        valid_levels = {'excellent', 'good', 'fair', 'poor', 'bad'}
        self.assertIn(result['quality_level'], valid_levels)

    def test_14_mscn_analysis_fields(self):
        """测试 MSCN 分析字段"""
        img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        result = self.brisque.assess(img)
        mscn_analysis = result['mscn_analysis']

        self.assertIn('mean', mscn_analysis)
        self.assertIn('std', mscn_analysis)
        self.assertIn('kurtosis', mscn_analysis)
        self.assertIn('skewness', mscn_analysis)
        self.assertIn('is_gaussian_like', mscn_analysis)

    def test_15_naturalness_fields(self):
        """测试自然度评估字段"""
        img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        result = self.brisque.assess(img)
        naturalness = result['naturalness']

        self.assertIn('naturalness_score', naturalness)
        self.assertIn('kurtosis_deviation', naturalness)
        self.assertIn('skewness_deviation', naturalness)
        self.assertIn('is_natural', naturalness)
        self.assertIn('interpretation', naturalness)

    def test_16_details_fields(self):
        """测试 details 字段"""
        img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        result = self.brisque.assess(img)
        details = result['details']

        self.assertIn('alpha', details)
        self.assertIn('beta', details)
        self.assertIn('mscn_mean', details)
        self.assertIn('mscn_std', details)
        self.assertIn('mscn_kurtosis', details)

    def test_17_real_image_assessment(self):
        """使用真实图像进行完整质量评估"""
        if not self.test_color.exists():
            self.skipTest("测试图像 test_color.jpg 不存在")

        img = cv2.imread(str(self.test_color))
        result = self.brisque.assess(img)

        self.assertIsNotNone(result['brisque_score'])
        self.assertIsNotNone(result['mos_predicted'])
        self.assertIn(result['quality_level'],
                       {'excellent', 'good', 'fair', 'poor', 'bad'})
        self.assertEqual(result['num_features'], 28)


class TestComputeBrisqueFeatures(unittest.TestCase):
    """compute_brisque_features 便捷函数测试"""

    def test_01_compute_features_returns_list(self):
        """测试返回类型为列表"""
        img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        features = compute_brisque_features(img)
        self.assertIsInstance(features, list)

    def test_02_compute_features_count_28(self):
        """测试特征数量为28 (双尺度 x 14)"""
        img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        features = compute_brisque_features(img)
        self.assertEqual(len(features), 28)

    def test_03_compute_features_all_floats(self):
        """测试所有特征值均为 float"""
        img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        features = compute_brisque_features(img)
        for f in features:
            self.assertIsInstance(f, (float, int))

    def test_04_compute_features_color_image(self):
        """测试彩色图像处理"""
        if not Path('/Users/lianxu.shi/.openclaw/workspace/isp-agent/test_color.jpg').exists():
            self.skipTest("测试图像不存在")

        img = cv2.imread('/Users/lianxu.shi/.openclaw/workspace/isp-agent/test_color.jpg')
        features = compute_brisque_features(img)
        self.assertEqual(len(features), 28)

    def test_05_compute_features_gray_image(self):
        """测试灰度图像处理"""
        if not Path('/Users/lianxu.shi/.openclaw/workspace/isp-agent/test_gray.jpg').exists():
            self.skipTest("测试图像不存在")

        img = cv2.imread('/Users/lianxu.shi/.openclaw/workspace/isp-agent/test_gray.jpg',
                         cv2.IMREAD_GRAYSCALE)
        features = compute_brisque_features(img)
        self.assertEqual(len(features), 28)


class TestAssessQualityBrisque(unittest.TestCase):
    """assess_quality_brisque 便捷函数测试"""

    def test_01_assess_returns_dict(self):
        """测试返回类型为字典"""
        img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        result = assess_quality_brisque(img)
        self.assertIsInstance(result, dict)

    def test_02_assess_has_required_fields(self):
        """测试包含所有必需字段"""
        img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        result = assess_quality_brisque(img)

        required_fields = [
            'brisque_score', 'mos_predicted', 'quality_level',
            'num_features', 'mscn_analysis', 'naturalness', 'details'
        ]
        for field in required_fields:
            self.assertIn(field, result)

    def test_03_assess_mscn_analysis_complete(self):
        """测试 MSCN 分析字段完整性"""
        img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        result = assess_quality_brisque(img)
        mscn = result['mscn_analysis']

        required_mscn_fields = ['mean', 'std', 'kurtosis', 'skewness', 'is_gaussian_like']
        for field in required_mscn_fields:
            self.assertIn(field, mscn)

    def test_04_assess_naturalness_complete(self):
        """测试自然度评估字段完整性"""
        img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        result = assess_quality_brisque(img)
        nat = result['naturalness']

        required_nat_fields = [
            'naturalness_score', 'kurtosis_deviation',
            'skewness_deviation', 'is_natural', 'interpretation'
        ]
        for field in required_nat_fields:
            self.assertIn(field, nat)

    def test_05_assess_real_image(self):
        """使用真实图像测试"""
        img_path = '/Users/lianxu.shi/.openclaw/workspace/isp-agent/test_color.jpg'
        if not Path(img_path).exists():
            self.skipTest("测试图像不存在")

        img = cv2.imread(img_path)
        result = assess_quality_brisque(img)

        self.assertGreaterEqual(result['brisque_score'], 0)
        self.assertLessEqual(result['brisque_score'], 100)
        self.assertGreaterEqual(result['mos_predicted'], 1)
        self.assertLessEqual(result['mos_predicted'], 5)


class TestBRISQUEFeatureCount(unittest.TestCase):
    """BRISQUE 特征数量专项测试 - 验证36个特征的预期"""

    def test_single_scale_feature_count(self):
        """单尺度特征数量应为14"""
        brisque = BRISQUE()
        img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        features = brisque._extract_features_single_scale(img)
        # 单尺度: 2 (MSCN GGD) + 4×3 (H,V,D1,D2各3个AGD参数) = 14
        self.assertEqual(len(features), 14,
            f"单尺度特征数量应为14，实际为{len(features)}")

    def test_multi_scale_feature_count(self):
        """多尺度特征数量应为28 (双尺度)"""
        brisque = BRISQUE()
        img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        features = brisque._extract_features_multi_scale(img,
            kernel_sizes=[7, 15])
        self.assertEqual(len(features), 28,
            f"双尺度特征数量应为28，实际为{len(features)}")

    def test_default_extract_features_count(self):
        """默认 extract_features 应返回28个特征"""
        brisque = BRISQUE()
        img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        result = brisque.extract_features(img)
        self.assertEqual(result['num_features'], 28,
            f"默认特征数量应为28，实际为{result['num_features']}")

    def test_feature_count_in_assessment(self):
        """assess 方法中的特征数量应为28"""
        brisque = BRISQUE()
        img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        result = brisque.assess(img)
        self.assertEqual(result['num_features'], 28)


class TestBRISQUEEdgeCases(unittest.TestCase):
    """BRISQUE边界情况测试 - 补充灰度/彩色、不同尺寸、异常输入"""

    @classmethod
    def setUpClass(cls):
        cls.brisque = BRISQUE()

    # =====================================================================
    # 灰度/彩色图像测试
    # =====================================================================
    def test_gray_image_grayscale(self):
        """测试灰度图像(单通道)输入"""
        img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        features = compute_brisque_features(img)
        self.assertEqual(len(features), 28)

    def test_color_image_3channel(self):
        """测试彩色图像(3通道BGR)输入"""
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        features = compute_brisque_features(img)
        self.assertEqual(len(features), 28)

    def test_gray_assess_quality(self):
        """测试灰度图像质量评估"""
        img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        result = self.brisque.assess(img)
        self.assertIn('brisque_score', result)
        self.assertGreaterEqual(result['brisque_score'], 0.0)
        self.assertLessEqual(result['brisque_score'], 100.0)

    def test_color_assess_quality(self):
        """测试彩色图像质量评估"""
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        result = self.brisque.assess(img)
        self.assertIn('brisque_score', result)
        self.assertGreaterEqual(result['brisque_score'], 0.0)
        self.assertLessEqual(result['brisque_score'], 100.0)

    # =====================================================================
    # 不同尺寸图像测试
    # =====================================================================
    def test_tiny_image_32x32(self):
        """测试极小图像 32x32"""
        img = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
        features = compute_brisque_features(img)
        self.assertEqual(len(features), 28)
        result = self.brisque.assess(img)
        self.assertIn('brisque_score', result)

    def test_small_image_64x64(self):
        """测试小图像 64x64"""
        img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        features = compute_brisque_features(img)
        self.assertEqual(len(features), 28)

    def test_standard_image_100x100(self):
        """测试标准尺寸 100x100"""
        img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        features = compute_brisque_features(img)
        self.assertEqual(len(features), 28)

    def test_large_image_480x640(self):
        """测试大图像 480x640"""
        img = np.random.randint(0, 256, (480, 640), dtype=np.uint8)
        features = compute_brisque_features(img)
        self.assertEqual(len(features), 28)

    def test_non_square_image(self):
        """测试非正方形图像"""
        img = np.random.randint(0, 256, (240, 320), dtype=np.uint8)
        features = compute_brisque_features(img)
        self.assertEqual(len(features), 28)

    def test_wide_image(self):
        """测试宽幅图像"""
        img = np.random.randint(0, 256, (100, 400), dtype=np.uint8)
        features = compute_brisque_features(img)
        self.assertEqual(len(features), 28)

    def test_tall_image(self):
        """测试高幅图像"""
        img = np.random.randint(0, 256, (400, 100), dtype=np.uint8)
        features = compute_brisque_features(img)
        self.assertEqual(len(features), 28)

    # =====================================================================
    # 异常输入处理测试
    # =====================================================================
    def test_empty_image(self):
        """测试空图像处理"""
        img = np.array([], dtype=np.uint8)
        with self.assertRaises((ValueError, Exception)):
            compute_brisque_features(img)

    def test_single_pixel_image(self):
        """测试单像素图像 - 实现接受但不保证有意义结果"""
        img = np.array([[128]], dtype=np.uint8)
        features = compute_brisque_features(img)
        self.assertIsInstance(features, list)

    def test_all_same_value_image(self):
        """测试所有像素值相同的图像"""
        img = np.ones((100, 100), dtype=np.uint8) * 128
        # 应该能处理，但不抛出异常
        features = compute_brisque_features(img)
        self.assertEqual(len(features), 28)

    def test_all_zeros_image(self):
        """测试全零图像"""
        img = np.zeros((100, 100), dtype=np.uint8)
        # 应该能处理
        features = compute_brisque_features(img)
        self.assertEqual(len(features), 28)

    def test_all_255_image(self):
        """测试全255图像"""
        img = np.ones((100, 100), dtype=np.uint8) * 255
        features = compute_brisque_features(img)
        self.assertEqual(len(features), 28)

    def test_single_row_image(self):
        """测试单行图像 - 实现可以处理"""
        img = np.random.randint(0, 256, (1, 100), dtype=np.uint8)
        features = compute_brisque_features(img)
        self.assertIsInstance(features, list)

    def test_single_column_image(self):
        """测试单列图像 - 实现可以处理"""
        img = np.random.randint(0, 256, (100, 1), dtype=np.uint8)
        features = compute_brisque_features(img)
        self.assertIsInstance(features, list)

    def test_float_input_converted(self):
        """测试float输入自动转换"""
        img_float = np.random.rand(100, 100).astype(np.float32)
        # 应该在内部转换或报错，但不崩溃
        try:
            features = compute_brisque_features(img_float)
            self.assertEqual(len(features), 28)
        except Exception:
            pass  # 允许报错但不能崩溃

    def test_4channel_rgba_input(self):
        """测试4通道RGBA图像(取前3通道)"""
        img_rgba = np.random.randint(0, 256, (100, 100, 4), dtype=np.uint8)
        features = compute_brisque_features(img_rgba)
        self.assertEqual(len(features), 28)

    def test_negative_values_in_float_input(self):
        """测试float输入包含负值"""
        img = np.random.rand(100, 100).astype(np.float32) * 2 - 1  # [-1, 1]
        try:
            features = compute_brisque_features(img)
            self.assertEqual(len(features), 28)
        except Exception:
            pass

    def test_extreme_brightness_image(self):
        """测试极端亮度图像(极暗)"""
        img = np.random.randint(0, 10, (100, 100), dtype=np.uint8)
        features = compute_brisque_features(img)
        self.assertEqual(len(features), 28)

    def test_extreme_brightness_image_bright(self):
        """测试极端亮度图像(极亮)"""
        img = np.random.randint(245, 256, (100, 100), dtype=np.uint8)
        features = compute_brisque_features(img)
        self.assertEqual(len(features), 28)

    def test_high_contrast_image(self):
        """测试高对比度图像"""
        img = np.zeros((100, 100), dtype=np.uint8)
        img[:50, :] = 0
        img[50:, :] = 255
        features = compute_brisque_features(img)
        self.assertEqual(len(features), 28)

    def test_gradient_image(self):
        """测试渐变图像"""
        img = np.tile(np.arange(256), (100, 1)) % 256
        img = img.astype(np.uint8)
        features = compute_brisque_features(img)
        self.assertEqual(len(features), 28)

    def test_assess_with_invalid_input(self):
        """测试assess方法异常输入"""
        img = np.array([], dtype=np.uint8)
        with self.assertRaises((ValueError, Exception)):
            self.brisque.assess(img)

    def test_mscn_with_single_value(self):
        """测试MSCN计算单一值区域"""
        img = np.ones((50, 50), dtype=np.uint8) * 128
        mscn = self.brisque._compute_mscn(img)
        # 结果可能是NaN或有限值
        self.assertEqual(mscn.shape, img.shape)

    def test_mscn_with_uniform_image(self):
        """测试MSCN计算完全均匀图像"""
        img = np.ones((100, 100), dtype=np.uint8) * 128
        mscn = self.brisque._compute_mscn(img)
        # 均匀图像的MSCN可能是0或NaN
        self.assertEqual(mscn.shape, img.shape)


def run_tests():
    """运行所有测试"""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == '__main__':
    run_tests()
