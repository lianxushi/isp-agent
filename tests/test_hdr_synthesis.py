#!/usr/bin/env python3
"""
Phase 2.2: HDR合成与质量评估测试
"""
import os
import sys
import unittest
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools.raw_processor import (
    synthesize_hdr_exposures, align_exposures,
    hdr_synthesize, align_images
)
from src.tools.hdr_processor import HDRAnalyzer


class TestHDRSynthesis(unittest.TestCase):
    """HDR合成测试"""

    @classmethod
    def setUpClass(cls):
        cls.test_dir = Path(__file__).parent.parent
        cls.test_image_path = cls.test_dir / 'test_color.jpg'
        cls.img = cv2.imread(str(cls.test_image_path))
        if cls.img is None:
            raise FileNotFoundError(f"测试图像不存在: {cls.test_image_path}")
        cls.img_float = cls.img.astype(np.float32) / 255.0

    def test_hdr_synthesis_basic(self):
        """测试基本HDR合成"""
        # 模拟3帧不同曝光的图像
        exposures = [1/125.0, 1/60.0, 1/30.0]
        frames = [
            np.clip(self.img_float * 0.5, 0, 1),   # 欠曝
            self.img_float,                           # 正常
            np.clip(self.img_float * 2.0, 0, 1),    # 过曝
        ]

        hdr = synthesize_hdr_exposures(frames, exposures)

        self.assertEqual(hdr.dtype, np.float32)
        self.assertEqual(hdr.shape, self.img_float.shape)
        self.assertTrue(0.0 <= hdr.min() <= hdr.max() <= 1.0)
        print(f"HDR合成: shape={hdr.shape}, range=[{hdr.min():.4f}, {hdr.max():.4f}]")

    def test_hdr_synthesis_two_frames(self):
        """测试2帧HDR合成"""
        frames = [
            np.clip(self.img_float * 0.6, 0, 1),
            np.clip(self.img_float * 1.8, 0, 1),
        ]
        exposures = [1/200.0, 1/50.0]

        hdr = synthesize_hdr_exposures(frames, exposures)

        self.assertEqual(hdr.shape, self.img_float.shape)
        self.assertEqual(hdr.dtype, np.float32)

    def test_hdr_synthesis_single_frame_fails(self):
        """测试单帧HDR合成应报错"""
        frames = [self.img_float]
        exposures = [1/60.0]

        with self.assertRaises(ValueError):
            synthesize_hdr_exposures(frames, exposures)

    def test_hdr_synthesis_mismatched_lengths(self):
        """测试图像与曝光时间数量不匹配时报错"""
        frames = [self.img_float, self.img_float]
        exposures = [1/60.0]  # 少一个

        with self.assertRaises(ValueError):
            synthesize_hdr_exposures(frames, exposures)

    def test_hdr_synthesis_uint8_input(self):
        """测试uint8输入"""
        frames_uint8 = [
            np.clip(self.img * 60, 0, 255).astype(np.uint8),
            self.img,
            np.clip(self.img * 200, 0, 255).astype(np.uint8),
        ]
        exposures = [1/200.0, 1/60.0, 1/30.0]

        hdr = synthesize_hdr_exposures(frames_uint8, exposures)

        self.assertEqual(hdr.dtype, np.float32)
        self.assertEqual(hdr.shape, self.img.shape)
        print(f"uint8 HDR合成: shape={hdr.shape}, range=[{hdr.min():.4f}, {hdr.max():.4f}]")

    def test_align_exposures_basic(self):
        """测试图像对齐"""
        frames = [self.img_float, self.img_float, self.img_float]
        aligned = align_exposures(frames)

        self.assertEqual(len(aligned), 3)
        self.assertEqual(aligned[0].shape, self.img_float.shape)

    def test_align_single_image(self):
        """测试单帧对齐"""
        aligned = align_exposures([self.img_float])
        self.assertEqual(len(aligned), 1)
        self.assertEqual(aligned[0].shape, self.img_float.shape)

    def test_hdr_synthesize_convenience(self):
        """测试便捷函数"""
        frames = [self.img_float * 0.7, self.img_float * 1.5]
        exposures = [1/125.0, 1/30.0]

        hdr = hdr_synthesize(frames, exposures)
        self.assertEqual(hdr.dtype, np.float32)

    def test_align_images_convenience(self):
        """测试对齐便捷函数"""
        frames = [self.img_float, self.img_float]
        aligned = align_images(frames)
        self.assertEqual(len(aligned), 2)


class TestHDRQualityAssessment(unittest.TestCase):
    """HDR质量评估测试"""

    @classmethod
    def setUpClass(cls):
        cls.test_dir = Path(__file__).parent.parent
        cls.test_image_path = cls.test_dir / 'test_color.jpg'
        cls.img = cv2.imread(str(cls.test_image_path))
        if cls.img is None:
            raise FileNotFoundError(f"测试图像不存在: {cls.test_image_path}")
        cls.img_float = cls.img.astype(np.float32) / 255.0

    def test_calculate_dynamic_range(self):
        """测试动态范围计算"""
        analyzer = HDRAnalyzer(self.img_float)

        # test_color.jpg 应该是低动态范围图像
        dr = analyzer.calculate_dynamic_range()
        print(f"test_color.jpg 动态范围: {dr} stops")
        self.assertIsInstance(dr, (float, np.floating))
        self.assertGreaterEqual(float(dr), 0.0)

        # 高对比度假图像
        high_dr_img = np.zeros((100, 100), dtype=np.float32)
        high_dr_img[:50, :] = 0.01
        high_dr_img[50:, :] = 0.99
        dr_high = analyzer.calculate_dynamic_range(high_dr_img)
        print(f"高对比度图像动态范围: {dr_high} stops")
        self.assertGreater(float(dr_high), 5.0)

    def test_calculate_dynamic_range_uint8(self):
        """测试uint8输入"""
        analyzer = HDRAnalyzer(self.img)
        dr = analyzer.calculate_dynamic_range()
        self.assertIsInstance(dr, (float, np.floating))
        self.assertGreaterEqual(float(dr), 0.0)

    def test_evaluate_tone_mapping_basic(self):
        """测试Tone Mapping质量评估"""
        analyzer = HDRAnalyzer(self.img_float)

        # 原始HDR: 高动态范围
        hdr_orig = np.zeros_like(self.img_float)
        hdr_orig[:self.img_float.shape[0]//2] = 0.01
        hdr_orig[self.img_float.shape[0]//2:] = 0.99

        # TM后: LDR图像 (uint8)
        tm_ldr = (self.img_float * 255).astype(np.uint8)

        result = analyzer.evaluate_tone_mapping(hdr_orig, tm_ldr)

        self.assertIn('dynamic_range_original', result)
        self.assertIn('dynamic_range_tm', result)
        self.assertIn('dr_preservation_rate', result)
        self.assertIn('highlight_clipping', result)
        self.assertIn('shadow_detail', result)
        self.assertIn('contrast_preservation', result)
        self.assertIn('overall_score', result)
        self.assertIn('recommendations', result)

        print(f"Tone Mapping评估结果:")
        print(f"  原始DR: {result['dynamic_range_original']} stops")
        print(f"  TM后DR: {result['dynamic_range_tm']} stops")
        print(f"  DR保留率: {result['dr_preservation_rate']}%")
        print(f"  高光溢出: {result['highlight_clipping']}")
        print(f"  暗部评估: {result['shadow_detail']}")
        print(f"  对比度保留: {result['contrast_preservation']}")
        print(f"  综合评分: {result['overall_score']}/100")
        print(f"  建议: {result['recommendations']}")

        self.assertIsInstance(result['overall_score'], (float, np.floating))
        self.assertGreaterEqual(float(result['overall_score']), 0.0)
        self.assertLessEqual(float(result['overall_score']), 100.0)

    def test_evaluate_tone_mapping_float_input(self):
        """测试float格式TM图像输入"""
        analyzer = HDRAnalyzer(self.img_float)

        hdr_orig = np.random.rand(*self.img_float.shape).astype(np.float32) * 0.8 + 0.1
        tm_float = self.img_float.copy()

        result = analyzer.evaluate_tone_mapping(hdr_orig, tm_float)

        self.assertIsInstance(result['overall_score'], float)

    def test_hdr_analyzer_init(self):
        """测试HDRAnalyzer初始化"""
        # uint8 输入
        analyzer_u8 = HDRAnalyzer(self.img)
        self.assertEqual(analyzer_u8.img.dtype, np.float32)

        # float 输入
        analyzer_f = HDRAnalyzer(self.img_float)
        self.assertEqual(analyzer_f.img.dtype, np.float32)

        # 灰度输入
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        analyzer_g = HDRAnalyzer(gray)
        self.assertEqual(analyzer_g.img.ndim == 2 or analyzer_g.img.ndim == 3, True)

    def test_dynamic_range_with_real_image(self):
        """用test_color.jpg测试真实图像动态范围"""
        analyzer = HDRAnalyzer(self.img_float)
        dr = analyzer.calculate_dynamic_range()
        print(f"\n真实图像 test_color.jpg 动态范围: {dr} stops")
        self.assertIsInstance(dr, (float, np.floating))

    def test_tone_mapping_with_real_image(self):
        """用test_color.jpg测试真实Tone Mapping评估"""
        analyzer = HDRAnalyzer(self.img_float)

        # 模拟HDR: 扩展动态范围
        hdr_simulated = np.clip(self.img_float * 1.5, 0, 1)

        # TM: 还原到LDR
        tm_result = np.clip(self.img_float * 1.0, 0, 1)

        result = analyzer.evaluate_tone_mapping(hdr_simulated, tm_result)
        print(f"\n真实图像TM评估: overall_score={result['overall_score']}")
        self.assertIn('overall_score', result)


if __name__ == '__main__':
    unittest.main(verbosity=2)
