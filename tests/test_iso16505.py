#!/usr/bin/env python3
"""
ISO16505 测试
"""
import sys
import unittest
from pathlib import Path
from dataclasses import asdict
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools.iso16505 import (
    rgb_to_lab, delta_e_76, delta_e_94, delta_e_00,
    compute_mtf, compute_mtf50, compute_sharpness_mtf,
    compute_snr, compute_noise_iso16505,
    ISO16505Evaluator, ISO16505Result, DeltaEScale
)


class TestColorSpaceConversion(unittest.TestCase):
    """色彩空间转换测试"""

    def test_rgb_to_lab_basic(self):
        """测试RGB到Lab转换基本功能"""
        # 纯白色
        white = np.array([[[255, 255, 255]]], dtype=np.uint8)
        lab = rgb_to_lab(white)
        self.assertEqual(lab.shape, (1, 1, 3))
        # 白色L应该接近100
        self.assertGreater(lab[0, 0, 0], 90)

    def test_rgb_to_lab_black(self):
        """测试黑色转换"""
        black = np.array([[[0, 0, 0]]], dtype=np.uint8)
        lab = rgb_to_lab(black)
        # 黑色L应该接近0
        self.assertLess(lab[0, 0, 0], 10)

    def test_rgb_to_lab_red(self):
        """测试红色转换"""
        red = np.array([[[0, 0, 255]]], dtype=np.uint8)
        lab = rgb_to_lab(red)
        # 红色a分量应该为正(偏红)
        self.assertGreater(lab[0, 0, 1], 0)

    def test_rgb_to_lab_float_input(self):
        """测试float输入"""
        img = np.random.rand(100, 100, 3).astype(np.float32)
        lab = rgb_to_lab(img)
        self.assertEqual(lab.shape, img.shape)

    def test_rgb_to_lab_grayscale(self):
        """测试灰度输入"""
        gray = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        # 应该转换为3通道
        # rgb_to_lab期望3通道，这里不直接支持灰度
        # 使用3通道灰度
        gray3ch = np.stack([gray, gray, gray], axis=-1)
        lab = rgb_to_lab(gray3ch)
        self.assertEqual(lab.shape, gray3ch.shape)


class TestDeltaE(unittest.TestCase):
    """ΔE色差计算测试"""

    def test_delta_e_76_identical(self):
        """相同图像ΔE=0"""
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        de = delta_e_76(img, img)
        self.assertAlmostEqual(float(np.mean(de)), 0.0, places=5)

    def test_delta_e_94_identical(self):
        """相同图像ΔE94=0"""
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        de = delta_e_94(img, img)
        self.assertAlmostEqual(float(np.mean(de)), 0.0, places=5)

    def test_delta_e_00_identical(self):
        """相同图像ΔE00=0"""
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        de = delta_e_00(img, img)
        self.assertAlmostEqual(float(np.mean(de)), 0.0, places=5)

    def test_delta_e_different_images(self):
        """不同图像ΔE>0"""
        img1 = np.full((100, 100, 3), 200, dtype=np.uint8)
        img2 = np.full((100, 100, 3), 50, dtype=np.uint8)
        for func, name in [(delta_e_76, 'DE76'), (delta_e_94, 'DE94'), (delta_e_00, 'DE00')]:
            de = func(img1, img2)
            self.assertGreater(float(np.mean(de)), 5.0, f"{name} should be > 5 for very different images")
            print(f"  {name}: mean={float(np.mean(de)):.2f}, max={float(np.max(de)):.2f}")

    def test_delta_e_uint8_input(self):
        """测试uint8输入"""
        img1 = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        de = delta_e_76(img1, img2)
        # 应该产生有效结果
        self.assertGreater(float(np.mean(de)), 0)
        self.assertLess(float(np.mean(de)), 200)


class TestSharpness(unittest.TestCase):
    """锐度/MTF测试"""

    @classmethod
    def setUpClass(cls):
        cls.test_dir = Path(__file__).parent.parent
        cls.test_image_path = cls.test_dir / 'test_color.jpg'

    def test_compute_mtf_basic(self):
        """测试MTF计算基本功能"""
        # 创建棋盘格图像(高频)
        img = np.zeros((128, 128), dtype=np.uint8)
        img[::2, ::2] = 255
        img[1::2, 1::2] = 255

        mtf = compute_mtf(img, 'horizontal')
        self.assertGreater(len(mtf), 0)
        self.assertTrue(0 <= mtf[0] <= 1.1)  # 归一化后DC应该≈1

    def test_compute_mtf50(self):
        """测试MTF50计算"""
        mtf = np.array([1.0, 0.9, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1])
        freq, val = compute_mtf50(mtf)
        self.assertGreater(freq, 0)
        self.assertAlmostEqual(val, 0.5, places=1)

    def test_compute_sharpness_mtf_basic(self):
        """测试锐度MTF评估"""
        # 创建模糊图像
        blurred = np.zeros((100, 100, 3), dtype=np.uint8)
        blurred[:, :, :] = 128

        result = compute_sharpness_mtf(blurred)
        self.assertIn('mtf_50', result)
        self.assertIn('mtf_nyquist', result)
        self.assertIn('acuity_score', result)
        self.assertIn('pass_iso16505', result)
        print(f"  模糊图像: mtf_nyquist={result['mtf_nyquist']}, score={result['acuity_score']}")

    def test_compute_sharpness_mtf_real_image(self):
        """使用真实图像测试"""
        if not self.test_image_path.exists():
            self.skipTest("测试图像不存在")

        img = cv2.imread(str(self.test_image_path))
        result = compute_sharpness_mtf(img)
        self.assertGreater(result['acuity_score'], 0)
        print(f"  test_color.jpg: mtf_nyquist={result['mtf_nyquist']}, score={result['acuity_score']}")

    def test_sharpness_blur_vs_sharp(self):
        """测试清晰vs模糊图像对比"""
        # 清晰图像(高对比度边缘)
        sharp = np.zeros((100, 100, 3), dtype=np.uint8)
        sharp[:50, :, :] = 255
        sharp[50:, :, :] = 0

        # 模糊版本
        blur_kernel = np.ones((5, 5), dtype=np.float32) / 25
        blurred = cv2.filter2D(sharp.astype(np.float32), -1, blur_kernel).astype(np.uint8)

        sharp_result = compute_sharpness_mtf(sharp)
        blur_result = compute_sharpness_mtf(blurred)

        # 清晰图像的MTF应该高于模糊版本
        self.assertGreater(
            sharp_result['mtf_nyquist'],
            blur_result['mtf_nyquist'],
            "Sharp image should have higher MTF"
        )
        print(f"  清晰: {sharp_result['mtf_nyquist']:.3f} vs 模糊: {blur_result['mtf_nyquist']:.3f}")


class TestNoise(unittest.TestCase):
    """噪声评估测试"""

    def test_compute_snr_basic(self):
        """测试SNR计算基本功能"""
        # 创建平滑图像
        smooth = np.full((100, 100), 128, dtype=np.uint8)
        # 添加少量噪声
        noisy = np.clip(smooth.astype(np.int16) + np.random.randint(-5, 5, (100, 100)), 0, 255).astype(np.uint8)

        snr = compute_snr(noisy)
        self.assertGreater(snr, 0)
        print(f"  平滑+噪声图像 SNR: {snr:.1f}dB")

    def test_compute_snr_no_noise(self):
        """测试无噪声图像"""
        img = np.full((100, 100), 128, dtype=np.uint8)
        snr = compute_snr(img)
        self.assertGreater(snr, 50)  # 无噪声应该SNR很高

    def test_compute_noise_iso16505_basic(self):
        """测试ISO16505噪声评估"""
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        result = compute_noise_iso16505(img)

        self.assertIn('snr_db', result)
        self.assertIn('noise_score', result)
        self.assertIn('pass_iso16505', result)
        self.assertIn('iso16505_level', result)
        print(f"  随机噪声图像: SNR={result['snr_db']:.1f}dB, score={result['noise_score']:.1f}")

    def test_noise_iso16505_levels(self):
        """测试不同噪声水平的评级"""
        # 极低噪声
        clean = np.full((100, 100, 3), 128, dtype=np.uint8)
        result_clean = compute_noise_iso16505(clean)
        self.assertEqual(result_clean['iso16505_level'], 'excellent')

        # 高噪声
        noisy = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        result_noisy = compute_noise_iso16505(noisy)
        self.assertIn(result_noisy['iso16505_level'], ['acceptable', 'poor', 'good'])


class TestISO16505Evaluator(unittest.TestCase):
    """ISO16505评估器测试"""

    @classmethod
    def setUpClass(cls):
        cls.test_dir = Path(__file__).parent.parent
        cls.test_image_path = cls.test_dir / 'test_color.jpg'

    def test_evaluator_no_reference(self):
        """无参考图像评估"""
        evaluator = ISO16505Evaluator()

        if self.test_image_path.exists():
            img = cv2.imread(str(self.test_image_path))
        else:
            img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

        result = evaluator.evaluate(img)
        self.assertIsInstance(result, ISO16505Result)
        self.assertIn('overall_score', asdict(result))
        print(f"  整体评分: {result.overall_score}, 合规: {result.compliant}")

    def test_evaluator_with_reference(self):
        """有参考图像评估"""
        if not self.test_image_path.exists():
            self.skipTest("测试图像不存在")

        img1 = cv2.imread(str(self.test_image_path))
        # 创建轻微失真版本
        img2 = cv2.GaussianBlur(img1, (3, 3), 0)

        evaluator = ISO16505Evaluator(img1)
        result = evaluator.evaluate(img2)

        self.assertGreater(result.overall_score, 0)
        self.assertIn(result.level, ['excellent', 'good', 'acceptable', 'poor', 'non_compliant'])
        print(f"  DE评估: ΔE={result.color_accuracy.get('mean_de', 'N/A')}")

    def test_evaluator_all_fields(self):
        """测试所有评估字段"""
        evaluator = ISO16505Evaluator()

        img = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        result = evaluator.evaluate(img)

        self.assertIn('color_accuracy', asdict(result))
        self.assertIn('sharpness', asdict(result))
        self.assertIn('noise', asdict(result))
        self.assertIn('overall_score', asdict(result))
        self.assertIn('compliant', asdict(result))
        self.assertIn('level', asdict(result))

    def test_to_dict(self):
        """测试结果转字典"""
        evaluator = ISO16505Evaluator()
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        result = evaluator.evaluate(img)
        d = evaluator.to_dict(result)

        self.assertIsInstance(d, dict)
        self.assertIn('overall_score', d)
        self.assertIn('color_accuracy', d)

    def test_grade_boundaries(self):
        """测试评分边界"""
        evaluator = ISO16505Evaluator()

        # 干净图像
        clean = np.full((100, 100, 3), 128, dtype=np.uint8)
        r_clean = evaluator.evaluate(clean)
        self.assertGreaterEqual(r_clean.overall_score, 0)
        self.assertLessEqual(r_clean.overall_score, 100)

        # 随机(高噪声)图像
        noisy = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        r_noisy = evaluator.evaluate(noisy)
        self.assertGreaterEqual(r_noisy.overall_score, 0)
        self.assertLessEqual(r_noisy.overall_score, 100)


class TestIntegration(unittest.TestCase):
    """集成测试"""

    def test_full_pipeline(self):
        """完整ISO16505评估流程"""
        if not Path(__file__).parent.parent.joinpath('test_color.jpg').exists():
            self.skipTest("测试图像不存在")

        from src.tools.iso16505 import assess_iso16505

        img_path = str(Path(__file__).parent.parent / 'test_color.jpg')
        result = assess_iso16505(img_path)

        self.assertIn('overall_score', result)
        self.assertIn('compliant', result)
        print(f"  test_color.jpg 评估完成: score={result['overall_score']}, compliant={result['compliant']}")


if __name__ == '__main__':
    from dataclasses import asdict
    unittest.main(verbosity=2)
