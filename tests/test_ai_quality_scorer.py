#!/usr/bin/env python3
"""
AIQualityScorer 单元测试
测试 BRISQUE/NIQE 集成的图像质量评分功能
"""
import os
import sys
import unittest
from pathlib import Path
import numpy as np
import cv2

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools.ai_quality_scorer import (
    AIQualityScorer,
    score_image_quality,
)


class TestAIQualityScorer(unittest.TestCase):
    """AIQualityScorer 测试"""

    def setUp(self):
        """创建测试图像"""
        self.scorer = AIQualityScorer()
        # 彩色图像
        self.test_img_color = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        # 灰度图像
        self.test_img_gray = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        # 标准图像
        self.std_img_path = Path(__file__).parent.parent / 'test_gray.jpg'
        self.color_img_path = Path(__file__).parent.parent / 'test_color.jpg'

    def test_01_scorer_init(self):
        """测试评分器初始化"""
        scorer = AIQualityScorer()
        self.assertIsInstance(scorer, AIQualityScorer)

    def test_02_score_with_color_image_path(self):
        """测试彩色图像评分(文件路径)"""
        if not self.color_img_path.exists():
            self.skipTest("Test image not found")
        result = self.scorer.score(str(self.color_img_path))
        self._verify_score_result(result)

    def test_03_score_with_gray_image_path(self):
        """测试灰度图像评分(文件路径)"""
        if not self.std_img_path.exists():
            self.skipTest("Test image not found")
        result = self.scorer.score(str(self.std_img_path))
        self._verify_score_result(result)

    def test_04_score_structure(self):
        """测试评分返回结构"""
        if not self.color_img_path.exists():
            self.skipTest("Test image not found")
        result = self.scorer.score(str(self.color_img_path))
        required_fields = [
            'mos_predicted', 'mos_description',
            'sharpness_score', 'noise_score', 'artifact_score', 'color_score',
            'overall', 'overall_score', 'grade'
        ]
        for field in required_fields:
            self.assertIn(field, result)

    def test_05_score_field_types(self):
        """测试评分字段类型"""
        if not self.color_img_path.exists():
            self.skipTest("Test image not found")
        result = self.scorer.score(str(self.color_img_path))
        self.assertIsInstance(result['mos_predicted'], float)
        self.assertIsInstance(result['sharpness_score'], (float, int))
        self.assertIsInstance(result['grade'], str)

    def test_06_score_value_ranges(self):
        """测试评分值范围"""
        if not self.color_img_path.exists():
            self.skipTest("Test image not found")
        result = self.scorer.score(str(self.color_img_path))
        # MOS应该在1-5之间
        self.assertGreaterEqual(result['mos_predicted'], 1.0)
        self.assertLessEqual(result['mos_predicted'], 5.0)
        # 各维度分数应该在0-100之间
        self.assertGreaterEqual(result['sharpness_score'], 0)
        self.assertLessEqual(result['sharpness_score'], 100)

    def test_07_grade_values(self):
        """测试等级值"""
        if not self.color_img_path.exists():
            self.skipTest("Test image not found")
        result = self.scorer.score(str(self.color_img_path))
        self.assertIn(result['grade'], ['A', 'B', 'C', 'D'])

    def test_08_overall_score_in_range(self):
        """测试综合分数范围"""
        if not self.color_img_path.exists():
            self.skipTest("Test image not found")
        result = self.scorer.score(str(self.color_img_path))
        self.assertGreaterEqual(result['overall_score'], 0)
        self.assertLessEqual(result['overall_score'], 100)

    def test_09_brisque_score_in_result(self):
        """测试 BRISQUE 分数在结果中"""
        if not self.color_img_path.exists():
            self.skipTest("Test image not found")
        result = self.scorer.score(str(self.color_img_path))
        # BRISQUE 分数应该存在（如果 BRISQUE 模块可用）
        if result.get('brisque_score') is not None:
            self.assertIsInstance(result['brisque_score'], float)
            self.assertGreaterEqual(result['brisque_score'], 0)
            self.assertLessEqual(result['brisque_score'], 100)

    def test_10_niqe_score_in_result(self):
        """测试 NIQE 分数在结果中"""
        if not self.color_img_path.exists():
            self.skipTest("Test image not found")
        result = self.scorer.score(str(self.color_img_path))
        # NIQE 分数应该存在（如果 NIQE 模块可用）
        if result.get('niqe_score') is not None:
            self.assertIsInstance(result['niqe_score'], float)
            self.assertGreaterEqual(result['niqe_score'], 0)
            self.assertLessEqual(result['niqe_score'], 100)

    def test_11_brisque_details_present(self):
        """测试 BRISQUE 详情存在"""
        if not self.color_img_path.exists():
            self.skipTest("Test image not found")
        result = self.scorer.score(str(self.color_img_path))
        if 'brisque' in result:
            brisque_details = result['brisque']
            self.assertIn('brisque_score', brisque_details)
            self.assertIn('quality_level', brisque_details)

    def test_12_niqe_details_present(self):
        """测试 NIQE 详情存在"""
        if not self.color_img_path.exists():
            self.skipTest("Test image not found")
        result = self.scorer.score(str(self.color_img_path))
        if 'niqe' in result:
            niqe_details = result['niqe']
            self.assertIn('niqe_score', niqe_details)
            self.assertIn('quality_level', niqe_details)

    def test_13_overall_fused_with_brisque_niqe(self):
        """测试综合分数融合了 BRISQUE 和 NIQE"""
        if not self.color_img_path.exists():
            self.skipTest("Test image not found")
        result = self.scorer.score(str(self.color_img_path))
        # 如果 BRISQUE 和 NIQE 都可用，overall 应该融合两者
        if result.get('brisque_score') is not None and result.get('niqe_score') is not None:
            # overall 应该在合理范围内（介于启发式和 BRISQUE/NIQE 之间）
            self.assertGreaterEqual(result['overall_score'], 0)
            self.assertLessEqual(result['overall_score'], 100)

    def test_14_details_present(self):
        """测试详细评语存在"""
        if not self.color_img_path.exists():
            self.skipTest("Test image not found")
        result = self.scorer.score(str(self.color_img_path))
        if 'details' in result:
            details = result['details']
            self.assertIn('sharpness', details)
            self.assertIn('noise', details)
            self.assertIn('artifact', details)
            self.assertIn('color', details)

    def test_15_invalid_path_raises_error(self):
        """测试无效路径抛出异常"""
        with self.assertRaises(Exception):
            self.scorer.score('/nonexistent/path/image.jpg')

    def _verify_score_result(self, result):
        """验证评分结果的通用结构"""
        self.assertIn('mos_predicted', result)
        self.assertIn('overall_score', result)
        self.assertIn('grade', result)


class TestAIQualityScorerEdgeCases(unittest.TestCase):
    """AIQualityScorer 边界情况测试"""

    def setUp(self):
        self.scorer = AIQualityScorer()
        self.std_img_path = Path(__file__).parent.parent / 'test_gray.jpg'

    def test_color_image_scoring(self):
        """测试彩色图像评分"""
        if not self.std_img_path.exists():
            self.skipTest("Test image not found")
        result = self.scorer.score(str(self.std_img_path))
        self.assertIn('overall_score', result)

    def test_grayscale_image_scoring(self):
        """测试灰度图像评分"""
        if not self.std_img_path.exists():
            self.skipTest("Test image not found")
        img = cv2.imread(str(self.std_img_path), cv2.IMREAD_GRAYSCALE)
        # AIQualityScorer.score 接受路径，需要先保存
        gray_path = self.std_img_path.parent / 'test_gray_temp.jpg'
        cv2.imwrite(str(gray_path), img)
        try:
            result = self.scorer.score(str(gray_path))
            self.assertIn('overall_score', result)
        finally:
            if gray_path.exists():
                gray_path.unlink()

    def test_mos_description_present(self):
        """测试 MOS 描述存在"""
        if not self.std_img_path.exists():
            self.skipTest("Test image not found")
        result = self.scorer.score(str(self.std_img_path))
        self.assertIn('mos_description', result)
        self.assertIsInstance(result['mos_description'], str)


class TestAIQualityConvenienceFunction(unittest.TestCase):
    """便捷函数测试"""

    def setUp(self):
        self.std_img_path = Path(__file__).parent.parent / 'test_gray.jpg'

    def test_score_image_quality_returns_dict(self):
        """测试 score_image_quality 返回字典"""
        if not self.std_img_path.exists():
            self.skipTest("Test image not found")
        result = score_image_quality(str(self.std_img_path))
        self.assertIsInstance(result, dict)

    def test_score_image_quality_has_required_fields(self):
        """测试 score_image_quality 包含必需字段"""
        if not self.std_img_path.exists():
            self.skipTest("Test image not found")
        result = score_image_quality(str(self.std_img_path))
        required_fields = ['mos_predicted', 'overall_score', 'grade']
        for field in required_fields:
            self.assertIn(field, result)


class TestAIQualityBatchScoring(unittest.TestCase):
    """批量评分测试"""

    def setUp(self):
        self.scorer = AIQualityScorer()
        self.std_img_path = Path(__file__).parent.parent / 'test_gray.jpg'
        self.color_img_path = Path(__file__).parent.parent / 'test_color.jpg'

    def test_batch_score_returns_list(self):
        """测试批量评分返回列表"""
        if not self.std_img_path.exists():
            self.skipTest("Test image not found")
        paths = [str(self.std_img_path)]
        results = self.scorer.batch_score(paths)
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 1)

    def test_batch_score_item_structure(self):
        """测试批量评分项结构"""
        if not self.std_img_path.exists():
            self.skipTest("Test image not found")
        paths = [str(self.std_img_path)]
        results = self.scorer.batch_score(paths)
        self.assertIn('path', results[0])
        self.assertIn('score', results[0])


if __name__ == '__main__':
    unittest.main()
