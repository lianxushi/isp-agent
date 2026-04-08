#!/usr/bin/env python3
"""
HTML Report Generator 测试
"""
import sys
import unittest
import tempfile
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.isp_compare.reports.html_generator import (
    HTMLReportGenerator, ComparisonData, generate_html_report
)


class TestHTMLReportGenerator(unittest.TestCase):
    """HTML报告生成器测试"""

    @classmethod
    def setUpClass(cls):
        cls.gen = HTMLReportGenerator()

    def test_generator_init(self):
        """初始化"""
        gen = HTMLReportGenerator()
        self.assertIsNotNone(gen)

    def test_generate_basic_report(self):
        """生成基本报告"""
        data = ComparisonData(
            report_id='TEST-001',
            timestamp='2026-04-08 12:00:00',
            version_a='v1.0',
            version_b='v1.1',
            overall_status='similar',
            processing_time_ms=500.0,
            summary='测试摘要',
            recommendations=['建议1', '建议2']
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'report.html')
            result = self.gen.generate(data, output_path)

            self.assertTrue(os.path.exists(output_path))
            with open(output_path, 'r', encoding='utf-8') as f:
                content = f.read()
            self.assertIn('ISP Version Comparison Report', content)
            self.assertIn('TEST-001', content)
            self.assertIn('v1.0', content)
            self.assertIn('v1.1', content)

    def test_generate_with_metrics(self):
        """带指标的报告"""
        data = ComparisonData(
            report_id='TEST-002',
            timestamp='2026-04-08 12:00:00',
            version_a='v1.0',
            version_b='v1.1',
            overall_status='a_improved',
            metrics={
                'brisque': {'name': 'BRISQUE', 'a_value': 75.0, 'b_value': 68.0, 'delta': -7.0, 'better': 'A'},
                'snr': {'name': 'SNR', 'a_value': 40.0, 'b_value': 38.0, 'delta': -2.0, 'better': 'A'},
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'report.html')
            self.gen.generate(data, output_path)

            with open(output_path, 'r', encoding='utf-8') as f:
                content = f.read()
            self.assertIn('BRISQUE', content)
            self.assertIn('SNR', content)

    def test_generate_with_hdr(self):
        """带HDR分析的报告"""
        data = ComparisonData(
            report_id='TEST-003',
            timestamp='2026-04-08 12:00:00',
            version_a='v1.0',
            version_b='v1.1',
            hdr_analysis={
                'dynamic_range': {
                    'stops': 8.5,
                    'min': 0.01,
                    'max': 10.0,
                },
                'exposure_analysis': {
                    'over_exposed_percent': 5.0,
                    'under_exposed_percent': 8.0,
                    'exposure_assessment': '良好 - 曝光均衡',
                },
                'local_contrast': {
                    '5x5': 25.0,
                    '15x15': 30.0,
                }
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'report.html')
            self.gen.generate(data, output_path)

            with open(output_path, 'r', encoding='utf-8') as f:
                content = f.read()
            self.assertIn('HDR', content)
            self.assertIn('8.5', content)

    def test_generate_with_iso16505(self):
        """带ISO16505的报告"""
        data = ComparisonData(
            report_id='TEST-004',
            timestamp='2026-04-08 12:00:00',
            version_a='v1.0',
            version_b='v1.1',
            iso16505={
                'overall_score': 78.5,
                'compliant': True,
                'level': 'good',
                'color_accuracy': {'mean_de': 8.5, 'score': 85.0, 'level': 'good', 'pass_iso16505': True},
                'sharpness': {'acuity_score': 75.0, 'mtf_nyquist': 0.35, 'pass_iso16505': True},
                'noise': {'noise_score': 78.0, 'snr_db': 42.0, 'pass_iso16505': True},
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'report.html')
            self.gen.generate(data, output_path)

            with open(output_path, 'r', encoding='utf-8') as f:
                content = f.read()
            self.assertIn('ISO 16505', content)
            self.assertIn('78.5', content)
            self.assertIn('合规', content)

    def test_generate_with_images(self):
        """带图像的报告"""
        test_dir = Path(__file__).parent.parent
        img_path = test_dir / 'test_color.jpg'

        if img_path.exists():
            data = ComparisonData(
                report_id='TEST-005',
                timestamp='2026-04-08 12:00:00',
                version_a='v1.0',
                version_b='v1.1',
            )
            images = {
                'Golden': str(img_path),
                'Version A': str(img_path),
                'Version B': str(img_path),
            }

            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = os.path.join(tmpdir, 'report.html')
                self.gen.generate(data, output_path, images=images)

                with open(output_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                self.assertIn('视觉对比', content)
                self.assertIn('test_color.jpg', content)

    def test_generate_from_dict(self):
        """从字典生成报告"""
        data_dict = {
            'report_id': 'TEST-006',
            'timestamp': '2026-04-08 12:00:00',
            'version_a': 'v1.0',
            'version_b': 'v1.1',
            'overall_status': 'similar',
            'summary': '测试报告',
            'recommendations': ['建议1'],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'report.html')
            result = self.gen.generate_from_dict(data_dict, output_path)

            self.assertTrue(os.path.exists(output_path))

    def test_generate_empty_recommendations(self):
        """空建议列表"""
        data = ComparisonData(
            report_id='TEST-007',
            timestamp='2026-04-08 12:00:00',
            version_a='v1.0',
            version_b='v1.1',
            recommendations=[]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'report.html')
            self.gen.generate(data, output_path)
            self.assertTrue(os.path.exists(output_path))

    def test_generate_no_overall_score(self):
        """无综合评分时报告生成"""
        data = ComparisonData(
            report_id='TEST-008',
            timestamp='2026-04-08 12:00:00',
            version_a='v1.0',
            version_b='v1.1',
            overall_status='similar',
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'report.html')
            self.gen.generate(data, output_path)
            self.assertTrue(os.path.exists(output_path))


if __name__ == '__main__':
    gen = HTMLReportGenerator()  # module-level instance
    unittest.main(verbosity=2)
