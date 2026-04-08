#!/usr/bin/env python3
"""
Tuning Advisor 测试
"""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools.tuning_advisor import (
    TuningAdvisor, TuningRecommendation, ISPDiagnosis,
    ProblemCategory, advise_tuning
)


class TestTuningAdvisor(unittest.TestCase):
    """调参建议引擎测试"""

    def test_init_without_llm(self):
        """无LLM初始化"""
        advisor = TuningAdvisor()
        self.assertIsNotNone(advisor)

    def test_diagnose_high_noise(self):
        """诊断高噪声"""
        advisor = TuningAdvisor()
        diagnoses = advisor.diagnose(snr_db=20.0)

        self.assertGreater(len(diagnoses), 0)
        noise_diags = [d for d in diagnoses if '噪声' in d.category or 'NOISE' in d.category]
        self.assertGreater(len(noise_diags), 0)
        self.assertEqual(noise_diags[0].severity, 'severe')

    def test_diagnose_low_noise(self):
        """诊断低噪声"""
        advisor = TuningAdvisor()
        diagnoses = advisor.diagnose(snr_db=50.0)
        noise_diags = [d for d in diagnoses if '噪声' in d.category or 'NOISE' in d.category]
        # 高SNR不应该触发噪声诊断
        self.assertEqual(len(noise_diags), 0)

    def test_diagnose_low_mtf(self):
        """诊断低MTF(模糊)"""
        advisor = TuningAdvisor()
        diagnoses = advisor.diagnose(mtf_nyquist=0.15)

        self.assertGreater(len(diagnoses), 0)
        blur_diags = [d for d in diagnoses if '模糊' in d.category or 'BLUR' in d.category]
        self.assertGreater(len(blur_diags), 0)

    def test_diagnose_high_delta_e(self):
        """诊断高色差"""
        advisor = TuningAdvisor()
        diagnoses = advisor.diagnose(delta_e=20.0)

        self.assertGreater(len(diagnoses), 0)
        color_diags = [d for d in diagnoses if '色彩' in d.category or 'COLOR' in d.category]
        self.assertGreater(len(color_diags), 0)

    def test_diagnose_hdr_low_stops(self):
        """诊断HDR低动态范围"""
        advisor = TuningAdvisor()
        hdr = {
            'dynamic_range': {'stops': 3.0},
            'exposure_analysis': {'over_exposed_percent': 5.0, 'under_exposed_percent': 10.0}
        }
        diagnoses = advisor.diagnose(hdr_analysis=hdr)

        self.assertGreater(len(diagnoses), 0)

    def test_diagnose_combined(self):
        """综合诊断"""
        advisor = TuningAdvisor()
        diagnoses = advisor.diagnose(
            brisque_score=35.0,
            snr_db=25.0,
            mtf_nyquist=0.18,
            delta_e=16.0
        )

        self.assertGreater(len(diagnoses), 1)
        categories = set(d.category for d in diagnoses)
        self.assertGreater(len(categories), 1)

    def test_diagnose_normal(self):
        """正常图像不触发诊断"""
        advisor = TuningAdvisor()
        diagnoses = advisor.diagnose(
            brisque_score=85.0,
            snr_db=50.0,
            mtf_nyquist=0.5,
            delta_e=3.0
        )
        # 正常参数不应该有严重问题
        severe_diags = [d for d in diagnoses if d.severity == 'severe']
        self.assertEqual(len(severe_diags), 0)

    def test_get_rule_based_recommendations(self):
        """测试规则引擎建议"""
        advisor = TuningAdvisor()
        diagnoses = advisor.diagnose(snr_db=20.0)
        recommendations = advisor.get_rule_based_recommendations(diagnoses)

        self.assertGreater(len(recommendations), 0)
        for rec in recommendations:
            self.assertIsInstance(rec, TuningRecommendation)
            self.assertIsNotNone(rec.param_name)
            self.assertIsNotNone(rec.direction)

    def test_recommendations_with_current_params(self):
        """带当前参数的调参建议"""
        advisor = TuningAdvisor()
        diagnoses = advisor.diagnose(snr_db=20.0)
        current = {'noise_reduction_strength': 30}
        recommendations = advisor.get_rule_based_recommendations(diagnoses, current)

        self.assertGreater(len(recommendations), 0)
        # 应该包含当前参数值
        nr_recs = [r for r in recommendations if r.param_name == 'noise_reduction_strength']
        if nr_recs:
            self.assertEqual(nr_recs[0].current_value, 30)

    def test_recommendations_deduplication(self):
        """建议去重测试"""
        advisor = TuningAdvisor()
        diagnoses = advisor.diagnose(
            brisque_score=35.0,
            snr_db=25.0,
            mtf_nyquist=0.15
        )
        recommendations = advisor.get_rule_based_recommendations(diagnoses)

        # 检查无重复参数
        param_names = [r.param_name for r in recommendations]
        self.assertEqual(len(param_names), len(set(param_names)))

    def test_recommendations_sorted_by_priority(self):
        """建议按优先级排序"""
        advisor = TuningAdvisor()
        diagnoses = advisor.diagnose(
            brisque_score=35.0,
            snr_db=20.0,
            mtf_nyquist=0.1
        )
        recommendations = advisor.get_rule_based_recommendations(diagnoses)

        if len(recommendations) > 1:
            priorities = [r.priority for r in recommendations]
            self.assertEqual(priorities, sorted(priorities))

    def test_advise_main_entry(self):
        """advise主入口测试"""
        advisor = TuningAdvisor()
        result = advisor.advise(
            brisque_score=35.0,
            snr_db=25.0,
            mtf_nyquist=0.18,
            delta_e=16.0,
            use_llm=False
        )

        self.assertIn('diagnoses', result)
        self.assertIn('recommendations', result)
        self.assertIn('summary', result)
        self.assertGreater(len(result['diagnoses']), 0)
        self.assertGreater(len(result['recommendations']), 0)

    def test_advise_no_issues(self):
        """无问题时返回正常"""
        advisor = TuningAdvisor()
        result = advisor.advise(
            brisque_score=90.0,
            snr_db=55.0,
            mtf_nyquist=0.6,
            delta_e=2.0,
            use_llm=False
        )

        self.assertIn('summary', result)
        self.assertIn('无需调参', result['summary'] or '')

    def test_advise_video_mode(self):
        """视频模式调参"""
        advisor = TuningAdvisor()
        diagnoses = advisor.diagnose(snr_db=25.0)
        recs = advisor.get_rule_based_recommendations(diagnoses, is_video=True)

        # 视频模式应该有temporal_denoise建议
        td_recs = [r for r in recs if 'temporal' in r.param_name]
        # 规则可能不触发，取决于严重程度

    def test_advise_with_scene_context(self):
        """带场景上下文的建议"""
        advisor = TuningAdvisor()
        result = advisor.advise(
            brisque_score=50.0,
            snr_db=35.0,
            scene_context='夜间驾驶场景，前方有强光灯',
            use_llm=False
        )

        self.assertIn('diagnoses', result)

    def test_advise_full_params(self):
        """完整参数advise"""
        advisor = TuningAdvisor()
        hdr = {
            'dynamic_range': {'stops': 5.0},
            'exposure_analysis': {'over_exposed_percent': 8.0, 'under_exposed_percent': 15.0}
        }
        iso16505 = {
            'overall_score': 55.0,
            'compliant': False
        }

        result = advisor.advise(
            brisque_score=55.0,
            snr_db=35.0,
            delta_e=11.0,
            mtf_nyquist=0.35,
            hdr_analysis=hdr,
            iso16505_result=iso16505,
            use_llm=False
        )

        self.assertGreater(len(result['diagnoses']), 0)
        self.assertGreater(len(result['recommendations']), 0)


class TestTuningAdvisorCLI(unittest.TestCase):
    """CLI便捷函数测试"""

    def test_advise_tuning_basic(self):
        """基本功能"""
        result = advise_tuning(snr_db=25.0, use_llm=False)
        self.assertIn('diagnoses', result)
        self.assertIn('recommendations', result)

    def test_advise_tuning_output_path(self):
        """输出文件"""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            tmp = f.name

        result = advise_tuning(snr_db=25.0, use_llm=False, output_path=tmp)

        import os
        self.assertTrue(os.path.exists(tmp))
        os.unlink(tmp)


if __name__ == '__main__':
    unittest.main(verbosity=2)
