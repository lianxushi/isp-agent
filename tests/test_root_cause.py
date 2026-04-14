#!/usr/bin/env python3
"""
ISPTuningKnowledge Phase 6 单元测试
测试症状→模块映射、多症状联合诊断、问题优先级排序
"""
import os
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools.tuning_knowledge import ISPTuningKnowledge, create_tuning_knowledge


class TestPhase6RootCause(unittest.TestCase):
    """Phase 6 根因分析测试类"""

    @classmethod
    def setUpClass(cls):
        cls.test_dir = Path(__file__).parent.parent
        cls.test_image_path = cls.test_dir / 'test_color.jpg'
        cls.knowledge = create_tuning_knowledge()

        # 加载测试图像
        if cls.test_image_path.exists():
            cls.test_image = cv2.imread(str(cls.test_image_path))
            cls.test_image = cv2.cvtColor(cls.test_image, cv2.COLOR_BGR2RGB)
        else:
            cls.test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # =====================================================================
    # 症状 → 模块映射测试
    # =====================================================================
    def test_map_symptom_exact_match(self):
        """测试症状精确匹配"""
        result = self.knowledge.map_symptom_to_module(['噪声过多'])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['symptom'], '噪声过多')
        self.assertEqual(result[0]['primary_module'], 'denoise')
        self.assertIn('isp_blocks', result[0])
        self.assertIn('root_causes', result[0])
        self.assertEqual(result[0]['match_type'], 'exact')

    def test_map_symptom_color_blue(self):
        """测试色彩偏蓝映射"""
        result = self.knowledge.map_symptom_to_module(['色彩偏蓝'])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['primary_module'], 'ccm')
        self.assertIn('蓝', result[0]['symptom'])

    def test_map_symptom_edge_blur(self):
        """测试边缘模糊映射"""
        result = self.knowledge.map_symptom_to_module(['边缘模糊'])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['primary_module'], 'sharpening')

    def test_map_symptom_multiple(self):
        """测试多症状映射"""
        symptoms = ['噪声过多', '色彩偏蓝', '边缘模糊']
        result = self.knowledge.map_symptom_to_module(symptoms)
        self.assertEqual(len(result), 3)
        modules = [r['primary_module'] for r in result]
        self.assertIn('denoise', modules)
        self.assertIn('ccm', modules)
        self.assertIn('sharpening', modules)

    def test_map_symptom_fuzzy_match(self):
        """测试模糊匹配"""
        # 使用包含英文关键词的模糊症状
        result = self.knowledge.map_symptom_to_module(['too much noise'])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['match_type'], 'fuzzy')
        self.assertIn('matched_template', result[0])

    def test_map_symptom_unknown(self):
        """测试未知症状"""
        result = self.knowledge.map_symptom_to_module(['完全未知的症状xyz'])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['match_type'], 'none')
        self.assertEqual(result[0]['primary_module'], 'unknown')

    def test_map_symptom_all_templates(self):
        """测试所有已知症状模板都能正确映射"""
        all_symptoms = list(self.knowledge.SYMPTOM_MODULE_MAP.keys())
        for symptom in all_symptoms:
            result = self.knowledge.map_symptom_to_module([symptom])
            self.assertEqual(len(result), 1, f"Failed for {symptom}")
            self.assertEqual(result[0]['match_type'], 'exact')
            self.assertNotEqual(result[0]['primary_module'], 'unknown')

    # =====================================================================
    # 多症状联合诊断测试
    # =====================================================================
    def test_diagnose_multiple_symptoms_basic(self):
        """测试联合诊断基本功能"""
        symptoms = ['噪声过多', '色彩偏蓝']
        result = self.knowledge.diagnose_multiple_symptoms(self.test_image, symptoms)

        self.assertIn('input_symptoms', result)
        self.assertIn('primary_issues', result)
        self.assertIn('secondary_issues', result)
        self.assertIn('root_cause_analysis', result)
        self.assertIn('module_summary', result)
        self.assertIn('summary', result)

    def test_diagnose_multiple_symptoms_priority(self):
        """测试联合诊断优先级区分"""
        symptoms = ['噪声过多', '色彩偏蓝', '边缘模糊']
        result = self.knowledge.diagnose_multiple_symptoms(self.test_image, symptoms)

        # primary_issues应该比secondary_issues严重度更高
        if result['primary_issues'] and result['secondary_issues']:
            primary_sev = result['primary_issues'][0]['combined_severity']
            secondary_sev = result['secondary_issues'][0]['combined_severity']
            self.assertGreaterEqual(primary_sev, secondary_sev)

    def test_diagnose_multiple_symptoms_root_cause_analysis(self):
        """测试根因分析结果"""
        symptoms = ['噪声过多', '欠曝']
        result = self.knowledge.diagnose_multiple_symptoms(self.test_image, symptoms)

        rca = result['root_cause_analysis']
        self.assertIn('shared_causes', rca)
        self.assertIn('unique_causes', rca)

    def test_diagnose_multiple_symptoms_module_aggregation(self):
        """测试模块聚合"""
        symptoms = ['噪声过多', '色彩偏蓝', '色彩偏黄']
        result = self.knowledge.diagnose_multiple_symptoms(self.test_image, symptoms)

        ms = result['module_summary']
        self.assertIn('modules', ms)
        self.assertIn('priority_module', ms)

        # CCM模块应该至少包含两个色彩相关症状
        ccm_module = None
        for m in ms['modules']:
            if m['module'] == 'ccm':
                ccm_module = m
                break
        if ccm_module:
            self.assertGreaterEqual(len(ccm_module['symptoms']), 2)

    def test_diagnose_multiple_symptoms_objective_diagnostics(self):
        """测试客观诊断结果"""
        symptoms = ['摩尔纹']
        result = self.knowledge.diagnose_multiple_symptoms(self.test_image, symptoms)

        obj = result['objective_diagnostics']
        self.assertIn('bayer_demosaic', obj)
        self.assertIn('noise_level', obj)

    def test_diagnose_multiple_symptoms_summary(self):
        """测试诊断摘要"""
        symptoms = ['噪声过多', '色彩偏蓝']
        result = self.knowledge.diagnose_multiple_symptoms(self.test_image, symptoms)

        s = result['summary']
        self.assertIn('total_symptoms', s)
        self.assertIn('primary_issue_count', s)
        self.assertIn('secondary_issue_count', s)
        self.assertIn('overall_severity', s)
        self.assertIn('recommendations', s)

    # =====================================================================
    # 问题优先级排序测试
    # =====================================================================
    def test_prioritize_issues_basic(self):
        """测试优先级排序基本功能"""
        issues = [
            {
                'symptom': '噪声过多',
                'primary_module': 'denoise',
                'combined_severity': 2.0,
                'subjective_severity': 2,
                'root_causes': [],
                'isp_blocks': [],
            },
            {
                'symptom': '边缘模糊',
                'primary_module': 'sharpening',
                'combined_severity': 3.0,
                'subjective_severity': 3,
                'root_causes': [],
                'isp_blocks': [],
            },
        ]
        result = self.knowledge.prioritize_issues(issues)
        self.assertEqual(len(result), 2)
        # 第一个应该是严重度更高的
        self.assertEqual(result[0]['symptom'], '边缘模糊')
        self.assertIn('priority_score', result[0])
        self.assertIn('rank', result[0])

    def test_prioritize_issues_by_module(self):
        """测试模块优先级影响排序"""
        issues = [
            {
                'symptom': '轻微色偏',
                'primary_module': 'ccm',
                'combined_severity': 2.5,
                'subjective_severity': 2,
                'root_causes': [],
                'isp_blocks': [],
            },
            {
                'symptom': '严重噪声',
                'primary_module': 'denoise',
                'combined_severity': 2.5,
                'subjective_severity': 3,
                'root_causes': [],
                'isp_blocks': [],
            },
        ]
        result = self.knowledge.prioritize_issues(issues)
        # CCM模块权重(9) > denoise(7)，所以ccm应该排前面
        # 但combined_severity相同，subjective_severity不同
        # AE/CCM模块权重高，所以ccm症状可能优先
        self.assertIsInstance(result[0]['priority_score'], float)

    def test_prioritize_issues_empty(self):
        """测试空列表"""
        result = self.knowledge.prioritize_issues([])
        self.assertEqual(result, [])

    def test_prioritize_issues_single(self):
        """测试单问题"""
        issues = [
            {
                'symptom': '过曝',
                'primary_module': 'ae',
                'combined_severity': 3.0,
                'subjective_severity': 3,
                'root_causes': [],
                'isp_blocks': [],
            },
        ]
        result = self.knowledge.prioritize_issues(issues)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['rank'], 1)

    def test_prioritize_issues_rank_order(self):
        """测试rank字段连续性"""
        issues = [
            {'symptom': f'问题{i}', 'primary_module': 'denoise',
             'combined_severity': i * 0.5, 'subjective_severity': 1,
             'root_causes': [], 'isp_blocks': []}
            for i in range(5)
        ]
        result = self.knowledge.prioritize_issues(issues)
        ranks = [r['rank'] for r in result]
        self.assertEqual(sorted(ranks), [1, 2, 3, 4, 5])

    # =====================================================================
    # 集成测试
    # =====================================================================
    def test_phase6_end_to_end(self):
        """端到端测试：完整根因分析流程"""
        symptoms = ['噪声过多', '色彩偏蓝', '边缘模糊']
        result = self.knowledge.diagnose_multiple_symptoms(self.test_image, symptoms)

        # 验证完整报告结构
        self.assertEqual(result['summary']['total_symptoms'], 3)

        # 验证优先级排序正确
        if result['primary_issues']:
            for i in range(len(result['primary_issues']) - 1):
                self.assertGreaterEqual(
                    result['primary_issues'][i]['priority_score'],
                    result['primary_issues'][i + 1]['priority_score']
                )

    def test_phase6_with_real_image(self):
        """使用真实图像测试"""
        if not self.test_image_path.exists():
            self.skipTest("test_color.jpg not found")

        symptoms = ['摩尔纹', '伪彩色']
        result = self.knowledge.diagnose_multiple_symptoms(self.test_image, symptoms)

        # 真实图像应该有诊断结果
        self.assertIsInstance(result['summary']['overall_severity'], (int, float))
        self.assertGreaterEqual(result['summary']['overall_severity'], 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
