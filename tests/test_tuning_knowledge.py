#!/usr/bin/env python3
"""
ISPTuningKnowledge Phase 4 单元测试
测试 Bayer/Demosaic诊断、降噪建议、锐化伪影、色彩空间诊断
"""
import os
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools.tuning_knowledge import ISPTuningKnowledge, create_tuning_knowledge


class TestTuningKnowledgePhase4(unittest.TestCase):
    """Phase 4 调优知识测试类"""

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

        # 创建合成测试图像
        cls.synth_flat = np.ones((256, 256, 3), dtype=np.float32) * 0.5
        cls.synth_edge = np.zeros((256, 256, 3), dtype=np.float32)
        cls.synth_edge[128:, :] = 1.0
        cls.synth_saturated = np.zeros((256, 256, 3), dtype=np.float32)
        cls.synth_saturated[:, :, 0] = 1.0   # R=1 (超sRGB边界)
        cls.synth_saturated[:, :, 1] = 0.0   # G=0
        cls.synth_saturated[:, :, 2] = 0.0   # B=0

    def test_instance_creation(self):
        """测试知识库实例创建"""
        k = create_tuning_knowledge()
        self.assertIsInstance(k, ISPTuningKnowledge)

    # =====================================================================
    # Bayer / Demosaic 诊断测试
    # =====================================================================
    def test_diagnose_bayer_issues_basic(self):
        """测试 Bayer/Demosaic 诊断基本功能"""
        result = self.knowledge.diagnose_bayer_issues(self.test_image)
        
        self.assertIn('moire', result)
        self.assertIn('false_color', result)
        self.assertIn('zipper_artifacts', result)
        self.assertIn('demosaic_accuracy', result)
        self.assertIn('summary', result)
        
        # 每个子项都应该有severity和score
        for key in ['moire', 'false_color', 'zipper_artifacts', 'demosaic_accuracy']:
            self.assertIn('severity', result[key])
            self.assertIn('score', result[key])
            self.assertIn('detail', result[key])
            self.assertIsInstance(result[key]['severity'], int)
            self.assertGreaterEqual(result[key]['severity'], 0)
            self.assertLessEqual(result[key]['severity'], 3)

    def test_diagnose_bayer_issues_synth(self):
        """测试合成图像的Bayer诊断"""
        result = self.knowledge.diagnose_bayer_issues(self.synth_flat)
        self.assertIsInstance(result, dict)
        self.assertIn('summary', result)
        
        result_edge = self.knowledge.diagnose_bayer_issues(self.synth_edge)
        self.assertIsInstance(result_edge['zipper_artifacts']['score'], float)

    def test_moire_detection(self):
        """测试摩尔纹检测"""
        result = self.knowledge.diagnose_bayer_issues(self.test_image)
        moire = result['moire']
        self.assertIn('hf_ratio', moire)
        self.assertIn('periodicity', moire)
        self.assertGreaterEqual(moire['severity'], 0)
        self.assertLessEqual(moire['severity'], 3)

    def test_false_color_detection(self):
        """测试伪彩色检测"""
        result = self.knowledge.diagnose_bayer_issues(self.test_image)
        fc = result['false_color']
        self.assertIn('cb_std', fc)
        self.assertIn('cr_std', fc)
        self.assertGreaterEqual(fc['severity'], 0)
        self.assertLessEqual(fc['severity'], 3)

    def test_zipper_artifacts_detection(self):
        """测试拉链效应检测"""
        result = self.knowledge.diagnose_bayer_issues(self.test_image)
        zipper = result['zipper_artifacts']
        self.assertIn('h_flip_rate', zipper)
        self.assertIn('v_flip_rate', zipper)
        self.assertGreaterEqual(zipper['severity'], 0)
        self.assertLessEqual(zipper['severity'], 3)

    def test_demosaic_accuracy(self):
        """测试demosaic精度评估"""
        result = self.knowledge.diagnose_bayer_issues(self.test_image)
        acc = result['demosaic_accuracy']
        self.assertIn('channel_inconsistency', acc)
        self.assertIn('chroma_smoothness', acc)
        self.assertGreaterEqual(acc['severity'], 0)
        self.assertLessEqual(acc['severity'], 3)

    # =====================================================================
    # 降噪参数建议测试
    # =====================================================================
    def test_suggest_denoise_params_basic(self):
        """测试降噪参数建议基本功能"""
        result = self.knowledge.suggest_denoise_params(self.test_image)
        
        self.assertIn('noise_level', result)
        self.assertIn('scene_complexity', result)
        self.assertIn('motion_score', result)
        self.assertIn('recommendations', result)
        self.assertIn('strategy', result)

    def test_noise_level_estimation(self):
        """测试噪声水平评估"""
        result = self.knowledge.suggest_denoise_params(self.test_image)
        noise = result['noise_level']
        
        self.assertIn('sigma', noise)
        self.assertIn('noise_db', noise)
        self.assertIn('severity', noise)
        self.assertIn('label', noise)
        self.assertIsInstance(noise['sigma'], float)
        self.assertGreaterEqual(noise['severity'], 0)
        self.assertLessEqual(noise['severity'], 3)

    def test_denoise_recommendations(self):
        """测试降噪参数推荐"""
        result = self.knowledge.suggest_denoise_params(self.test_image)
        rec = result['recommendations']
        
        self.assertIn('spatial_strength', rec)
        self.assertIn('temporal_strength', rec)
        self.assertIn('use_temporal_nr', rec)
        self.assertIn('spatial_nr_radius', rec)
        self.assertIn('luminance_strength', rec)
        self.assertIn('chroma_strength', rec)
        
        self.assertIsInstance(rec['use_temporal_nr'], bool)
        self.assertGreaterEqual(rec['spatial_strength'], 0)
        self.assertLessEqual(rec['spatial_strength'], 4)

    def test_denoise_flat_vs_noisy(self):
        """测试平坦图像vs噪声图像的降噪建议差异"""
        # 创建极低噪声的平坦图像
        flat_clean = np.ones((256, 256, 3), dtype=np.float32) * 0.5
        result_clean = self.knowledge.suggest_denoise_params(flat_clean)
        
        # 创建高噪声图像
        noise = np.random.normal(0, 0.1, (256, 256, 3)).astype(np.float32)
        flat_noisy = np.clip(flat_clean + noise, 0, 1)
        result_noisy = self.knowledge.suggest_denoise_params(flat_noisy)
        
        # 噪声图像的sigma应该更大
        sigma_clean = result_clean['noise_level']['sigma']
        sigma_noisy = result_noisy['noise_level']['sigma']
        self.assertGreater(sigma_noisy, sigma_clean)

    # =====================================================================
    # 锐化伪影诊断测试
    # =====================================================================
    def test_diagnose_sharpening_artifacts_basic(self):
        """测试锐化伪影诊断基本功能"""
        result = self.knowledge.diagnose_sharpening_artifacts(self.test_image)
        
        self.assertIn('ringing', result)
        self.assertIn('over_sharpening', result)
        self.assertIn('edge_artifacts', result)
        self.assertIn('summary', result)
        
        for key in ['ringing', 'over_sharpening', 'edge_artifacts']:
            self.assertIn('severity', result[key])
            self.assertIn('score', result[key])
            self.assertIn('detail', result[key])

    def test_ringing_detection(self):
        """测试振铃效应检测"""
        result = self.knowledge.diagnose_sharpening_artifacts(self.test_image)
        ringing = result['ringing']
        self.assertIn('lap_mean', ringing)
        self.assertIn('lap_signedness', ringing)
        self.assertGreaterEqual(ringing['severity'], 0)
        self.assertLessEqual(ringing['severity'], 3)

    def test_over_sharpening_detection(self):
        """测试过锐化检测"""
        result = self.knowledge.diagnose_sharpening_artifacts(self.test_image)
        os = result['over_sharpening']
        self.assertIn('avg_overshoot', os)
        self.assertIn('avg_undershoot', os)
        self.assertIn('hf_ratio', os)
        self.assertGreaterEqual(os['severity'], 0)
        self.assertLessEqual(os['severity'], 3)

    def test_edge_artifacts_detection(self):
        """测试边缘伪影检测"""
        result = self.knowledge.diagnose_sharpening_artifacts(self.test_image)
        ea = result['edge_artifacts']
        self.assertIn('edge_density', ea)
        self.assertIn('angle_entropy_ratio', ea)
        self.assertGreaterEqual(ea['severity'], 0)
        self.assertLessEqual(ea['severity'], 3)

    # =====================================================================
    # 色彩空间诊断测试
    # =====================================================================
    def test_diagnose_colorspace_issues_basic(self):
        """测试色彩空间诊断基本功能"""
        result = self.knowledge.diagnose_colorspace_issues(self.test_image)
        
        self.assertIn('colorspace_estimate', result)
        self.assertIn('gamut_overflow', result)
        self.assertIn('gamma_issues', result)
        self.assertIn('summary', result)

    def test_gamut_overflow_detection(self):
        """测试色域溢出检测"""
        result = self.knowledge.diagnose_colorspace_issues(self.synth_saturated)
        overflow = result['gamut_overflow']
        
        self.assertIn('severity', overflow)
        self.assertIn('score', overflow)
        self.assertIn('type', overflow)
        self.assertGreaterEqual(overflow['severity'], 0)
        self.assertLessEqual(overflow['severity'], 3)

    def test_gamut_normal_image(self):
        """测试正常图像的色域检测"""
        result = self.knowledge.diagnose_colorspace_issues(self.synth_flat)
        overflow = result['gamut_overflow']
        # 正常图像不应该有严重溢出
        self.assertLessEqual(overflow['severity'], 3)

    def test_gamma_issues_detection(self):
        """测试Gamma转换问题检测"""
        result = self.knowledge.diagnose_colorspace_issues(self.test_image)
        gamma = result['gamma_issues']
        
        self.assertIn('estimated_gamma', gamma)
        self.assertIn('gamma_error', gamma)
        self.assertIn('black_clip_ratio', gamma)
        self.assertIn('white_clip_ratio', gamma)
        self.assertGreaterEqual(gamma['severity'], 0)
        self.assertLessEqual(gamma['severity'], 3)

    def test_colorspace_estimate_with_metadata(self):
        """测试带元数据的色彩空间估算"""
        # 提供元数据时应直接返回
        result = self.knowledge.diagnose_colorspace_issues(
            self.test_image,
            metadata={'color_space': 'Adobe RGB'}
        )
        cs = result['colorspace_estimate']
        self.assertEqual(cs['detected'], 'adobe rgb')
        self.assertEqual(cs['confidence'], 1.0)

    def test_colorspace_estimate_srgb(self):
        """测试sRGB色彩空间估算"""
        # 正常图像应该是sRGB
        result = self.knowledge.diagnose_colorspace_issues(self.synth_flat)
        cs = result['colorspace_estimate']
        self.assertIn('detected', cs)
        self.assertIn('confidence', cs)

    # =====================================================================
    # 集成测试
    # =====================================================================
    def test_all_functions_produce_dict(self):
        """确保所有Phase 4函数返回有效dict"""
        funcs = [
            lambda: self.knowledge.diagnose_bayer_issues(self.test_image),
            lambda: self.knowledge.suggest_denoise_params(self.test_image),
            lambda: self.knowledge.diagnose_sharpening_artifacts(self.test_image),
            lambda: self.knowledge.diagnose_colorspace_issues(self.test_image),
        ]
        for f in funcs:
            result = f()
            self.assertIsInstance(result, dict)

    def test_severity_aggregation(self):
        """测试severity聚合"""
        bayer = self.knowledge.diagnose_bayer_issues(self.test_image)
        sharp = self.knowledge.diagnose_sharpening_artifacts(self.test_image)
        cs = self.knowledge.diagnose_colorspace_issues(self.test_image)
        
        for r in [bayer, sharp, cs]:
            summary = r['summary']
            self.assertIn('issues_found', summary)
            self.assertIn('overall_severity', summary)
            self.assertGreaterEqual(summary['overall_severity'], 0)
            self.assertLessEqual(summary['overall_severity'], 3)


class TestTuningKnowledgePhase5(unittest.TestCase):
    """Phase 5 场景识别与参数推荐测试类"""

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

        # 创建合成测试图像
        # 晴天场景：高亮度、高对比度
        cls.synth_sunny = np.ones((256, 256, 3), dtype=np.float32) * 0.75
        cls.synth_sunny[:, :, 0] = 0.9  # 高R
        cls.synth_sunny[:, :, 1] = 0.85  # 高G
        cls.synth_sunny[:, :, 2] = 0.7  # 适中B
        # 添加对比度变化
        cls.synth_sunny[128:, :, :] = np.clip(cls.synth_sunny[128:, :, :] * 1.2, 0, 1)
        
        # 阴天场景：中等亮度、低对比度
        cls.synth_cloudy = np.ones((256, 256, 3), dtype=np.float32) * 0.55
        cls.synth_cloudy += np.random.randn(*cls.synth_cloudy.shape).astype(np.float32) * 0.03
        
        # 夜景场景：低亮度
        cls.synth_night = np.ones((256, 256, 3), dtype=np.float32) * 0.12
        # 添加一些噪声模拟夜景
        cls.synth_night += np.random.randn(256, 256, 3).astype(np.float32) * 0.03
        
        # 室内场景：亮度变化大但整体不太暗，有明显的人工光源特征
        cls.synth_indoor = np.ones((256, 256, 3), dtype=np.float32) * 0.42
        cls.synth_indoor[40:120, 80:180, :] = 0.8   # 窗户区域（亮）- 局部高亮
        cls.synth_indoor[160:, :, :] = 0.28  # 暗区 - 形成强烈对比
        
        # 低光环境
        cls.synth_low_light = np.ones((256, 256, 3), dtype=np.float32) * 0.12
        cls.synth_low_light += np.random.randn(256, 256, 3).astype(np.float32) * 0.02
        
        # 高光环境
        cls.synth_high_light = np.ones((256, 256, 3), dtype=np.float32) * 0.82
        cls.synth_high_light[:, :, 0] = 0.9

    # =====================================================================
    # 场景识别测试
    # =====================================================================
    def test_detect_scene_type_basic(self):
        """测试场景类型检测基本功能"""
        scene = self.knowledge.detect_scene_type(self.test_image)
        self.assertIn(scene, ['sunny', 'cloudy', 'night', 'indoor'])

    def test_detect_scene_type_sunny(self):
        """测试晴天场景识别"""
        scene = self.knowledge.detect_scene_type(self.synth_sunny)
        self.assertEqual(scene, 'sunny')

    def test_detect_scene_type_cloudy(self):
        """测试阴天场景识别"""
        scene = self.knowledge.detect_scene_type(self.synth_cloudy)
        self.assertEqual(scene, 'cloudy')

    def test_detect_scene_type_night(self):
        """测试夜景场景识别"""
        scene = self.knowledge.detect_scene_type(self.synth_night)
        self.assertEqual(scene, 'night')

    def test_detect_scene_type_indoor(self):
        """测试室内场景识别"""
        scene = self.knowledge.detect_scene_type(self.synth_indoor)
        self.assertEqual(scene, 'indoor')

    def test_detect_scene_type_returns_valid_string(self):
        """测试场景检测返回有效字符串"""
        for img in [self.test_image, self.synth_sunny, self.synth_cloudy, 
                    self.synth_night, self.synth_indoor]:
            scene = self.knowledge.detect_scene_type(img)
            self.assertIsInstance(scene, str)
            self.assertIn(scene, ['sunny', 'cloudy', 'night', 'indoor'])

    # =====================================================================
    # 场景参数推荐测试
    # =====================================================================
    def test_recommend_params_for_scene_auto(self):
        """测试自动场景检测参数推荐"""
        result = self.knowledge.recommend_params_for_scene(self.test_image)
        
        self.assertIn('scene_type', result)
        self.assertIn('description', result)
        self.assertIn('recommended_params', result)
        self.assertIn('suggestions', result)
        self.assertIn('isp_settings', result)
        self.assertIn('scene_confidence', result)
        
        params = result['recommended_params']
        self.assertIn('exposure_bias', params)
        self.assertIn('contrast', params)
        self.assertIn('denoise', params)

    def test_recommend_params_for_scene_sunny(self):
        """测试晴天场景参数推荐"""
        result = self.knowledge.recommend_params_for_scene(
            self.synth_sunny, scene_type='sunny'
        )
        
        self.assertEqual(result['scene_type'], 'sunny')
        self.assertIn('highlight_priority', result['recommended_params'])
        # 晴天应该降低对比度
        self.assertLess(result['recommended_params']['contrast'], 1.0)

    def test_recommend_params_for_scene_night(self):
        """测试夜景场景参数推荐"""
        result = self.knowledge.recommend_params_for_scene(
            self.synth_night, scene_type='night'
        )
        
        self.assertEqual(result['scene_type'], 'night')
        # 夜景应该强降噪
        self.assertGreater(result['recommended_params']['denoise'], 1.5)

    def test_recommend_params_for_scene_indoor(self):
        """测试室内场景参数推荐"""
        result = self.knowledge.recommend_params_for_scene(
            self.synth_indoor, scene_type='indoor'
        )
        
        self.assertEqual(result['scene_type'], 'indoor')
        self.assertIn('shadow_recovery', result['recommended_params'])
        # 室内应该适度提亮暗部
        self.assertGreater(result['recommended_params']['shadow_recovery'], 1.0)

    def test_recommend_params_for_scene_cloudy(self):
        """测试阴天场景参数推荐"""
        result = self.knowledge.recommend_params_for_scene(
            self.synth_cloudy, scene_type='cloudy'
        )
        
        self.assertEqual(result['scene_type'], 'cloudy')
        # 阴天应该略微提亮
        self.assertGreater(result['recommended_params']['exposure_bias'], 0)

    def test_recommend_params_for_scene_with_suggestions(self):
        """测试参数推荐包含建议"""
        result = self.knowledge.recommend_params_for_scene(self.test_image)
        self.assertIsInstance(result['suggestions'], list)
        self.assertGreater(len(result['suggestions']), 0)

    def test_scene_confidence_range(self):
        """测试场景置信度范围"""
        result = self.knowledge.recommend_params_for_scene(self.test_image)
        confidence = result['scene_confidence']
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

    # =====================================================================
    # 运动/静止场景参数测试
    # =====================================================================
    def test_recommend_motion_params_basic(self):
        """测试运动参数推荐基本功能"""
        result = self.knowledge.recommend_motion_params(self.test_image)
        
        self.assertIn('scene', result)
        self.assertIn('motion_blur_level', result)
        self.assertIn('recommendations', result)
        self.assertIn('suggestions', result)
        self.assertIn('freeze_motion', result)
        self.assertIn('allow_long_exposure', result)
        
        rec = result['recommendations']
        self.assertIn('shutter_speed', rec)
        self.assertIn('denoise', rec)

    def test_recommend_motion_params_motion_scene(self):
        """测试运动场景参数"""
        result = self.knowledge.recommend_motion_params(
            self.test_image, motion_blur=0.7
        )
        
        self.assertEqual(result['scene'], 'motion')
        self.assertTrue(result['freeze_motion'])
        self.assertFalse(result['allow_long_exposure'])
        self.assertIn('shutter_priority', result['recommendations']['exposure_mode'])

    def test_recommend_motion_params_static_scene(self):
        """测试静止场景参数"""
        result = self.knowledge.recommend_motion_params(
            self.test_image, motion_blur=0.1
        )
        
        self.assertEqual(result['scene'], 'static')
        self.assertFalse(result['freeze_motion'])
        self.assertTrue(result['allow_long_exposure'])
        self.assertIn('aperture_priority', result['recommendations']['exposure_mode'])

    def test_recommend_motion_params_auto_estimate(self):
        """测试运动模糊自动估计"""
        result = self.knowledge.recommend_motion_params(self.test_image)
        self.assertGreaterEqual(result['motion_blur_level'], 0.0)
        self.assertLessEqual(result['motion_blur_level'], 1.0)

    def test_recommend_motion_params_with_sharpness(self):
        """测试运动场景包含锐化建议"""
        result = self.knowledge.recommend_motion_params(
            self.test_image, motion_blur=0.8
        )
        # 运动场景应该建议更高的锐化
        self.assertGreater(result['recommendations']['sharpness'], 1.0)

    # =====================================================================
    # 极端光照环境测试
    # =====================================================================
    def test_recommend_extreme_light_params_basic(self):
        """测试极端光照参数推荐基本功能"""
        result = self.knowledge.recommend_extreme_light_params(self.test_image)
        
        self.assertIn('environment_type', result)
        self.assertIn('brightness', result)
        self.assertIn('distribution', result)
        self.assertIn('recommendations', result)
        self.assertIn('suggestions', result)
        
        dist = result['distribution']
        self.assertIn('dark', dist)
        self.assertIn('shadow', dist)
        self.assertIn('mid', dist)
        self.assertIn('highlight', dist)
        self.assertIn('white', dist)

    def test_recommend_extreme_light_params_low_light(self):
        """测试低光环境参数"""
        result = self.knowledge.recommend_extreme_light_params(self.synth_low_light)
        
        self.assertEqual(result['environment_type'], 'low_light')
        # 低光应该强降噪
        self.assertGreater(result['recommendations']['denoise'], 1.5)
        # 应该提亮
        self.assertGreater(result['recommendations']['exposure_bias'], 0)

    def test_recommend_extreme_light_params_high_light(self):
        """测试高光环境参数"""
        result = self.knowledge.recommend_extreme_light_params(self.synth_high_light)
        
        self.assertIn(result['environment_type'], ['high_light', 'high_contrast'])
        # 高光应该降低曝光
        self.assertLess(result['recommendations']['exposure_bias'], 0)
        # 应该启用高光保护
        self.assertGreater(result['recommendations']['highlight_priority'], 1.0)

    def test_recommend_extreme_light_params_high_contrast(self):
        """测试大光比场景"""
        # 创建同时包含亮部和暗部的图像
        high_contrast = np.zeros((256, 256, 3), dtype=np.float32)
        high_contrast[:128, :, :] = 0.9  # 亮部
        high_contrast[128:, :, :] = 0.1  # 暗部
        
        result = self.knowledge.recommend_extreme_light_params(high_contrast)
        
        self.assertIn(result['environment_type'], ['high_light', 'high_contrast', 'normal'])
        self.assertIn('hdr_recommended', result)

    def test_recommend_extreme_light_hdr_recommended(self):
        """测试HDR推荐"""
        result = self.knowledge.recommend_extreme_light_params(self.synth_high_light)
        self.assertTrue(result['hdr_recommended'])
        
        # 正常光照不应该推荐HDR
        normal = np.ones((256, 256, 3), dtype=np.float32) * 0.5
        result_normal = self.knowledge.recommend_extreme_light_params(normal)
        # 正常光照HDR推荐应该是False
        self.assertFalse(result_normal['hdr_recommended'])

    def test_brightness_range(self):
        """测试亮度分析范围"""
        result = self.knowledge.recommend_extreme_light_params(self.test_image)
        brightness = result['brightness']
        self.assertGreaterEqual(brightness, 0.0)
        self.assertLessEqual(brightness, 1.0)

    def test_distribution_sums_to_one(self):
        """测试亮度分布比例和为1"""
        result = self.knowledge.recommend_extreme_light_params(self.test_image)
        dist = result['distribution']
        total = sum(dist.values())
        self.assertAlmostEqual(total, 1.0, places=2)

    # =====================================================================
    # 集成测试
    # =====================================================================
    def test_all_phase5_functions_return_valid_types(self):
        """确保所有Phase 5函数返回有效类型"""
        # detect_scene_type 返回字符串
        scene = self.knowledge.detect_scene_type(self.test_image)
        self.assertIsInstance(scene, str)
        self.assertIn(scene, ['sunny', 'cloudy', 'night', 'indoor'])
        
        # 其他函数返回dict
        for func, name in [
            (lambda: self.knowledge.recommend_params_for_scene(self.test_image), 'recommend_params_for_scene'),
            (lambda: self.knowledge.recommend_motion_params(self.test_image), 'recommend_motion_params'),
            (lambda: self.knowledge.recommend_extreme_light_params(self.test_image), 'recommend_extreme_light_params'),
        ]:
            result = func()
            self.assertIsInstance(result, dict, f"{name} should return dict")

    def test_scene_detection_consistency(self):
        """测试场景检测一致性"""
        # 同一图像多次检测应该返回相同结果
        scenes = [self.knowledge.detect_scene_type(self.test_image) for _ in range(3)]
        self.assertEqual(len(set(scenes)), 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
