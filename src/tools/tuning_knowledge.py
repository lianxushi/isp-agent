#!/usr/bin/env python3
"""
ISP调优知识库
基于规则和经验的调优建议生成
"""
from typing import Dict, Any, List, Optional
from ..utils.logger import setup_logger

logger = setup_logger('isp-agent.knowledge')


class ISPTuningKnowledge:
    """
    ISP调优知识库
    
    包含：
    - 场景识别与参数推荐
    - 常见问题诊断
    - 调优建议生成
    """
    
    # 场景参数模板
    SCENE_PRESETS = {
        'landscape': {
            'name': '风景',
            'iso': 100,
            'saturation': 1.2,
            'contrast': 1.1,
            'sharpness': 1.0,
            'denoise': 0.8,
        },
        'portrait': {
            'name': '人像',
            'iso': 200,
            'saturation': 0.9,
            'contrast': 1.0,
            'sharpness': 0.8,
            'denoise': 1.0,
        },
        'night': {
            'name': '夜景',
            'iso': 1600,
            'saturation': 1.1,
            'contrast': 1.2,
            'sharpness': 1.1,
            'denoise': 1.5,
        },
        'sports': {
            'name': '运动',
            'iso': 800,
            'saturation': 1.0,
            'contrast': 1.1,
            'sharpness': 1.3,
            'denoise': 0.7,
        },
        'indoor': {
            'name': '室内',
            'iso': 400,
            'saturation': 1.0,
            'contrast': 1.05,
            'sharpness': 1.0,
            'denoise': 1.2,
        },
        'automotive': {
            'name': '车载',
            'iso': 600,
            'saturation': 1.1,
            'contrast': 1.15,
            'sharpness': 1.2,
            'denoise': 1.0,
        },
    }
    
    # 问题症状与解决方案
    PROBLEM_SOLUTIONS = {
        'too_dark': {
            'symptoms': ['画面偏暗', '暗部细节丢失'],
            'causes': ['曝光不足', 'ISO过低', '快门过快'],
            'solutions': [
                '增加曝光补偿 +0.5~1.0 EV',
                '提高ISO (建议不超过1600)',
                '降低快门速度',
                '启用HDR或多帧合成',
            ]
        },
        'too_bright': {
            'symptoms': ['画面过曝', '高光溢出'],
            'causes': ['曝光过度', 'ISO过高', '快门过慢'],
            'solutions': [
                '减少曝光补偿 -0.5~1.0 EV',
                '降低ISO',
                '提高快门速度',
                '启用自动曝光锁定',
            ]
        },
        'noise': {
            'symptoms': ['噪点明显', '颗粒感强'],
            'causes': ['ISO过高', '降噪不足', '暗光拍摄'],
            'solutions': [
                '降低ISO',
                '增强降噪强度 (LDC 1.2~1.5)',
                '使用多帧降噪',
                '增加环境光照',
            ]
        },
        'blur': {
            'symptoms': ['画面模糊', '细节丢失'],
            'causes': ['抖动', '对焦失败', '快门过慢'],
            'solutions': [
                '启用EIS/OIS防抖',
                '提高快门速度 (1/快门 > 1/焦距*2)',
                '检查对焦系统',
                '使用三脚架',
            ]
        },
        'color_cast': {
            'symptoms': ['色彩偏色', '白平衡不准'],
            'causes': ['白平衡设置错误', '光源色温不匹配'],
            'solutions': [
                '调整白平衡模式 (自动/日光/阴天/钨丝灯)',
                '手动调整色温 (K值)',
                '使用灰卡校准',
                '调整R/B增益',
            ]
        },
        'low_saturation': {
            'symptoms': ['色彩暗淡', '不够鲜艳'],
            'causes': ['饱和度设置低', '拍摄场景本身色彩低'],
            'solutions': [
                '提高饱和度 +0.2~+0.5',
                '提高自然饱和度',
                '启用场景模式',
            ]
        },
        'high_saturation': {
            'symptoms': ['色彩过艳', '颜色溢出'],
            'causes': ['饱和度过高', '色彩空间设置错误'],
            'solutions': [
                '降低饱和度 -0.2~-0.5',
                '检查色彩空间 (sRGB/Rec.709)',
            ]
        },
        'artifact': {
            'symptoms': ['块效应', '振铃', '摩尔纹'],
            'causes': ['压缩过度', '锐化过度', '传感器混叠'],
            'solutions': [
                '降低压缩率',
                '调整锐化强度',
                '使用抗混叠滤镜',
                '安装光学低通滤镜',
            ]
        },
    }
    
    # 自动驾驶场景特定建议
    AUTOMOTIVE_RECOMMENDATIONS = {
        'adas_front': {
            'description': '前视ADAS摄像头',
            'priority': ['sharpness', 'dynamic_range', 'framerate'],
            'suggestions': [
                '推荐分辨率: 1920x1080 或更高',
                '帧率: 30fps (高速场景60fps)',
                '启用HDR以应对逆光',
                '锐化适度 (避免边缘伪影)',
                '色彩: 自然还原，无需艺术处理',
            ]
        },
        'adas_rear': {
            'description': '后视ADAS摄像头',
            'priority': ['exposure', 'dynamic_range'],
            'suggestions': [
                '宽动态范围 (100dB+)',
                '应对车库/隧道等大光比场景',
                '帧率: 30fps',
                '夜间补光建议',
            ]
        },
        'surround': {
            'description': '环视摄像头',
            'priority': ['distortion', 'stitching', 'brightness'],
            'suggestions': [
                '低畸变镜头 (桶形畸变<10%)',
                '亮度均匀性',
                '拼接边缘融合优化',
                '帧率: 15-30fps',
            ]
        },
        'dms': {
            'description': '驾驶员监控摄像头',
            'priority': ['sharpness', 'ir_response', 'low_light'],
            'suggestions': [
                '近红外响应 (940nm)',
                '高锐度用于面部识别',
                '逆光补偿',
                '帧率: 25-30fps',
                '隐私保护处理',
            ]
        },
    }
    
    def __init__(self):
        pass
    
    def get_preset(self, scene_type: str) -> Optional[Dict[str, Any]]:
        """获取场景参数模板"""
        return self.SCENE_PRESETS.get(scene_type)
    
    def diagnose(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        基于分析结果诊断问题
        
        Args:
            analysis_result: 图像分析结果 (来自ImageAnalyzer)
        
        Returns:
            Dict: 诊断结果和建议
        """
        issues = []
        solutions = []
        
        # 亮度问题
        brightness = analysis_result.get('brightness')
        if brightness:
            if brightness < 50:
                issues.append('too_dark')
            elif brightness > 200:
                issues.append('too_bright')
        
        # 噪声问题
        noise = analysis_result.get('noise_level')
        if noise and noise > 30:
            issues.append('noise')
        
        # 清晰度问题
        contrast = analysis_result.get('contrast')
        if contrast and contrast < 30:
            issues.append('blur')
        
        # 色彩问题
        color_analysis = analysis_result.get('color_analysis')
        if color_analysis:
            wb = color_analysis.get('white_balance', '')
            if '偏' in wb:
                issues.append('color_cast')
            
            sat = color_analysis.get('saturation', 100)
            if sat < 40:
                issues.append('low_saturation')
            elif sat > 180:
                issues.append('high_saturation')
        
        # 动态范围
        dynamic_range = analysis_result.get('dynamic_range')
        if dynamic_range:
            useful_range = dynamic_range.get('useful_range', 0)
            if useful_range < 100:
                issues.append('low_dynamic_range')
        
        # 生成建议
        for issue in issues:
            if issue in self.PROBLEM_SOLUTIONS:
                sol = self.PROBLEM_SOLUTIONS[issue]
                solutions.append({
                    'issue': sol['symptoms'][0],
                    'causes': sol['causes'],
                    'solutions': sol['solutions']
                })
        
        return {
            'issues_found': len(issues),
            'issues': issues,
            'recommendations': solutions,
        }
    
    def get_automotive_recommendations(
        self,
        scene_type: str,
        quality_scores: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """获取车载场景特定建议"""
        if scene_type not in self.AUTOMOTIVE_RECOMMENDATIONS:
            return {'error': f'未知场景类型: {scene_type}'}
        
        rec = self.AUTOMOTIVE_RECOMMENDATIONS[scene_type]
        
        # 如果有质量评分，生成针对性建议
        specific = []
        if quality_scores:
            scores = quality_scores
            
            if scores.get('sharpness', 100) < 70:
                specific.append('⚠️ 清晰度不足，建议增加锐化强度')
            
            if scores.get('noise', 100) < 60:
                specific.append('⚠️ 噪声明显，建议增强降噪或增加照明')
            
            if scores.get('dynamic_range', 100) < 60:
                specific.append('⚠️ 动态范围不足，建议启用HDR模式')
        
        return {
            'scene_type': scene_type,
            'description': rec['description'],
            'priority': rec['priority'],
            'general_suggestions': rec['suggestions'],
            'specific_suggestions': specific,
        }
    
    def generate_tuning_suggestions(
        self,
        analysis_result: Dict[str, Any],
        scene_type: str = 'auto'
    ) -> Dict[str, Any]:
        """
        生成综合调优建议
        
        Args:
            analysis_result: 图像分析结果
            scene_type: 场景类型
        
        Returns:
            Dict: 调优建议
        """
        # 诊断问题
        diagnosis = self.diagnose(analysis_result)
        
        # 获取场景预设
        preset = self.get_preset(scene_type) if scene_type != 'auto' else None
        
        # 生成参数建议
        params_suggestion = {}
        
        if preset:
            params_suggestion = {
                'iso': preset.get('iso'),
                'saturation': preset.get('saturation'),
                'contrast': preset.get('contrast'),
                'sharpness': preset.get('sharpness'),
                'denoise': preset.get('denoise'),
            }
        
        return {
            'diagnosis': diagnosis,
            'scene_preset': preset,
            'suggested_params': params_suggestion,
            'overall_assessment': self._generate_assessment(diagnosis),
        }
    
    def _generate_assessment(self, diagnosis: Dict[str, Any]) -> str:
        """生成总体评估"""
        issues_count = diagnosis.get('issues_found', 0)
        
        if issues_count == 0:
            return "图像质量良好，无需重大调整"
        elif issues_count == 1:
            return "发现1个问题，建议按上述方案调整"
        elif issues_count <= 3:
            return f"发现{issues_count}个问题，建议逐一排查调整"
        else:
            return f"发现{issues_count}个问题，建议系统性地检查ISP pipeline各模块"


def create_tuning_knowledge() -> ISPTuningKnowledge:
    """创建知识库实例"""
    return ISPTuningKnowledge()
