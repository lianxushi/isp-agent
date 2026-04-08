#!/usr/bin/env python3
"""
调参建议引擎 (Tuning Advisor)
基于ISP分析结果的LLM调参建议

调用MiniMax/GPT API，根据以下分析结果生成ISP参数调优建议:
- BRISQUE质量评分
- HDR动态范围分析
- ISO16505合规性评估
- ΔE色彩准确性
- 锐度/MTF评估
- 噪声水平分析

参考 PRD 2.3 节 ISP参数诊断内容

Author: ISP Team
"""

import os
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum

try:
    from ..agent.llm_client import LLMClient, LLMAPIError
    from ..utils.logger import setup_logger
except ImportError:
    LLMClient = None
    import logging
    def setup_logger(name):
        logging.basicConfig(level=20, format='%(message)s')
        return logging.getLogger(name)

logger = setup_logger('isp-agent.tuning_advisor')


class ProblemCategory(Enum):
    """问题分类"""
    NOISE = "高噪声"
    BLUR = "模糊/锐度不足"
    COLOR = "色彩偏差"
    HDR = "动态范围问题"
    EXPOSURE = "曝光问题"
    ARTIFACT = "伪影/畸变"
    OVERALL = "综合质量问题"


@dataclass
class ISPDiagnosis:
    """ISP诊断结果"""
    category: str = ""
    severity: str = "normal"  # normal, mild, moderate, severe
    score: float = 100.0
    findings: List[str] = field(default_factory=list)
    affected_params: List[str] = field(default_factory=list)


@dataclass
class TuningRecommendation:
    """调参建议"""
    param_name: str = ""
    current_value: Any = None
    suggested_value: Any = None
    direction: str = ""  # increase, decrease, adjust
    reason: str = ""
    priority: int = 0  # 1=最高优先级
    confidence: float = 1.0  # 0-1


class TuningAdvisor:
    """
    ISP调参建议引擎

    根据图像质量分析结果，结合专家知识库，
    生成针对性的ISP参数调整建议。

    支持的ISP参数调优:
    - 曝光: AE_target, exposure_time, analog_gain
    - 降噪: noise_reduction_strength, spatial_denoise, temporal_denoise
    - 锐化: sharpen_strength, edge_enhancement
    - 色彩: saturation, hue, white_balance, color_correction
    - 对比度: contrast, tone_mapping, gamma
    - HDR: hdr_strength, tone_mapping_mode, local_dimming
    """

    # ISP参数专家知识库 (问题 → 参数映射)
    EXPERT_RULES: Dict[ProblemCategory, List[Dict[str, Any]]] = {
        ProblemCategory.NOISE: [
            {
                'param': 'noise_reduction_strength',
                'direction': 'increase',
                'condition': 'severity >= moderate',
                'priority': 1,
                'reason': '图像噪声明显，建议增强降噪强度'
            },
            {
                'param': 'spatial_denoise',
                'direction': 'increase',
                'condition': 'severity >= mild',
                'priority': 2,
                'reason': '空间域降噪可有效减少随机噪声'
            },
            {
                'param': 'temporal_denoise',
                'direction': 'increase',
                'condition': 'severity >= moderate and is_video',
                'priority': 2,
                'reason': '时域降噪可平滑帧间噪声'
            },
            {
                'param': 'analog_gain',
                'direction': 'decrease',
                'condition': 'severity >= mild and exposure_too_high',
                'priority': 3,
                'reason': '过高增益会放大噪声，考虑降低模拟增益'
            },
        ],
        ProblemCategory.BLUR: [
            {
                'param': 'sharpen_strength',
                'direction': 'increase',
                'condition': 'severity >= mild',
                'priority': 1,
                'reason': '锐度不足，建议增强锐化强度'
            },
            {
                'param': 'edge_enhancement',
                'direction': 'increase',
                'condition': 'severity >= moderate',
                'priority': 2,
                'reason': '边缘增强可改善轮廓清晰度'
            },
            {
                'param': 'noise_reduction_strength',
                'direction': 'decrease',
                'condition': 'severity >= mild and nr_too_high',
                'priority': 1,
                'reason': '过度降噪可能导致细节丢失，适当降低'
            },
        ],
        ProblemCategory.COLOR: [
            {
                'param': 'saturation',
                'direction': 'adjust',
                'condition': 'severity >= mild',
                'priority': 1,
                'reason': '色彩饱和度偏差，建议调整'
            },
            {
                'param': 'white_balance',
                'direction': 'adjust',
                'condition': 'severity >= moderate and is_color_cast',
                'priority': 1,
                'reason': '白平衡偏差，建议重新校准'
            },
            {
                'param': 'color_correction',
                'direction': 'adjust',
                'condition': 'severity >= mild',
                'priority': 2,
                'reason': '色彩校正矩阵可能需要调整'
            },
        ],
        ProblemCategory.HDR: [
            {
                'param': 'hdr_strength',
                'direction': 'increase',
                'condition': 'severity >= moderate and dr_low',
                'priority': 1,
                'reason': '动态范围不足，建议增强HDR处理'
            },
            {
                'param': 'tone_mapping_mode',
                'direction': 'adjust',
                'condition': 'severity >= mild',
                'priority': 2,
                'reason': 'Tone mapping模式可能不适合当前场景'
            },
            {
                'param': 'local_dimming',
                'direction': 'increase',
                'condition': 'severity >= moderate and local_contrast_low',
                'priority': 3,
                'reason': '局部对比度不足，建议增强局部调光'
            },
        ],
        ProblemCategory.EXPOSURE: [
            {
                'param': 'AE_target',
                'direction': 'adjust',
                'condition': 'severity >= mild',
                'priority': 1,
                'reason': '曝光目标值需要调整'
            },
            {
                'param': 'exposure_time',
                'direction': 'adjust',
                'condition': 'severity >= moderate',
                'priority': 2,
                'reason': '曝光时间建议根据场景调整'
            },
            {
                'param': 'analog_gain',
                'direction': 'decrease',
                'condition': 'severity >= mild and gain_too_high',
                'priority': 2,
                'reason': '过高增益引入噪声，建议优先调整曝光时间'
            },
        ],
        ProblemCategory.ARTIFACT: [
            {
                'param': 'noise_reduction_strength',
                'direction': 'decrease',
                'condition': 'severity >= mild and banding_present',
                'priority': 1,
                'reason': '降噪可能引入色带/条纹伪影，建议降低'
            },
            {
                'param': 'sharpen_strength',
                'direction': 'decrease',
                'condition': 'severity >= moderate and ringing_present',
                'priority': 1,
                'reason': '过强锐化导致振铃伪影，建议降低'
            },
            {
                'param': 'compression_quality',
                'direction': 'increase',
                'condition': 'severity >= mild and blockiness_present',
                'priority': 2,
                'reason': '压缩伪影明显，建议降低压缩率'
            },
        ],
    }

    # 参数默认调整步长
    PARAM_STEPS: Dict[str, Dict[str, Any]] = {
        'noise_reduction_strength': {'min': 0, 'max': 100, 'step': 10, 'unit': ''},
        'spatial_denoise': {'min': 0, 'max': 100, 'step': 10, 'unit': ''},
        'temporal_denoise': {'min': 0, 'max': 100, 'step': 10, 'unit': ''},
        'sharpen_strength': {'min': 0, 'max': 100, 'step': 5, 'unit': ''},
        'edge_enhancement': {'min': 0, 'max': 100, 'step': 5, 'unit': ''},
        'saturation': {'min': 0, 'max': 200, 'step': 10, 'unit': '%', 'neutral': 100},
        'AE_target': {'min': 0, 'max': 255, 'step': 5, 'unit': ''},
        'exposure_time': {'min': 0, 'max': 1000000, 'step': 1000, 'unit': 'us'},
        'analog_gain': {'min': 0, 'max': 800, 'step': 50, 'unit': 'permille', 'neutral': 1000},
        'hdr_strength': {'min': 0, 'max': 100, 'step': 10, 'unit': ''},
        'contrast': {'min': 0, 'max': 200, 'step': 10, 'unit': '%', 'neutral': 100},
        'gamma': {'min': 1.0, 'max': 3.0, 'step': 0.1, 'unit': '', 'neutral': 2.2},
        'tone_mapping_mode': {'values': ['reinhard', 'aces', 'drago', 'mantik'], 'unit': ''},
        'local_dimming': {'min': 0, 'max': 100, 'step': 10, 'unit': ''},
    }

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        provider: str = 'minimax',
        model: str = 'MiniMax-M2.5'
    ):
        """
        初始化调参建议引擎

        Args:
            llm_client: 已有LLM客户端实例
            provider: LLM提供商 (minimax/openai)
            model: 模型名称
        """
        self.llm_client = llm_client
        self.provider = provider
        self.model = model
        self._init_llm_client()

    def _init_llm_client(self):
        """初始化LLM客户端"""
        if self.llm_client is not None:
            return

        if LLMClient is None:
            logger.warning("LLMClient导入失败，调参建议将使用规则引擎")
            return

        try:
            self.llm_client = LLMClient(
                provider=self.provider,
                model=self.model,
                temperature=0.3,
                max_tokens=2000
            )
            logger.info(f"TuningAdvisor LLM客户端初始化: {self.provider}/{self.model}")
        except Exception as e:
            logger.warning(f"LLM客户端初始化失败: {e}，使用规则引擎")
            self.llm_client = None

    def diagnose(
        self,
        brisque_score: Optional[float] = None,
        hdr_analysis: Optional[Dict[str, Any]] = None,
        iso16505_result: Optional[Dict[str, Any]] = None,
        delta_e: Optional[float] = None,
        snr_db: Optional[float] = None,
        mtf_nyquist: Optional[float] = None,
        **kwargs
    ) -> List[ISPDiagnosis]:
        """
        基于分析结果进行ISP问题诊断

        Args:
            brisque_score: BRISQUE质量评分 (0-100)
            hdr_analysis: HDR动态范围分析结果
            iso16505_result: ISO16505评估结果
            delta_e: ΔE色彩误差
            snr_db: 信噪比(dB)
            mtf_nyquist: MTF@Nyquist值
            **kwargs: 其他指标

        Returns:
            诊断结果列表
        """
        diagnoses = []

        # BRISQUE综合评分诊断
        if brisque_score is not None:
            if brisque_score < 40:
                diagnoses.append(ISPDiagnosis(
                    category=ProblemCategory.OVERALL.value,
                    severity='severe',
                    score=brisque_score,
                    findings=[f'BRISQUE综合评分偏低: {brisque_score:.1f}'],
                    affected_params=['noise_reduction_strength', 'sharpen_strength', 'contrast']
                ))
            elif brisque_score < 60:
                diagnoses.append(ISPDiagnosis(
                    category=ProblemCategory.OVERALL.value,
                    severity='moderate',
                    score=brisque_score,
                    findings=[f'BRISQUE综合评分一般: {brisque_score:.1f}'],
                    affected_params=['noise_reduction_strength', 'sharpen_strength']
                ))

        # 噪声诊断 (SNR)
        if snr_db is not None:
            if snr_db < 30:
                diagnoses.append(ISPDiagnosis(
                    category=ProblemCategory.NOISE.value,
                    severity='severe',
                    score=max(0, min(100, snr_db * 3.3)),
                    findings=[f'SNR过低: {snr_db:.1f}dB'],
                    affected_params=['noise_reduction_strength', 'spatial_denoise', 'analog_gain']
                ))
            elif snr_db < 40:
                diagnoses.append(ISPDiagnosis(
                    category=ProblemCategory.NOISE.value,
                    severity='moderate',
                    score=max(0, min(100, snr_db * 2.5)),
                    findings=[f'SNR偏低: {snr_db:.1f}dB'],
                    affected_params=['noise_reduction_strength', 'spatial_denoise']
                ))

        # 锐度诊断 (MTF)
        if mtf_nyquist is not None:
            if mtf_nyquist < 0.25:
                diagnoses.append(ISPDiagnosis(
                    category=ProblemCategory.BLUR.value,
                    severity='severe',
                    score=max(0, min(100, mtf_nyquist * 400)),
                    findings=[f'MTF@Nyquist过低: {mtf_nyquist:.3f}，图像模糊'],
                    affected_params=['sharpen_strength', 'edge_enhancement', 'noise_reduction_strength']
                ))
            elif mtf_nyquist < 0.40:
                diagnoses.append(ISPDiagnosis(
                    category=ProblemCategory.BLUR.value,
                    severity='moderate',
                    score=max(0, min(100, mtf_nyquist * 250)),
                    findings=[f'锐度不足: MTF={mtf_nyquist:.3f}'],
                    affected_params=['sharpen_strength', 'edge_enhancement']
                ))

        # 色彩准确性诊断 (ΔE)
        if delta_e is not None:
            if delta_e > 14:
                diagnoses.append(ISPDiagnosis(
                    category=ProblemCategory.COLOR.value,
                    severity='severe',
                    score=max(0, min(100, 100 - (delta_e - 14) * 5)),
                    findings=[f'色彩偏差过大: ΔE={delta_e:.1f}'],
                    affected_params=['saturation', 'white_balance', 'color_correction']
                ))
            elif delta_e > 10:
                diagnoses.append(ISPDiagnosis(
                    category=ProblemCategory.COLOR.value,
                    severity='mild',
                    score=max(0, min(100, 100 - (delta_e - 10) * 10)),
                    findings=[f'色彩轻微偏差: ΔE={delta_e:.1f}'],
                    affected_params=['saturation', 'color_correction']
                ))

        # HDR动态范围诊断
        if hdr_analysis:
            dr = hdr_analysis.get('dynamic_range', {})
            stops = dr.get('stops', 0)
            exposure = hdr_analysis.get('exposure_analysis', {})
            over_pct = exposure.get('over_exposed_percent', 0)
            under_pct = exposure.get('under_exposed_percent', 0)

            if stops < 6:
                diagnoses.append(ISPDiagnosis(
                    category=ProblemCategory.HDR.value,
                    severity='moderate' if stops < 4 else 'mild',
                    score=max(0, min(100, stops * 16.7)),
                    findings=[f'动态范围受限: {stops:.1f} stops'],
                    affected_params=['hdr_strength', 'tone_mapping_mode', 'local_dimming']
                ))

            if over_pct > 10 or under_pct > 20:
                diagnoses.append(ISPDiagnosis(
                    category=ProblemCategory.EXPOSURE.value,
                    severity='severe' if over_pct > 15 or under_pct > 30 else 'moderate',
                    score=max(0, min(100, 100 - (over_pct + under_pct))),
                    findings=[f'曝光不均衡: 过曝{over_pct:.1f}%, 欠曝{under_pct:.1f}%'],
                    affected_params=['AE_target', 'exposure_time', 'hdr_strength']
                ))

        # ISO16505综合诊断
        if iso16505_result:
            score = iso16505_result.get('overall_score', 100)
            if score < 60:
                diagnoses.append(ISPDiagnosis(
                    category=ProblemCategory.OVERALL.value,
                    severity='severe',
                    score=score,
                    findings=[f'ISO16505综合评分不足: {score:.1f}'],
                    affected_params=['noise_reduction_strength', 'sharpen_strength', 'hdr_strength']
                ))

        return diagnoses

    def get_rule_based_recommendations(
        self,
        diagnoses: List[ISPDiagnosis],
        current_params: Optional[Dict[str, Any]] = None,
        is_video: bool = False
    ) -> List[TuningRecommendation]:
        """
        基于规则的调参建议 (不依赖LLM)

        Args:
            diagnoses: 诊断结果列表
            current_params: 当前ISP参数值
            is_video: 是否为视频模式

        Returns:
            调参建议列表
        """
        recommendations = []
        current = current_params or {}

        for diag in diagnoses:
            try:
                category = ProblemCategory(diag.category)
            except ValueError:
                category = ProblemCategory.OVERALL

            rules = self.EXPERT_RULES.get(category, [])
            severity_val = {'normal': 0, 'mild': 1, 'moderate': 2, 'severe': 3}.get(diag.severity, 0)

            for rule in rules:
                # 检查条件
                condition = rule.get('condition', '')
                # 简单条件解析
                cond_met = True
                if 'severity >=' in condition:
                    parts = condition.split('>=')
                    level_str = parts[1].strip().replace(')', '')
                    level_val = {'mild': 1, 'moderate': 2, 'severe': 3}.get(level_str, 0)
                    cond_met = cond_met and severity_val >= level_val

                if 'is_video' in condition and not is_video:
                    cond_met = False

                if not cond_met:
                    continue

                param = rule['param']
                cur_val = current.get(param)
                step_info = self.PARAM_STEPS.get(param, {'step': 10})

                # 计算建议值
                direction = rule['direction']
                if direction == 'increase':
                    if 'values' in step_info:
                        # 枚举类型，循环选择下一个
                        vals = step_info['values']
                        idx = 0 if cur_val not in vals else vals.index(cur_val) + 1
                        suggested = vals[idx % len(vals)]
                    else:
                        cur = float(cur_val) if cur_val is not None else step_info.get('min', 0)
                        step = step_info.get('step', 10)
                        suggested = min(cur + step, step_info.get('max', 100))
                elif direction == 'decrease':
                    if 'values' in step_info:
                        vals = step_info['values']
                        idx = len(vals) - 1 if cur_val not in vals else vals.index(cur_val) - 1
                        suggested = vals[max(0, idx)]
                    else:
                        cur = float(cur_val) if cur_val is not None else step_info.get('max', 100)
                        step = step_info.get('step', 10)
                        suggested = max(cur - step, step_info.get('min', 0))
                else:  # adjust
                    suggested = cur_val if cur_val is not None else step_info.get('neutral', step_info.get('step', 50))

                unit = step_info.get('unit', '')
                recommendations.append(TuningRecommendation(
                    param_name=param,
                    current_value=cur_val,
                    suggested_value=suggested,
                    direction=direction,
                    reason=rule['reason'],
                    priority=rule.get('priority', 5),
                    confidence=0.85
                ))

        # 去重并按优先级排序
        seen = set()
        unique_recs = []
        for rec in sorted(recommendations, key=lambda r: r.priority):
            if rec.param_name not in seen:
                seen.add(rec.param_name)
                unique_recs.append(rec)

        return unique_recs

    def generate_llm_recommendations(
        self,
        diagnoses: List[ISPDiagnosis],
        recommendations: List[TuningRecommendation],
        scene_context: Optional[str] = None
    ) -> str:
        """
        使用LLM生成自然语言调参建议

        Args:
            diagnoses: 诊断结果
            recommendations: 规则引擎生成的初步建议
            scene_context: 场景上下文描述

        Returns:
            LLM生成的调参建议文本
        """
        if self.llm_client is None:
            return self._format_recommendations_text(recommendations)

        # 构建prompt
        diag_text = '\n'.join(
            f"- [{d.severity}] {d.category}: {', '.join(d.findings)} (评分: {d.score:.1f})"
            for d in diagnoses
        )

        rec_text = '\n'.join(
            f"- {r.param_name}: {r.current_value} → {r.suggested_value} ({r.direction})"
            f"  原因: {r.reason}"
            for r in recommendations
        )

        prompt = f"""你是一名ISP图像质量调优专家。根据以下诊断结果和初步调参建议，
生成详细的中文ISP参数调整指导。

## 诊断结果
{diag_text or '无明显异常'}

## 初步调参建议
{rec_text or '无需调参'}

## 场景信息
{scene_context or '通用场景'}

请按以下格式输出:
1. 问题总结 (用一句话概括主要问题)
2. 参数调整优先级列表 (编号, 参数名, 调整方向和幅度, 具体原因)
3. 调参注意事项 (如多参数冲突时的权衡策略)
4. 预期效果 (调整后可能的图像质量改善程度)

请使用中文回答，语言简洁专业。"""

        try:
            messages = [
                {"role": "system", "content": "你是一名专业的ISP图像质量调优专家，擅长根据图像分析结果给出精准的参数调整建议。"},
                {"role": "user", "content": prompt}
            ]
            response = self.llm_client.chat(messages, temperature=0.3)
            return response
        except Exception as e:
            logger.warning(f"LLM建议生成失败: {e}")
            return self._format_recommendations_text(recommendations)

    def _format_recommendations_text(self, recommendations: List[TuningRecommendation]) -> str:
        """将调参建议格式化为文本"""
        if not recommendations:
            return "✅ 当前图像质量良好，无需调参。"

        lines = ["📋 调参建议:", ""]
        for i, rec in enumerate(recommendations, 1):
            unit = self.PARAM_STEPS.get(rec.param_name, {}).get('unit', '')
            cur_str = f"{rec.current_value}{unit}" if rec.current_value is not None else "未设置"
            sug_str = f"{rec.suggested_value}{unit}" if rec.suggested_value is not None else "待设置"
            arrow = "↑" if rec.direction == 'increase' else "↓" if rec.direction == 'decrease' else "↔"
            lines.append(f"{i}. **{rec.param_name}**: {cur_str} {arrow} {sug_str}")
            lines.append(f"   原因: {rec.reason}")
            lines.append("")

        return '\n'.join(lines)

    def advise(
        self,
        brisque_score: Optional[float] = None,
        hdr_analysis: Optional[Dict[str, Any]] = None,
        iso16505_result: Optional[Dict[str, Any]] = None,
        delta_e: Optional[float] = None,
        snr_db: Optional[float] = None,
        mtf_nyquist: Optional[float] = None,
        current_params: Optional[Dict[str, Any]] = None,
        scene_context: Optional[str] = None,
        use_llm: bool = True,
        is_video: bool = False
    ) -> Dict[str, Any]:
        """
        主入口：生成完整调参建议

        Args:
            brisque_score: BRISQUE评分 (0-100)
            hdr_analysis: HDR分析结果
            iso16505_result: ISO16505评估结果
            delta_e: 色彩误差
            snr_db: 信噪比(dB)
            mtf_nyquist: MTF@Nyquist
            current_params: 当前ISP参数
            scene_context: 场景描述
            use_llm: 是否使用LLM生成建议
            is_video: 是否为视频

        Returns:
            Dict包含:
              - diagnoses: 诊断结果
              - recommendations: 调参建议列表
              - llm_advice: LLM自然语言建议(如果use_llm=True)
        """
        # 1. 诊断
        diagnoses = self.diagnose(
            brisque_score=brisque_score,
            hdr_analysis=hdr_analysis,
            iso16505_result=iso16505_result,
            delta_e=delta_e,
            snr_db=snr_db,
            mtf_nyquist=mtf_nyquist
        )

        # 2. 规则引擎生成初步建议
        recommendations = self.get_rule_based_recommendations(
            diagnoses, current_params, is_video
        )

        # 3. LLM生成自然语言建议
        llm_advice = ""
        if use_llm and diagnoses:
            llm_advice = self.generate_llm_recommendations(
                diagnoses, recommendations, scene_context
            )

        return {
            'diagnoses': [asdict(d) for d in diagnoses],
            'recommendations': [asdict(r) for r in recommendations],
            'llm_advice': llm_advice,
            'summary': (
                f"发现{len(diagnoses)}个问题，生成{len(recommendations)}条调参建议"
                if diagnoses else "图像质量正常，无需调参"
            )
        }


# =============================================================================
# CLI 便捷函数
# =============================================================================

def advise_tuning(
    brisque_score: Optional[float] = None,
    snr_db: Optional[float] = None,
    delta_e: Optional[float] = None,
    mtf_nyquist: Optional[float] = None,
    hdr_analysis: Optional[Dict[str, Any]] = None,
    iso16505_result: Optional[Dict[str, Any]] = None,
    use_llm: bool = True,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """便捷函数：生成调参建议"""
    advisor = TuningAdvisor()
    result = advisor.advise(
        brisque_score=brisque_score,
        snr_db=snr_db,
        delta_e=delta_e,
        mtf_nyquist=mtf_nyquist,
        hdr_analysis=hdr_analysis,
        iso16505_result=iso16505_result,
        use_llm=use_llm,
    )

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"调参建议已保存: {output_path}")

    return result
