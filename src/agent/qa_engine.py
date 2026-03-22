#!/usr/bin/env python3
"""
QA Engine - 智能问答引擎
基于LLM API进行对话和内容生成
"""
import json
from typing import List, Dict, Any, Optional

from ..agent.llm_client import LLMClient
from ..utils.logger import setup_logger

logger = setup_logger('isp-agent.qa')


class QAEngine:
    """智能问答引擎 - 本地Python + API大脑"""
    
    # System prompt
    SYSTEM_PROMPT = """你是一位专业的ISP（Image Signal Processor）图像调试工程师助手。

你的专业领域：
1. ISP pipeline各模块原理（bayer降噪、AWB、CCM、Demosaic、Gamma、EE等）
2. 图像质量评估（清晰度、噪声、色彩、动态范围）
3. 相机调参技巧（ISO、快门、光圈、焦距对图像的影响）
4. 常见图像问题诊断（偏色、噪点、模糊、条纹）
5. RAW/JPEG/DNG格式解析

请用中文回复，保持专业但易懂。
如果不确定某个参数的具体数值范围，请说明"通常情况下"。
提供具体的ISP参数调整建议时，给出合理的数值范围。"""
    
    # 图像分析上下文模板
    ANALYSIS_CONTEXT_TEMPLATE = """
当前图像分析结果：
- 分辨率: {width} x {height}
- 格式: {format}
- 大小: {size_kb:.1f} KB
- 动态范围: {dynamic_range}
- 噪声水平: {noise_level:.2f}
- 亮度: {brightness:.1f}
- 对比度: {contrast:.1f}
- 色彩分析: {color_analysis}
{full_exif}
"""
    
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        self.conversation_history: List[Dict[str, str]] = []
        self.max_history = 20  # 最多保留20轮对话
    
    def chat(
        self,
        user_input: str,
        context: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        处理用户问答
        
        Args:
            user_input: 用户输入
            context: 上下文（如图像分析结果）
            system_prompt: 自定义system prompt
        
        Returns:
            str: LLM回复
        """
        logger.info(f"处理问答: {user_input[:50]}...")
        
        # 构建消息列表
        messages = [{"role": "system", "content": system_prompt or self.SYSTEM_PROMPT}]
        
        # 添加历史对话
        messages.extend(self.conversation_history[-self.max_history:])
        
        # 构建当前用户消息
        user_message = user_input
        
        # 如果有上下文（图像分析结果），添加到用户消息中
        if context:
            context_str = self._build_context(context)
            user_message += f"\n\n{context_str}"
        
        messages.append({"role": "user", "content": user_message})
        
        # 调用LLM API
        response = self.llm.chat(messages)
        
        # 记录对话历史
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": response})
        
        logger.info(f"问答完成，回复长度: {len(response)}")
        return response
    
    def _build_context(self, context: Dict[str, Any]) -> str:
        """构建上下文字符串"""
        if 'width' in context:  # 图像分析结果
            exif_str = ""
            if context.get('exif'):
                exif_str = f"\n- EXIF信息: {json.dumps(context['exif'], ensure_ascii=False)}"
            
            return self.ANALYSIS_CONTEXT_TEMPLATE.format(
                width=context.get('width', ''),
                height=context.get('height', ''),
                format=context.get('format', ''),
                size_kb=context.get('size_kb', 0),
                dynamic_range=context.get('dynamic_range', {}),
                noise_level=context.get('noise_level', 0),
                brightness=context.get('brightness', 0),
                contrast=context.get('contrast', 0),
                color_analysis=context.get('color_analysis', {}),
                full_exif=exif_str
            )
        
        return str(context)
    
    def generate_suggestion(
        self,
        analysis_result: Dict[str, Any],
        focus_area: Optional[str] = None
    ) -> str:
        """
        根据图像分析结果生成调优建议
        
        Args:
            analysis_result: 图像分析结果
            focus_area: 重点关注领域 (noise/color/sharpness/dynamic_range)
        
        Returns:
            str: 调优建议
        """
        prompt = self._build_suggestion_prompt(analysis_result, focus_area)
        
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        
        return self.llm.chat(messages)
    
    def _build_suggestion_prompt(
        self,
        analysis_result: Dict[str, Any],
        focus_area: Optional[str] = None
    ) -> str:
        """构建调优建议提示"""
        prompt = "请基于以下图像分析结果，给出ISP参数调整建议：\n\n"
        
        # 基本信息
        prompt += f"【基本信息】\n"
        prompt += f"- 分辨率: {analysis_result.get('width', '?')}x{analysis_result.get('height', '?')}\n"
        prompt += f"- 格式: {analysis_result.get('format', '?')}\n\n"
        
        # 质量指标
        prompt += f"【质量指标】\n"
        if analysis_result.get('dynamic_range'):
            dr = analysis_result['dynamic_range']
            prompt += f"- 动态范围: {dr.get('useful_range', '?')}(有效范围) / {dr.get('range', '?')}(总范围)\n"
        
        if analysis_result.get('noise_level'):
            prompt += f"- 噪声水平: {analysis_result['noise_level']:.2f}\n"
        
        if analysis_result.get('brightness'):
            prompt += f"- 亮度: {analysis_result['brightness']:.1f} (正常范围: 80-160)\n"
        
        if analysis_result.get('contrast'):
            prompt += f"- 对比度: {analysis_result['contrast']:.1f} (正常范围: 40-80)\n"
        
        # 色彩
        if analysis_result.get('color_analysis'):
            ca = analysis_result['color_analysis']
            prompt += f"\n【色彩分析】\n"
            prompt += f"- 白平衡: {ca.get('white_balance', '?')}\n"
            prompt += f"- 饱和度: {ca.get('saturation', '?')}\n"
        
        # 重点领域
        if focus_area:
            prompt += f"\n【重点关注】{focus_area}\n"
        
        prompt += "\n请列出具体的ISP参数调整方向和推荐数值范围。"
        
        return prompt
    
    def explain_isp_module(self, module_name: str) -> str:
        """解释ISP模块原理"""
        prompt = f"""请解释ISP pipeline中"{module_name}"模块的工作原理和主要参数调优方法。"""
        
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        
        return self.llm.chat(messages)
    
    def diagnose_issue(self, issue_description: str) -> str:
        """诊断图像问题"""
        prompt = f"""请诊断以下图像问题可能的原因，并给出排查方向和解决方案：

问题描述: {issue_description}

请从以下几个方面分析：
1. 可能涉及的ISP模块
2. 需要检查的参数
3. 建议的调试步骤"""
        
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        
        return self.llm.chat(messages)
    
    def clear_history(self) -> None:
        """清空对话历史"""
        self.conversation_history = []
        logger.info("对话历史已清空")


def create_qa_engine(config: Dict[str, Any]) -> QAEngine:
    """从配置创建QA引擎"""
    from ..agent.llm_client import create_llm_client
    llm_client = create_llm_client(config)
    return QAEngine(llm_client)
