#!/usr/bin/env python3
"""
增强版ISP问答引擎
支持多轮对话、上下文理解、主动提问
"""
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import json

from .tuning_knowledge import ISPTuningKnowledge
from .pipeline_visualizer import ISPPipelineVisualizer


class QueryType(Enum):
    """问题类型"""
    ANALYSIS = "analysis"      # 分析结果查询
    TUNING = "tuning"         # 调优建议
    KNOWLEDGE = "knowledge"  # 知识问答
    COMPARISON = "comparison" # 对比分析
    RECOMMENDATION = "recommendation" # 推荐建议
    GENERAL = "general"       # 通用问答


@dataclass
class ConversationContext:
    """对话上下文"""
    image_analyzed: bool = False
    last_analysis: Optional[Dict] = None
    last_scene_type: Optional[str] = None
    user_preferences: Dict = field(default_factory=dict)
    history: List[Dict] = field(default_factory=list)


class EnhancedQAEngine:
    """
    增强版ISP问答引擎
    
    特性:
    - 多轮对话支持
    - 上下文理解
    - 主动提问引导
    - 场景识别
    - 个性化建议
    """
    
    def __init__(self):
        self.knowledge = ISPTuningKnowledge()
        self.pipeline = ISPPipelineVisualizer()
        self.context = ConversationContext()
    
    def ask(
        self,
        question: str,
        analysis_result: Optional[Dict[str, Any]] = None,
        context: Optional[ConversationContext] = None
    ) -> Dict[str, Any]:
        """
        处理用户问题
        
        Args:
            question: 用户问题
            analysis_result: 图像分析结果(如有)
            context: 对话上下文
        
        Returns:
            Dict: 回答结果
        """
        # 更新上下文
        if context:
            self.context = context
        
        if analysis_result:
            self.context.last_analysis = analysis_result
            self.context.image_analyzed = True
        
        # 分析问题类型
        query_type = self._classify_query(question)
        
        # 生成回答
        if query_type == QueryType.ANALYSIS:
            response = self._handle_analysis_query(question, analysis_result)
        elif query_type == QueryType.TUNING:
            response = self._handle_tuning_query(question, analysis_result)
        elif query_type == QueryType.KNOWLEDGE:
            response = self._handle_knowledge_query(question)
        elif query_type == QueryType.COMPARISON:
            response = self._handle_comparison_query(question)
        elif query_type == QueryType.RECOMMENDATION:
            response = self._handle_recommendation_query(question)
        else:
            response = self._handle_general_query(question)
        
        # 记录历史
        self.context.history.append({
            'question': question,
            'response': response,
            'query_type': query_type.value
        })
        
        return {
            'answer': response['text'],
            'suggestions': response.get('suggestions', []),
            'query_type': query_type.value,
            'needs_context': response.get('needs_context', False)
        }
    
    def _classify_query(self, question: str) -> QueryType:
        """分类问题类型"""
        q = question.lower()
        
        # 分析查询
        if any(k in q for k in ['分析', '怎么样', '如何', '质量', '评分', 'noise', 'sharp']):
            return QueryType.ANALYSIS
        
        # 调优查询
        if any(k in q for k in ['调优', '调整', '优化', '参数', 'tuning', 'setting', 'improve']):
            return QueryType.TUNING
        
        # 知识查询
        if any(k in q for k in ['什么是', '原理', '为什么', 'how', 'what', 'why', 'isp', 'pipeline']):
            return QueryType.KNOWLEDGE
        
        # 对比查询
        if any(k in q for k in ['对比', '比较', '差异', '区别', 'compare', 'different']):
            return QueryType.COMPARISON
        
        # 推荐查询
        if any(k in q for k in ['推荐', '建议', '选择', 'recommend', 'should', 'best']):
            return QueryType.RECOMMENDATION
        
        return QueryType.GENERAL
    
    def _handle_analysis_query(
        self,
        question: str,
        analysis_result: Optional[Dict]
    ) -> Dict:
        """处理分析查询"""
        if not analysis_result:
            return {
                'text': '请先上传图片，我才能分析它的质量。',
                'suggestions': ['帮我分析这张图片的质量', '看看这张照片怎么样'],
                'needs_context': True
            }
        
        # 提取关键指标
        dr = analysis_result.get('dynamic_range', {})
        noise = analysis_result.get('noise_level', 0)
        brightness = analysis_result.get('brightness', 0)
        
        # 生成分析
        issues = []
        if brightness < 50:
            issues.append('画面偏暗')
        elif brightness > 200:
            issues.append('画面过曝')
        
        if noise > 30:
            issues.append(f'噪声明显({noise:.1f})')
        
        if dr.get('useful_range', 0) < 100:
            issues.append('动态范围不足')
        
        if issues:
            text = f'图像分析结果：\n'
            text += f'发现 {len(issues)} 个问题：\n'
            for i, issue in enumerate(issues, 1):
                text += f'{i}. {issue}\n'
        else:
            text = '图像质量良好！各指标正常。'
        
        return {
            'text': text,
            'suggestions': ['如何改善这个问题', '给我调优建议']
        }
    
    def _handle_tuning_query(
        self,
        question: str,
        analysis_result: Optional[Dict]
    ) -> Dict:
        """处理调优查询"""
        if not analysis_result:
            return {
                'text': '请先上传图片，我才能给出针对性的调优建议。',
                'needs_context': True
            }
        
        # 使用知识库生成建议
        tuning = self.knowledge.generate_tuning_suggestions(
            analysis_result,
            self.context.last_scene_type or 'automotive'
        )
        
        text = '🔧 调优建议：\n\n'
        
        for rec in tuning['diagnosis'].get('recommendations', [])[:2]:
            text += f'问题：{rec.get("issue", "N/A")}\n'
            for sol in rec.get('solutions', [])[:2]:
                text += f'  → {sol}\n'
            text += '\n'
        
        if tuning.get('suggested_params'):
            text += '推荐参数：\n'
            for k, v in tuning['suggested_params'].items():
                text += f'  {k}: {v}\n'
        
        return {
            'text': text,
            'suggestions': ['这些参数是什么意思', '应用到视频怎么处理']
        }
    
    def _handle_knowledge_query(self, question: str) -> Dict:
        """处理知识查询"""
        q = question.lower()
        
        # ISP Pipeline相关
        if 'pipeline' in q or 'isp' in q:
            return {
                'text': self.pipeline.visualize('standard'),
                'suggestions': ['详细说说某个模块', '给我看配置']
            }
        
        # 特定模块
        for module in ['demosaic', 'denoise', 'sharpening', 'awb', 'gamma', 'lsc']:
            if module in q:
                return {
                    'text': self.pipeline.explain_stage(module.capitalize()),
                    'suggestions': [f'这个参数怎么调', '还有其他方法吗']
                }
        
        # 默认知识库查询
        return {
            'text': '这是一个ISP相关的知识问题，让我搜索一下...',
            'suggestions': ['ISP是什么', '车载摄像头推荐什么参数']
        }
    
    def _handle_comparison_query(self, question: str) -> Dict:
        """处理对比查询"""
        return {
            'text': '对比分析需要两个或多个样本，请提供更多图片。',
            'suggestions': ['帮我分析这张', '批量处理哪个好']
        }
    
    def _handle_recommendation_query(self, question: str) -> Dict:
        """处理推荐查询"""
        q = question.lower()
        
        if '夜景' in q or 'night' in q:
            return {
                'text': '''📸 夜景拍摄推荐参数：
• ISO: 800-1600
• 降噪: 1.2-1.5 (NLM)
• 锐化: 1.0
• 启用HDR/LTM
• 使用三脚架''',
                'suggestions': ['人像怎么调', '运动场景参数']
            }
        
        if '人像' in q or 'portrait' in q:
            return {
                'text': '''📸 人像拍摄推荐参数：
• ISO: 200-400
• 降噪: 1.0
• 锐化: 0.8 (避免皮肤瑕疵)
• 饱和度: 0.9
• 适当美颜''',
                'suggestions': ['夜景怎么调', '风景参数']
            }
        
        if '车载' in q or 'adas' in q or '车' in q:
            rec = self.knowledge.get_automotive_recommendations('adas_front')
            text = f"🚗 车载{rec.get('description', '')}\n\n"
            for s in rec.get('general_suggestions', [])[:5]:
                text += f'• {s}\n'
            return {'text': text, 'suggestions': [' DMS怎么调', '环视摄像头参数']}
        
        return {
            'text': '请告诉我是哪种拍摄场景，我可以给出推荐参数。',
            'suggestions': ['夜景怎么调', '人像参数', '车载推荐']
        }
    
    def _handle_general_query(self, question: str) -> Dict:
        """处理通用查询"""
        return {
            'text': '我是ISP图像助手，可以帮你分析图像质量、提供调优建议、了解ISP知识。请上传图片或直接问我问题。',
            'suggestions': ['帮我分析这张图', '什么是ISP pipeline', '夜景怎么调参']
        }
    
    def get_context(self) -> ConversationContext:
        """获取当前上下文"""
        return self.context
    
    def clear_context(self):
        """清除上下文"""
        self.context = ConversationContext()


def create_enhanced_qa() -> EnhancedQAEngine:
    """创建增强版问答引擎"""
    return EnhancedQAEngine()
