"""
LLM 客户端 - MiniMax API接入
"""

import os
import json
from typing import List, Dict
import openai
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class LLMClient:
    """LLM统一接入客户端"""
    
    def __init__(self, config: dict):
        self.api_key = config.get('api_key', os.environ.get('MINIMAX_API_KEY', ''))
        self.base_url = config.get('base_url', 'https://api.minimaxi.com/v1')
        self.model = config.get('model', 'abab6.5s-chat')
        self.temperature = config.get('temperature', 0.7)
        
        if not self.api_key:
            logger.warning("未配置LLM API Key，部分功能将不可用")
        
        # 配置OpenAI客户端（兼容MiniMax）
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        logger.info(f"LLM Client初始化完成: {self.model}")
    
    def chat(self, messages: List[Dict], temperature: float = None) -> str:
        """
        发送对话请求
        
        Args:
            messages: [{"role": "user/assistant/system", "content": "..."}]
            temperature: 0-2, 越低越确定性
        
        Returns:
            str: LLM回复内容
        """
        if not self.api_key:
            return "⚠️ LLM未配置，请先设置API Key"
        
        if temperature is None:
            temperature = self.temperature
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature
            )
            content = response.choices[0].message.content
            logger.info(f"LLM响应成功: {len(content)} 字符")
            return content
        except Exception as e:
            logger.error(f"LLM调用失败: {e}")
            return f"❌ LLM调用失败: {str(e)}"
    
    def chat_with_image(self, image_base64: str, prompt: str) -> str:
        """发送图文对话请求（如果模型支持）"""
        # MiniMax暂不支持图文，暂时返回文本回复
        messages = [
            {"role": "user", "content": f"[图片分析]\n{prompt}"}
        ]
        return self.chat(messages)
