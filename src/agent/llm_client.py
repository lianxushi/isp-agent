#!/usr/bin/env python3
"""
LLM Client - 支持MiniMax/OpenAI/Anthropic API调用
"""
import os
import json
import time
from typing import List, Dict, Any, Optional
import requests
from openai import OpenAI

from ..utils.logger import setup_logger

logger = setup_logger('isp-agent.llm')


class LLMAPIError(Exception):
    """LLM API调用异常"""
    pass


class LLMClient:
    """LLM统一客户端 - 本地Python通过API调用远程LLM"""
    
    # Provider配置
    PROVIDER_CONFIGS = {
        'minimax': {
            'base_url': 'https://api.minimax.chat/v1',
            'model_map': {
                'MiniMax-M2.5': 'MiniMax-M2.5',
                'MiniMax-M2.8': 'MiniMax-M2.8',
            }
        },
        'openai': {
            'base_url': 'https://api.openai.com/v1',
            'model_map': {
                'gpt-4o': 'gpt-4o',
                'gpt-4-turbo': 'gpt-4-turbo',
                'gpt-3.5-turbo': 'gpt-3.5-turbo',
            }
        },
        'anthropic': {
            'base_url': 'https://api.anthropic.com',
            'model_map': {
                'claude-sonnet-4-20250514': 'claude-sonnet-4-20250514',
                'claude-opus-4-20250514': 'claude-opus-4-20250514',
            }
        }
    }
    
    def __init__(
        self,
        provider: str = 'minimax',
        api_key: Optional[str] = None,
        model: str = 'MiniMax-M2.5',
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ):
        """
        初始化LLM客户端
        
        Args:
            provider: LLM提供商 (minimax/openai/anthropic)
            api_key: API密钥
            model: 模型名称
            base_url: API基础URL
            temperature: 生成温度
            max_tokens: 最大token数
        """
        self.provider = provider.lower()
        self.api_key = api_key or self._get_api_key()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # 获取base_url
        if base_url:
            self.base_url = base_url
        else:
            self.base_url = self.PROVIDER_CONFIGS.get(self.provider, {}).get('base_url', '')
        
        if not self.api_key:
            raise ValueError(f"未设置 {provider} API密钥")
        
        # 初始化OpenAI客户端 (兼容多provider)
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=60.0,
            max_retries=2
        )
        
        logger.info(f"LLM Client 初始化完成: {provider}/{model}")
    
    def _get_api_key(self) -> str:
        """从环境变量获取API密钥"""
        env_vars = {
            'minimax': 'MINIMAX_API_KEY',
            'openai': 'OPENAI_API_KEY',
            'anthropic': 'ANTHROPIC_API_KEY'
        }
        return os.environ.get(env_vars.get(self.provider, ''), '')
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        发送对话请求
        
        Args:
            messages: 消息列表 [{"role": "user/assistant/system", "content": "..."}]
            temperature: 温度参数
            max_tokens: 最大token数
        
        Returns:
            str: LLM回复内容
        """
        temperature = temperature or self.temperature
        max_tokens = max_tokens or self.max_tokens
        
        # 特殊处理MiniMax
        if self.provider == 'minimax':
            return self._chat_minimax(messages, temperature, max_tokens)
        
        # 标准OpenAI兼容接口
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            return response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"LLM API调用失败: {e}")
            raise LLMAPIError(f"API调用失败: {e}")
    
    def _chat_minimax(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int
    ) -> str:
        """MiniMax API调用"""
        try:
            # 构建请求
            url = f"{self.base_url}/text/chatcompletion_v2"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # 转换消息格式
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content']
        
        except requests.exceptions.RequestException as e:
            logger.error(f"MiniMax API调用失败: {e}")
            raise LLMAPIError(f"MiniMax API调用失败: {e}")
        except KeyError as e:
            logger.error(f"解析MiniMax响应失败: {e}")
            raise LLMAPIError(f"响应解析失败: {e}")
    
    def chat_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        **kwargs
    ) -> Dict[str, Any]:
        """
        工具调用模式
        
        Args:
            messages: 消息列表
            tools: 工具定义列表
        
        Returns:
            dict: 包含回复和工具调用的结果
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                temperature=kwargs.get('temperature', self.temperature),
                **kwargs
            )
            
            message = response.choices[0].message
            
            # 检查是否有工具调用
            if message.tool_calls:
                return {
                    'content': message.content,
                    'tool_calls': [
                        {
                            'name': tc.function.name,
                            'arguments': json.loads(tc.function.arguments)
                        }
                        for tc in message.tool_calls
                    ]
                }
            
            return {'content': message.content, 'tool_calls': []}
        
        except Exception as e:
            logger.error(f"工具调用模式失败: {e}")
            raise LLMAPIError(f"工具调用失败: {e}")
    
    def __repr__(self):
        return f"LLMClient(provider={self.provider}, model={self.model})"


def create_llm_client(config: Dict[str, Any]) -> LLMClient:
    """从配置创建LLM客户端的便捷函数"""
    llm_config = config.get('llm', {})
    return LLMClient(
        provider=llm_config.get('provider', 'minimax'),
        api_key=llm_config.get('api_key'),
        model=llm_config.get('model', 'MiniMax-M2.5'),
        base_url=llm_config.get('base_url'),
        temperature=llm_config.get('temperature', 0.7),
        max_tokens=llm_config.get('max_tokens', 2000)
    )
