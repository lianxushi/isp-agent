#!/usr/bin/env python3
"""
配置管理模块
"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class Config:
    """配置管理类"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """加载配置文件"""
        if config_path is None:
            # 默认配置文件 - 项目根目录的config/
            config_dir = Path(__file__).parent.parent.parent / 'config'
            config_path = config_dir / 'default.yaml'
        
        if not Path(config_path).exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 处理环境变量
        config = self._resolve_env_vars(config)
        
        return config
    
    def _resolve_env_vars(self, config: Any) -> Any:
        """递归解析环境变量"""
        if isinstance(config, dict):
            return {k: self._resolve_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._resolve_env_vars(item) for item in config]
        elif isinstance(config, str):
            # 检查是否是需要替换的环境变量 ${VAR}
            if config.startswith('${') and config.endswith('}'):
                var_name = config[2:-1]
                return os.environ.get(var_name, '')
            return config
        return config
    
    @property
    def llm_config(self) -> Dict[str, Any]:
        """获取LLM配置"""
        return self.config.get('llm', {})
    
    @property
    def tools_config(self) -> Dict[str, Any]:
        """获取工具配置"""
        return self.config.get('tools', {})
    
    @property
    def logging_config(self) -> Dict[str, Any]:
        """获取日志配置"""
        return self.config.get('logging', {})


def load_config(config_path: Optional[str] = None) -> Config:
    """加载配置的便捷函数"""
    return Config(config_path)
