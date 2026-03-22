#!/usr/bin/env python3
"""
配置管理单元测试
"""
import os
import sys
import unittest
from pathlib import Path
import tempfile

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import Config, load_config


class TestConfig(unittest.TestCase):
    """配置管理测试类"""
    
    def setUp(self):
        """测试初始化"""
        self.test_dir = Path(__file__).parent.parent
        self.config_file = self.test_dir / 'config' / 'default.yaml'
    
    def test_01_load_default_config(self):
        """测试加载默认配置"""
        if not self.config_file.exists():
            self.skipTest("配置文件不存在")
        
        config = load_config()
        
        self.assertIsNotNone(config)
        self.assertIsNotNone(config.config)
    
    def test_02_load_custom_config(self):
        """测试加载自定义配置"""
        if not self.config_file.exists():
            self.skipTest("配置文件不存在")
        
        config = load_config(str(self.config_file))
        
        self.assertIsNotNone(config.config)
        self.assertIn('llm', config.config)
    
    def test_03_load_nonexistent_config(self):
        """测试加载不存在的配置"""
        with self.assertRaises(FileNotFoundError) as context:
            load_config('/nonexistent/config.yaml')
        
        self.assertIn('配置文件不存在', str(context.exception))
    
    def test_04_llm_config(self):
        """测试LLM配置访问"""
        if not self.config_file.exists():
            self.skipTest("配置文件不存在")
        
        config = load_config(str(self.config_file))
        
        llm_config = config.llm_config
        
        self.assertIsInstance(llm_config, dict)
    
    def test_05_tools_config(self):
        """测试工具配置访问"""
        if not self.config_file.exists():
            self.skipTest("配置文件不存在")
        
        config = load_config(str(self.config_file))
        
        tools_config = config.tools_config
        
        self.assertIsInstance(tools_config, dict)
    
    def test_06_logging_config(self):
        """测试日志配置访问"""
        if not self.config_file.exists():
            self.skipTest("配置文件不存在")
        
        config = load_config(str(self.config_file))
        
        logging_config = config.logging_config
        
        self.assertIsInstance(logging_config, dict)
    
    def test_07_env_var_resolution(self):
        """测试环境变量解析"""
        # 创建临时配置文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
llm:
  api_key: "${TEST_API_KEY}"
  provider: "minimax"
""")
            temp_config_path = f.name
        
        try:
            # 设置环境变量
            os.environ['TEST_API_KEY'] = 'test-key-123'
            
            config = load_config(temp_config_path)
            
            # 验证环境变量被正确解析
            self.assertEqual(config.config['llm']['api_key'], 'test-key-123')
        
        finally:
            # 清理
            os.unlink(temp_config_path)
            if 'TEST_API_KEY' in os.environ:
                del os.environ['TEST_API_KEY']
    
    def test_08_env_var_not_set(self):
        """测试未设置的环境变量"""
        # 创建临时配置文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
llm:
  api_key: "${NONEXISTENT_VAR_12345}"
  provider: "minimax"
""")
            temp_config_path = f.name
        
        try:
            # 确保环境变量不存在
            if 'NONEXISTENT_VAR_12345' in os.environ:
                del os.environ['NONEXISTENT_VAR_12345']
            
            config = load_config(temp_config_path)
            
            # 验证未设置的环境变量被解析为空字符串
            self.assertEqual(config.config['llm']['api_key'], '')
        
        finally:
            # 清理
            os.unlink(temp_config_path)


def run_tests():
    """运行所有测试"""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == '__main__':
    run_tests()
