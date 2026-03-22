#!/usr/bin/env python3
"""
ISP-Agent 主入口
基于LLM的ISP图像/视频处理助手 (本地Python + API大脑)
"""
import sys
import argparse
from pathlib import Path
from typing import Optional

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.agent.llm_client import create_llm_client, LLMClient, LLMAPIError
from src.agent.qa_engine import QAEngine
from src.tools.image_analyzer import ImageAnalyzer, ImageValidationError, ImageProcessingError, ImageAnalysisError
from src.tools.video_analyzer import VideoAnalyzer, VideoValidationError, VideoProcessingError

logger = setup_logger('isp-agent.main')


class ISPAgent:
    """ISP Agent 主类"""
    
    def __init__(self, config_path: Optional[str] = None):
        # 加载配置
        self.config = load_config(config_path)
        
        # 初始化LLM客户端 (API调用)
        self.llm_client: LLMClient = create_llm_client(self.config.config)
        
        # 初始化分析器 (本地处理)
        self.image_analyzer = ImageAnalyzer(self.config.config)
        self.video_analyzer = VideoAnalyzer(self.config.config)
        
        # 初始化QA引擎
        self.qa_engine = QAEngine(self.llm_client)
        
        logger.info("=" * 50)
        logger.info("ISP-Agent 初始化完成")
        logger.info(f"LLM: {self.llm_client.provider}/{self.llm_client.model}")
        logger.info("=" * 50)
    
    def analyze_image(self, image_path: str) -> dict:
        """
        分析图像
        
        Args:
            image_path: 图像文件路径
        
        Returns:
            dict: 分析结果
        
        Raises:
            ImageValidationError: 图像验证失败
            ImageProcessingError: 图像处理失败
            ImageAnalysisError: 图像分析失败
        """
        logger.info(f"开始分析图像: {image_path}")
        
        try:
            result = self.image_analyzer.analyze(image_path)
            
            # 转换为字典
            result_dict = result.to_dict()
            
            logger.info(f"图像分析完成: {result.width}x{result.height}")
            return result_dict
            
        except (ImageValidationError, ImageProcessingError, ImageAnalysisError) as e:
            logger.error(f"图像分析失败: {e}")
            raise
        except Exception as e:
            logger.error(f"图像分析未知错误: {e}")
            raise ImageAnalysisError(f"图像分析失败: {e}")
    
    def analyze_video(self, video_path: str) -> dict:
        """
        分析视频
        
        Args:
            video_path: 视频文件路径
        
        Returns:
            dict: 分析结果
        
        Raises:
            VideoValidationError: 视频验证失败
            VideoProcessingError: 视频处理失败
        """
        logger.info(f"开始分析视频: {video_path}")
        
        try:
            result = self.video_analyzer.get_info(video_path)
            
            # 转换为字典
            result_dict = result.to_dict()
            
            logger.info(f"视频分析完成: {result.width}x{result.height}")
            return result_dict
            
        except (VideoValidationError, VideoProcessingError) as e:
            logger.error(f"视频分析失败: {e}")
            raise
        except Exception as e:
            logger.error(f"视频分析未知错误: {e}")
            raise
    
    def chat(self, user_input: str, context: Optional[dict] = None) -> str:
        """
        智能问答
        
        Args:
            user_input: 用户输入
            context: 上下文（如分析结果）
        
        Returns:
            str: LLM回复
        """
        logger.info(f"收到用户输入: {user_input[:80]}...")
        
        response = self.qa_engine.chat(user_input, context)
        
        logger.info(f"LLM回复: {response[:80]}...")
        return response
    
    def process_image(
        self,
        image_path: str,
        method: str,
        **params
    ) -> str:
        """
        图像处理
        
        Args:
            image_path: 图像路径
            method: 处理方法
            **params: 处理参数
        
        Returns:
            str: 处理结果
        
        Raises:
            ImageValidationError: 验证失败
            ImageProcessingError: 处理失败
        """
        logger.info(f"开始处理图像: {method}")
        
        try:
            result = self.image_analyzer.process(image_path, method, params)
            
            logger.info(f"图像处理完成: {result}")
            return result
            
        except (ImageValidationError, ImageProcessingError) as e:
            logger.error(f"图像处理失败: {e}")
            raise
        except Exception as e:
            logger.error(f"图像处理未知错误: {e}")
            raise ImageProcessingError(f"图像处理失败: {e}")
    
    def interactive_chat(self) -> None:
        """交互式对话"""
        print("\n" + "=" * 50)
        print("  ISP-Agent 对话模式")
        print("  输入 'quit' 或 'exit' 退出")
        print("  输入 'clear' 清空对话历史")
        print("=" * 50 + "\n")
        
        while True:
            try:
                user_input = input("你> ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("再见！👋")
                    break
                
                if user_input.lower() == 'clear':
                    self.qa_engine.clear_history()
                    print("对话历史已清空\n")
                    continue
                
                # 调用LLM
                response = self.chat(user_input)
                
                print(f"\nAI> {response}\n")
            
            except KeyboardInterrupt:
                print("\n\n再见！👋")
                break
            except Exception as e:
                logger.error(f"对话出错: {e}")
                print(f"错误: {e}\n")


def print_image_result(result: dict) -> None:
    """打印图像分析结果"""
    print("\n" + "=" * 50)
    print("  图像分析结果")
    print("=" * 50)
    
    print(f"\n📷 基本信息:")
    print(f"   文件名: {result.get('file_name', '')}")
    print(f"   分辨率: {result.get('width')} x {result.get('height')}")
    print(f"   格式: {result.get('format')}")
    print(f"   大小: {result.get('size_kb'):.1f} KB")
    
    if result.get('dynamic_range'):
        dr = result['dynamic_range']
        print(f"\n📊 动态范围:")
        print(f"   有效范围: {dr.get('useful_range', '')}")
        print(f"   总范围: {dr.get('range', '')}")
    
    if result.get('noise_level') is not None:
        print(f"\n🔊 噪声水平: {result['noise_level']:.2f}")
    
    if result.get('brightness') is not None:
        print(f"\n💡 亮度: {result['brightness']:.1f}")
    
    if result.get('contrast') is not None:
        print(f"   对比度: {result['contrast']:.1f}")
    
    if result.get('color_analysis'):
        ca = result['color_analysis']
        print(f"\n🎨 色彩分析:")
        print(f"   白平衡: {ca.get('white_balance', '')}")
        print(f"   饱和度: {ca.get('saturation', '')}")
        print(f"   B/G/R均值: {ca.get('B_mean')}/{ca.get('G_mean')}/{ca.get('R_mean')}")
    
    if result.get('exif'):
        exif = result['exif']
        print(f"\n📝 EXIF信息:")
        for k, v in exif.items():
            print(f"   {k}: {v}")
    
    print()


def print_video_result(result: dict) -> None:
    """打印视频分析结果"""
    print("\n" + "=" * 50)
    print("  视频分析结果")
    print("=" * 50)
    
    print(f"\n🎬 基本信息:")
    print(f"   文件名: {result.get('file_name', '')}")
    print(f"   格式: {result.get('format')}")
    print(f"   大小: {result.get('size_mb'):.1f} MB")
    
    print(f"\n📹 视频参数:")
    print(f"   分辨率: {result.get('width')} x {result.get('height')}")
    print(f"   帧率: {result.get('fps')} fps")
    print(f"   时长: {result.get('duration_seconds'):.1f} 秒")
    print(f"   总帧数: {result.get('total_frames')}")
    print(f"   编码: {result.get('codec')}")
    
    if result.get('bitrate'):
        print(f"   码率: {result['bitrate']}")
    
    if result.get('has_audio'):
        print(f"\n🔊 音频:")
        print(f"   编码: {result.get('audio_codec')}")
        print(f"   采样率: {result.get('audio_sample_rate')}")
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description='ISP-Agent - 基于LLM的ISP图像/视频处理助手'
    )
    
    # 子命令
    subparsers = parser.add_subparsers(dest='command', help='命令')
    
    # analyze 命令
    analyze_parser = subparsers.add_parser('analyze', help='分析图像或视频')
    analyze_parser.add_argument('file', help='文件路径')
    
    # chat 命令
    chat_parser = subparsers.add_parser('chat', help='对话模式')
    chat_parser.add_argument('--text', '-t', help='直接输入对话内容')
    
    # process 命令
    process_parser = subparsers.add_parser('process', help='图像处理')
    process_parser.add_argument('file', help='图像文件路径')
    process_parser.add_argument('method', help='处理方法 (denoise/sharpen/color)')
    
    # 配置
    parser.add_argument(
        '--config', '-c',
        help='配置文件路径',
        default=None
    )
    
    args = parser.parse_args()
    
    # 如果没有子命令，启动交互式对话
    if args.command is None:
        # 创建Agent
        agent = ISPAgent(args.config)
        
        # 检查是否有命令行参数
        if len(sys.argv) > 1:
            parser.print_help()
            return
        
        # 启动交互式对话
        agent.interactive_chat()
        return
    
    # 创建Agent实例
    agent = ISPAgent(args.config)
    
    if args.command == 'analyze':
        # 分析文件
        file_path = Path(args.file)
        
        if not file_path.exists():
            print(f"❌ 错误: 文件不存在: {args.file}")
            sys.exit(1)
        
        # 根据文件类型选择分析器
        suffix = file_path.suffix.lower()
        
        if suffix in ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.dng', '.bmp']:
            try:
                result = agent.analyze_image(str(file_path))
                print_image_result(result)
                
                # 可选：询问是否需要调优建议
                print("是否需要调优建议？(y/n)")
                if input("> ").strip().lower() == 'y':
                    print("\n正在生成调优建议...\n")
                    suggestion = agent.qa_engine.generate_suggestion(result)
                    print(f"💡 调优建议:\n{suggestion}\n")
            except ImageValidationError as e:
                print(f"❌ 图像验证失败: {e}")
                sys.exit(1)
            except ImageProcessingError as e:
                print(f"❌ 图像处理失败: {e}")
                sys.exit(1)
            except ImageAnalysisError as e:
                print(f"❌ 图像分析失败: {e}")
                sys.exit(1)
        
        elif suffix in ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']:
            try:
                result = agent.analyze_video(str(file_path))
                print_video_result(result)
            except VideoValidationError as e:
                print(f"❌ 视频验证失败: {e}")
                sys.exit(1)
            except VideoProcessingError as e:
                print(f"❌ 视频处理失败: {e}")
                sys.exit(1)
        
        else:
            print(f"❌ 错误: 不支持的文件格式: {suffix}")
            sys.exit(1)
    
    elif args.command == 'chat':
        # 对话模式
        if args.text:
            response = agent.chat(args.text)
            print(f"\nAI> {response}\n")
        else:
            agent.interactive_chat()
    
    elif args.command == 'process':
        # 图像处理
        if not Path(args.file).exists():
            print(f"❌ 错误: 文件不存在: {args.file}")
            sys.exit(1)
        
        try:
            result = agent.process_image(args.file, args.method)
            print(f"\n✅ {result}\n")
        except ImageValidationError as e:
            print(f"❌ 图像验证失败: {e}")
            sys.exit(1)
        except ImageProcessingError as e:
            print(f"❌ 图像处理失败: {e}")
            sys.exit(1)


if __name__ == '__main__':
    main()
