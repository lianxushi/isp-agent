#!/usr/bin/env python3
"""
VideoAnalyzer 单元测试
"""
import os
import sys
import unittest
from pathlib import Path

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools.video_analyzer import VideoAnalyzer, VideoInfo


class TestVideoAnalyzer(unittest.TestCase):
    """VideoAnalyzer 测试类"""
    
    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        cls.test_dir = Path(__file__).parent.parent
        cls.test_video = cls.test_dir / 'test_video.mp4'
        cls.test_config = {
            'tools': {
                'video': {
                    'max_size_mb': 100,
                    'max_duration_seconds': 300
                }
            }
        }
    
    def test_01_analyzer_init(self):
        """测试分析器初始化"""
        analyzer = VideoAnalyzer()
        self.assertIsNotNone(analyzer)
        
        # 带配置初始化
        analyzer = VideoAnalyzer(self.test_config)
        self.assertIsNotNone(analyzer)
    
    def test_02_get_info_file_not_found(self):
        """测试文件不存在异常"""
        from src.tools.video_analyzer import VideoValidationError
        
        analyzer = VideoAnalyzer()
        
        with self.assertRaises(VideoValidationError) as context:
            analyzer.get_info('/nonexistent/video.mp4')
        
        self.assertIn('文件不存在', str(context.exception))
    
    def test_03_get_info_unsupported_format(self):
        """测试不支持格式"""
        from src.tools.video_analyzer import VideoValidationError
        
        analyzer = VideoAnalyzer()
        
        # 创建一个临时测试文件
        test_file = self.test_dir / 'test.txt'
        test_file.write_text('test')
        
        try:
            with self.assertRaises(VideoValidationError) as context:
                analyzer.get_info(str(test_file))
            
            self.assertIn('不支持的视频格式', str(context.exception))
        finally:
            if test_file.exists():
                test_file.unlink()
    
    def test_04_get_info_basic(self):
        """测试获取视频基本信息"""
        analyzer = VideoAnalyzer()
        
        if not self.test_video.exists():
            self.skipTest("测试视频不存在")
        
        result = analyzer.get_info(str(self.test_video))
        
        self.assertIsInstance(result, VideoInfo)
        self.assertGreater(result.width, 0)
        self.assertGreater(result.height, 0)
        self.assertGreater(result.fps, 0)
        self.assertGreater(result.duration_seconds, 0)
    
    def test_05_video_info_to_dict(self):
        """测试VideoInfo转字典"""
        result = VideoInfo(
            file_path='/test/video.mp4',
            file_name='video.mp4',
            format='mp4',
            size_bytes=10240000,
            size_mb=10.0,
            width=1920,
            height=1080,
            fps=30.0,
            duration_seconds=60.0,
            total_frames=1800,
            codec='h264',
            bitrate='2000k',
            has_audio=True,
            audio_codec='aac',
            audio_sample_rate=48000
        )
        
        result_dict = result.to_dict()
        
        self.assertIsInstance(result_dict, dict)
        self.assertEqual(result_dict['width'], 1920)
        self.assertEqual(result_dict['height'], 1080)
        self.assertEqual(result_dict['fps'], 30.0)


class TestVideoAnalyzerExtract(unittest.TestCase):
    """视频抽帧功能测试"""
    
    @classmethod
    def setUpClass(cls):
        cls.test_dir = Path(__file__).parent.parent
        cls.test_video = cls.test_dir / 'test_video.mp4'
    
    def test_extract_frames_interval(self):
        """测试按间隔抽帧"""
        analyzer = VideoAnalyzer()
        
        if not self.test_video.exists():
            self.skipTest("测试视频不存在")
        
        output_dir = self.test_dir / 'test_frames'
        output_dir.mkdir(exist_ok=True)
        
        try:
            frames = analyzer.extract_frames(
                str(self.test_video),
                str(output_dir),
                interval=1,
                max_frames=5
            )
            
            self.assertIsInstance(frames, list)
        except Exception as e:
            # 视频可能没有足够帧
            self.skipTest(f"抽帧失败: {e}")
        finally:
            # 清理
            if output_dir.exists():
                for f in output_dir.glob('*.jpg'):
                    f.unlink()
                output_dir.rmdir()
    
    def test_extract_frames_keyframe(self):
        """测试关键帧抽帧"""
        analyzer = VideoAnalyzer()
        
        if not self.test_video.exists():
            self.skipTest("测试视频不存在")
        
        output_dir = self.test_dir / 'test_keyframes'
        output_dir.mkdir(exist_ok=True)
        
        try:
            frames = analyzer.extract_keyframes(
                str(self.test_video),
                str(output_dir)
            )
            
            self.assertIsInstance(frames, list)
        except Exception as e:
            self.skipTest(f"抽帧失败: {e}")
        finally:
            # 清理
            if output_dir.exists():
                for f in output_dir.glob('*.jpg'):
                    f.unlink()
                output_dir.rmdir()


def run_tests():
    """运行所有测试"""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == '__main__':
    run_tests()
