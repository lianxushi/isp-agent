#!/usr/bin/env python3
"""
视频分析器 - 本地FFmpeg/OpenCV处理
"""
import os
import json
import traceback
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List
import subprocess

import cv2

from ..utils.logger import setup_logger

logger = setup_logger('isp-agent.video')


class VideoAnalysisError(Exception):
    """视频分析异常"""
    pass


class VideoValidationError(VideoAnalysisError):
    """视频验证异常"""
    pass


class VideoProcessingError(VideoAnalysisError):
    """视频处理异常"""
    pass


@dataclass
class VideoInfo:
    """视频信息数据类"""
    file_path: str
    file_name: str
    format: str
    size_bytes: int
    size_mb: float
    
    # 编码信息
    codec: str
    width: int
    height: int
    fps: float
    duration_seconds: float
    total_frames: int
    bitrate: Optional[str] = None
    
    # 音频信息
    has_audio: bool = False
    audio_codec: Optional[str] = None
    audio_sample_rate: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


class VideoAnalyzer:
    """视频分析器"""
    
    SUPPORTED_FORMATS = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    MAX_SIZE = 100 * 1024 * 1024  # 100MB
    MAX_DURATION = 300  # 5分钟
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.tools_config = self.config.get('tools', {}).get('video', {})
        self.max_size = self.tools_config.get('max_size_mb', 100) * 1024 * 1024
        self.max_duration = self.tools_config.get('max_duration_seconds', 300)
    
    def get_info(self, video_path: str) -> VideoInfo:
        """
        获取视频信息
        
        Args:
            video_path: 视频文件路径
        
        Returns:
            VideoInfo: 视频信息
        
        Raises:
            VideoValidationError: 验证失败
            VideoAnalysisError: 分析失败
        """
        logger.info(f"开始分析视频: {video_path}")
        
        try:
            # 1. 校验文件
            self._validate(video_path)
            
            # 2. 使用OpenCV获取基本信息
            info = self._get_basic_info(video_path)
            
            # 3. 使用FFmpeg获取详细信息
            ff_info = self._get_ffmpeg_info(video_path)
            if ff_info:
                info.bitrate = ff_info.get('bitrate')
                info.has_audio = ff_info.get('has_audio', False)
                info.audio_codec = ff_info.get('audio_codec')
                info.audio_sample_rate = ff_info.get('audio_sample_rate')
            
            logger.info(f"视频分析完成: {info.width}x{info.height} @ {info.fps}fps")
            return info
            
        except VideoValidationError:
            raise
        except VideoAnalysisError:
            raise
        except Exception as e:
            logger.error(f"视频分析未知错误: {traceback.format_exc()}")
            raise VideoAnalysisError(f"视频分析失败: {e}")
    
    def _validate(self, video_path: str) -> None:
        """
        验证视频文件
        
        Args:
            video_path: 视频文件路径
        
        Raises:
            VideoValidationError: 验证失败
        """
        path = Path(video_path)
        
        if not path.exists():
            raise VideoValidationError(f"文件不存在: {video_path}")
        
        if not path.is_file():
            raise VideoValidationError(f"路径不是有效文件: {video_path}")
        
        suffix = path.suffix.lower()
        if suffix not in self.SUPPORTED_FORMATS:
            raise VideoValidationError(
                f"不支持的视频格式: {suffix}，支持的格式: {', '.join(self.SUPPORTED_FORMATS)}"
            )
        
        try:
            file_size = path.stat().st_size
            if file_size > self.max_size:
                raise VideoValidationError(
                    f"文件过大: {file_size / 1024 / 1024:.1f}MB，超过限制: {self.max_size / 1024 / 1024:.1f}MB"
                )
            if file_size == 0:
                raise VideoValidationError(f"文件为空: {video_path}")
        except OSError as e:
            raise VideoValidationError(f"无法获取文件信息: {e}")
    
    def _get_basic_info(self, video_path: str) -> VideoInfo:
        """
        使用OpenCV获取基本信息
        
        Args:
            video_path: 视频文件路径
        
        Returns:
            VideoInfo: 视频信息
        
        Raises:
            VideoProcessingError: 处理失败
        """
        path = Path(video_path)
        
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise VideoProcessingError(f"无法打开视频文件: {video_path}")
            
            try:
                # 获取属性
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps if fps > 0 else 0
                
                if width <= 0 or height <= 0:
                    raise VideoProcessingError(f"无效的视频分辨率: {width}x{height}")
                
                if fps <= 0:
                    raise VideoProcessingError(f"无效的视频帧率: {fps}")
                
                return VideoInfo(
                    file_path=str(path.absolute()),
                    file_name=path.name,
                    format=path.suffix[1:].upper(),
                    size_bytes=path.stat().st_size,
                    size_mb=path.stat().st_size / 1024 / 1024,
                    codec=self._get_codec_name(video_path),
                    width=width,
                    height=height,
                    fps=round(fps, 2),
                    duration_seconds=round(duration, 2),
                    total_frames=frame_count
                )
            finally:
                cap.release()
                
        except VideoProcessingError:
            raise
        except cv2.error as e:
            raise VideoProcessingError(f"OpenCV处理错误: {e}")
        except Exception as e:
            raise VideoProcessingError(f"获取视频信息失败: {e}")
    
    def _get_ffmpeg_info(self, video_path: str) -> Optional[Dict[str, Any]]:
        """使用FFmpeg获取详细信息"""
        try:
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                return None
            
            import json
            data = json.loads(result.stdout)
            
            info = {}
            
            # 视频流信息
            for stream in data.get('streams', []):
                if stream.get('codec_type') == 'video':
                    info['bitrate'] = stream.get('bit_rate')
                elif stream.get('codec_type') == 'audio':
                    info['has_audio'] = True
                    info['audio_codec'] = stream.get('codec_name')
                    info['audio_sample_rate'] = stream.get('sample_rate')
            
            # 格式信息
            format_info = data.get('format', {})
            if not info.get('bitrate'):
                info['bitrate'] = format_info.get('bit_rate')
            
            return info
        
        except Exception as e:
            logger.warning(f"FFprobe获取信息失败: {e}")
            return None
    
    def _get_codec_name(self, video_path: str) -> str:
        """获取编码器名称"""
        try:
            cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                   '-show_entries', 'stream=codec_name', '-of', 'default=noprint_wrappers=1:nokey=1',
                   video_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            return result.stdout.strip() or 'unknown'
        except:
            return 'unknown'
    
    def extract_frames(
        self,
        video_path: str,
        output_dir: str,
        interval: int = 1,
        max_frames: int = 100
    ) -> List[str]:
        """
        抽取视频帧
        
        Args:
            video_path: 视频路径
            output_dir: 输出目录
            interval: 每隔多少帧抽取一帧
            max_frames: 最大抽取帧数
        
        Returns:
            List[str]: 抽取的帧文件路径列表
        """
        logger.info(f"开始抽帧: {video_path}")
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        frame_paths = []
        
        try:
            frame_idx = 0
            saved_idx = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % interval == 0 and saved_idx < max_frames:
                    output_path = os.path.join(output_dir, f"frame_{saved_idx:04d}.jpg")
                    cv2.imwrite(output_path, frame)
                    frame_paths.append(output_path)
                    saved_idx += 1
                
                frame_idx += 1
            
            logger.info(f"抽帧完成: {len(frame_paths)} 帧")
            return frame_paths
        
        finally:
            cap.release()
    
    def extract_keyframes(self, video_path: str, output_dir: str) -> List[str]:
        """抽取关键帧"""
        # 简化实现：每10秒抽一帧作为关键帧
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        try:
            interval = int(fps * 10)  # 每10秒
            return self.extract_frames(video_path, output_dir, interval, 30)
        finally:
            cap.release()


def get_video_info(video_path: str, config: Optional[Dict[str, Any]] = None) -> VideoInfo:
    """便捷函数"""
    analyzer = VideoAnalyzer(config)
    return analyzer.get_info(video_path)
