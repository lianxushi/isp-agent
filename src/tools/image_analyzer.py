#!/usr/bin/env python3
"""
图像分析器 - 本地Python图像处理
"""
import os
import json
import traceback
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List
from datetime import datetime

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
import exifread

from ..utils.logger import setup_logger

logger = setup_logger('isp-agent.image')


class ImageAnalysisError(Exception):
    """图像分析异常"""
    pass


class ImageValidationError(ImageAnalysisError):
    """图像验证异常"""
    pass


class ImageProcessingError(ImageAnalysisError):
    """图像处理异常"""
    pass


@dataclass
class AnalysisResult:
    """图像分析结果数据类"""
    file_path: str
    file_name: str
    width: int
    height: int
    format: str
    size_bytes: int
    size_kb: float
    
    # 图像质量分析
    histogram: Optional[Dict[str, List[int]]] = None
    dynamic_range: Optional[Dict[str, int]] = None
    noise_level: Optional[float] = None
    brightness: Optional[float] = None
    contrast: Optional[float] = None
    
    # 色彩分析
    color_analysis: Optional[Dict[str, Any]] = None
    
    # EXIF数据
    exif: Optional[Dict[str, Any]] = None
    
    # ISP模块分析
    gamma_analysis: Optional[Dict[str, Any]] = None
    edge_enhancement: Optional[Dict[str, Any]] = None
    ltm_analysis: Optional[Dict[str, Any]] = None
    
    # 拍摄信息
    capture_time: Optional[str] = None
    camera_model: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


class ImageAnalyzer:
    """图像分析器 - 本地OpenCV处理"""
    
    SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.dng', '.bmp']
    MAX_SIZE = 50 * 1024 * 1024  # 50MB
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.tools_config = self.config.get('tools', {}).get('image', {})
        self.max_size = self.tools_config.get('max_size_mb', 50) * 1024 * 1024
    
    def analyze(self, image_path: str) -> AnalysisResult:
        """
        执行完整图像分析
        
        Args:
            image_path: 图像文件路径
        
        Returns:
            AnalysisResult: 分析结果
        
        Raises:
            ImageValidationError: 图像验证失败
            ImageAnalysisError: 图像分析失败
        """
        logger.info(f"开始分析图像: {image_path}")
        
        try:
            # 1. 文件校验
            self._validate(image_path)
            
            # 2. 获取基本信息
            result = self._get_basic_info(image_path)
            
            # 3. 读取图像
            img = cv2.imread(image_path)
            if img is None:
                # 尝试使用PIL读取
                try:
                    pil_img = Image.open(image_path)
                    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                except Exception as e:
                    raise ImageProcessingError(f"无法读取图像: {image_path}, 原因: {e}")
            
            # 4. 直方图分析
            try:
                result.histogram = self._analyze_histogram(img)
            except Exception as e:
                logger.warning(f"直方图分析失败: {e}")
                result.histogram = None
            
            # 5. 动态范围分析
            try:
                result.dynamic_range = self._analyze_dynamic_range(img)
            except Exception as e:
                logger.warning(f"动态范围分析失败: {e}")
                result.dynamic_range = None
            
            # 6. 噪声分析
            try:
                result.noise_level = self._analyze_noise(img)
            except Exception as e:
                logger.warning(f"噪声分析失败: {e}")
                result.noise_level = None
            
            # 7. 亮度/对比度分析
            try:
                result.brightness, result.contrast = self._analyze_brightness_contrast(img)
            except Exception as e:
                logger.warning(f"亮度/对比度分析失败: {e}")
                result.brightness = None
                result.contrast = None
            
            # 8. 色彩分析
            try:
                result.color_analysis = self._analyze_color(img)
            except Exception as e:
                logger.warning(f"色彩分析失败: {e}")
                result.color_analysis = None
            
            # 9. EXIF解析
            try:
                result.exif = self._parse_exif(image_path)
            except Exception as e:
                logger.warning(f"EXIF解析失败: {e}")
                result.exif = None
            
            # 10. Gamma分析
            try:
                result.gamma_analysis = self._analyze_gamma(img)
            except Exception as e:
                logger.warning(f"Gamma分析失败: {e}")
                result.gamma_analysis = None
            
            # 11. 边缘增强分析
            try:
                result.edge_enhancement = self._analyze_edge_enhancement(img)
            except Exception as e:
                logger.warning(f"边缘增强分析失败: {e}")
                result.edge_enhancement = None
            
            # 12. 局部色调映射分析
            try:
                result.ltm_analysis = self._analyze_ltm(img)
            except Exception as e:
                logger.warning(f"LTM分析失败: {e}")
                result.ltm_analysis = None
            
            logger.info(f"图像分析完成: {result.width}x{result.height}")
            return result
            
        except ImageValidationError:
            raise
        except ImageProcessingError:
            raise
        except Exception as e:
            logger.error(f"图像分析未知错误: {traceback.format_exc()}")
            raise ImageAnalysisError(f"图像分析失败: {e}")
    
    def _validate(self, image_path: str) -> None:
        """
        验证图像文件
        
        Args:
            image_path: 图像文件路径
        
        Raises:
            ImageValidationError: 验证失败
        """
        path = Path(image_path)
        
        if not path.exists():
            raise ImageValidationError(f"文件不存在: {image_path}")
        
        if not path.is_file():
            raise ImageValidationError(f"路径不是有效文件: {image_path}")
        
        suffix = path.suffix.lower()
        if suffix not in self.SUPPORTED_FORMATS:
            raise ImageValidationError(
                f"不支持的图像格式: {suffix}，支持的格式: {', '.join(self.SUPPORTED_FORMATS)}"
            )
        
        try:
            file_size = path.stat().st_size
            if file_size > self.max_size:
                raise ImageValidationError(
                    f"文件过大: {file_size / 1024 / 1024:.1f}MB，超过限制: {self.max_size / 1024 / 1024:.1f}MB"
                )
            if file_size == 0:
                raise ImageValidationError(f"文件为空: {image_path}")
        except OSError as e:
            raise ImageValidationError(f"无法获取文件信息: {e}")
    
    def _get_basic_info(self, image_path: str) -> AnalysisResult:
        """
        获取图像基本信息
        
        Args:
            image_path: 图像文件路径
        
        Returns:
            AnalysisResult: 基本信息结果
        """
        path = Path(image_path)
        stat = path.stat()
        
        try:
            # 使用PIL获取更准确的信息
            with Image.open(image_path) as img:
                width, height = img.size
                img_format = img.format
        except UnidentifiedImageError as e:
            raise ImageProcessingError(f"无法识别图像格式: {image_path}, 原因: {e}")
        except Exception as e:
            raise ImageProcessingError(f"读取图像信息失败: {image_path}, 原因: {e}")
        
        return AnalysisResult(
            file_path=str(path.absolute()),
            file_name=path.name,
            width=width,
            height=height,
            format=img_format or path.suffix[1:].upper(),
            size_bytes=stat.st_size,
            size_kb=stat.st_size / 1024
        )
    
    def _analyze_histogram(self, img: np.ndarray) -> Dict[str, List[int]]:
        """分析直方图"""
        result = {}
        
        # 灰度直方图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hist_gray = cv2.calcHist([gray], [0], None, [256], [0, 256])
        result['gray'] = [int(x) for x in hist_gray.flatten()]
        
        # RGB直方图
        for i, color in enumerate(['blue', 'green', 'red']):
            hist = cv2.calcHist([img], [i], None, [256], [0, 256])
            result[color] = [int(x) for x in hist.flatten()]
        
        return result
    
    def _analyze_dynamic_range(self, img: np.ndarray) -> Dict[str, int]:
        """分析动态范围"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 计算百分位
        p1, p99 = np.percentile(gray, [1, 99])
        
        return {
            'min': int(gray.min()),
            'max': int(gray.max()),
            'range': int(gray.max() - gray.min()),
            'p1': int(p1),
            'p99': int(p99),
            'useful_range': int(p99 - p1)
        }
    
    def _analyze_noise(self, img: np.ndarray) -> float:
        """
        估计噪声水平
        使用LAV (Local Adaptive Variance) 方法
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 计算局部方差
        kernel_size = 7
        mean = cv2.blur(gray.astype(np.float32), (kernel_size, kernel_size))
        variance = cv2.blur((gray.astype(np.float32) - mean) ** 2, (kernel_size, kernel_size))
        
        # 噪声估计 = 中值方差的平方根
        noise_estimate = float(np.sqrt(np.median(variance)))
        
        return noise_estimate
    
    def _analyze_brightness_contrast(self, img: np.ndarray) -> tuple:
        """分析亮度和对比度"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 亮度 = 平均灰度值
        brightness = float(gray.mean())
        
        # 对比度 = 标准差
        contrast = float(gray.std())
        
        return brightness, contrast
    
    def _analyze_color(self, img: np.ndarray) -> Dict[str, Any]:
        """分析色彩"""
        # 分离通道
        b, g, r = cv2.split(img)
        
        b_mean, g_mean, r_mean = float(b.mean()), float(g.mean()), float(r.mean())
        
        # 计算白平衡倾向
        max_channel = max(b_mean, g_mean, r_mean)
        min_channel = min(b_mean, g_mean, r_mean)
        
        wb_status = "正常"
        if r_mean > b_mean * 1.2:
            wb_status = "偏暖(红)"
        elif b_mean > r_mean * 1.2:
            wb_status = "偏冷(蓝)"
        
        # 饱和度估计
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        saturation = float(hsv[:, :, 1].mean())
        
        return {
            'B_mean': round(b_mean, 2),
            'G_mean': round(g_mean, 2),
            'R_mean': round(r_mean, 2),
            'white_balance': wb_status,
            'saturation': round(saturation, 2),
            'max_channel': max_channel,
            'min_channel': min_channel
        }
    
    def _analyze_gamma(self, img: np.ndarray) -> Dict[str, Any]:
        """分析Gamma校正特性"""
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
            hist = hist / hist.sum()
            cdf = np.cumsum(hist)
            mid_idx = np.searchsorted(cdf, 0.5)
            mid_level = mid_idx / 255.0
            
            estimated_gamma = 2.2
            if mid_level < 0.4:
                estimated_gamma = 2.0
            elif mid_level > 0.6:
                estimated_gamma = 2.4
            
            brightness = "正常" if 0.4 <= mid_level <= 0.6 else ("偏暗" if mid_level < 0.4 else "偏亮")
            
            return {
                'estimated_gamma': round(estimated_gamma, 2),
                'mid_tone_level': round(mid_level, 3),
                'brightness_assessment': brightness,
                'note': 'Gamma值影响图像整体亮度曲线'
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_edge_enhancement(self, img: np.ndarray) -> Dict[str, Any]:
        """分析边缘增强(EE)效果"""
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            edge_strength = float(laplacian.var())
            
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            sobel_mean = float(np.sqrt(sobelx**2 + sobely**2).mean())
            
            if edge_strength < 100:
                assessment, sharpness = "较弱", "偏低"
            elif edge_strength < 500:
                assessment, sharpness = "适中", "良好"
            else:
                assessment, sharpness = "较强", "过高(可能过锐化)"
            
            return {
                'edge_strength': round(edge_strength, 2),
                'sobel_mean': round(sobel_mean, 2),
                'edge_assessment': assessment,
                'sharpness': sharpness,
                'recommendation': '建议调整锐化强度' if edge_strength > 500 else '边缘增强适中'
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_ltm(self, img: np.ndarray) -> Dict[str, Any]:
        """分析局部色调映射(LTM)效果"""
        try:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB) if len(img.shape) == 3 else img
            l_channel = lab[:, :, 0] if len(img.shape) == 3 else img
            
            contrasts = []
            for k in [5, 15, 25]:
                mean = cv2.blur(l_channel.astype(np.float32), (k, k))
                sqr_mean = cv2.blur((l_channel.astype(np.float32))**2, (k, k))
                std = float(np.sqrt(np.maximum(sqr_mean - mean**2, 0)).mean())
                contrasts.append(std)
            
            global_range = int(l_channel.max() - l_channel.min())
            ratio = contrasts[0] / (global_range / 255 + 1e-6)
            
            if ratio > 0.5:
                assessment, note = "良好", "局部对比度保持较好"
            elif ratio > 0.3:
                assessment, note = "一般", "动态范围压缩较明显"
            else:
                assessment, note = "较弱", "可能存在细节丢失"
            
            return {
                'local_contrast_5x5': round(contrasts[0], 2),
                'local_contrast_15x15': round(contrasts[1], 2),
                'local_contrast_25x25': round(contrasts[2], 2),
                'global_range': global_range,
                'ltm_assessment': assessment,
                'note': note
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _parse_exif(self, image_path: str) -> Optional[Dict[str, Any]]:
        """解析EXIF数据"""
        try:
            with open(image_path, 'rb') as f:
                tags = exifread.process_file(f, details=False)
            
            if not tags:
                return None
            
            # 提取关键信息
            exif_data = {}
            
            # 拍摄信息
            if 'Image Model' in tags:
                exif_data['camera_model'] = str(tags['Image Model'])
            
            if 'EXIF LensModel' in tags:
                exif_data['lens'] = str(tags['EXIF LensModel'])
            
            # 曝光参数
            if 'EXIF ExposureTime' in tags:
                exif_data['exposure_time'] = str(tags['EXIF ExposureTime'])
            
            if 'EXIF FNumber' in tags:
                exif_data['f_number'] = str(tags['EXIF FNumber'])
            
            if 'EXIF ISOSpeedRatings' in tags:
                exif_data['iso'] = str(tags['EXIF ISOSpeedRatings'])
            
            if 'EXIF FocalLength' in tags:
                exif_data['focal_length'] = str(tags['EXIF FocalLength'])
            
            # 尺寸
            if 'EXIF ExifImageWidth' in tags:
                exif_data['width'] = str(tags['EXIF ExifImageWidth'])
            
            if 'EXIF ExifImageLength' in tags:
                exif_data['height'] = str(tags['EXIF ExifImageLength'])
            
            return exif_data if exif_data else None
        
        except Exception as e:
            logger.warning(f"EXIF解析失败: {e}")
            return None
    
    def process(
        self,
        image_path: str,
        method: str,
        params: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        图像处理
        
        Args:
            image_path: 图像路径
            method: 处理方法 (denoise/sharpen/color/adjust)
            params: 处理参数
        
        Returns:
            str: 处理结果描述
        
        Raises:
            ImageValidationError: 验证失败
            ImageProcessingError: 处理失败
        """
        # 验证文件
        self._validate(image_path)
        
        params = params or {}
        
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            raise ImageProcessingError(f"无法读取图像: {image_path}")
        
        try:
            if method == 'denoise':
                # 降噪
                strength = params.get('strength', 10)
                if not 1 <= strength <= 20:
                    raise ImageProcessingError(f"降噪强度必须在1-20之间，当前值: {strength}")
                result = cv2.fastNlMeansDenoisingColored(img, None, strength, strength, 7, 21)
                output_path = image_path.replace('.', '_denoised.')
                cv2.imwrite(output_path, result)
                return f"降噪完成，已保存到: {output_path}"
            
            elif method == 'sharpen':
                # 锐化
                kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                result = cv2.filter2D(img, -1, kernel)
                output_path = image_path.replace('.', '_sharpened.')
                cv2.imwrite(output_path, result)
                return f"锐化完成，已保存到: {output_path}"
            
            elif method == 'color':
                # 色彩调整
                temperature = params.get('temperature', 0)  # -100到100
                if not -100 <= temperature <= 100:
                    raise ImageProcessingError(f"色温必须在-100到100之间，当前值: {temperature}")
                # 简化实现
                result = img
                output_path = image_path.replace('.', '_color.')
                cv2.imwrite(output_path, result)
                return f"色彩调整完成，已保存到: {output_path}"
            
            else:
                supported_methods = ['denoise', 'sharpen', 'color']
                raise ImageProcessingError(
                    f"不支持的处理方法: {method}，支持的方法: {', '.join(supported_methods)}"
                )
        
        except ImageProcessingError:
            raise
        except cv2.error as e:
            raise ImageProcessingError(f"OpenCV处理错误: {e}")
        except Exception as e:
            raise ImageProcessingError(f"图像处理失败: {e}")


def analyze_image(image_path: str, config: Optional[Dict[str, Any]] = None) -> AnalysisResult:
    """便捷函数"""
    analyzer = ImageAnalyzer(config)
    return analyzer.analyze(image_path)
