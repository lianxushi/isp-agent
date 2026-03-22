"""
图像分析器
"""

import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class AnalysisResult:
    """分析结果数据类"""
    file_path: str
    width: int
    height: int
    format: str
    size_bytes: int
    histogram: Optional[Dict] = None
    dynamic_range: Optional[Dict] = None
    noise_level: Optional[float] = None
    color_analysis: Optional[Dict] = None
    exif: Optional[Dict] = None


class ImageAnalyzer:
    """图像分析器"""
    
    SUPPORTED_FORMATS = ['jpg', 'jpeg', 'png', 'tiff', 'tif', 'dng', 'bmp']
    MAX_SIZE = 50 * 1024 * 1024  # 50MB
    
    def __init__(self):
        logger.info("ImageAnalyzer 初始化完成")
    
    def _validate(self, image_path: str) -> None:
        """校验图像文件"""
        path = Path(image_path)
        
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {image_path}")
        
        if path.suffix[1:].lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(f"不支持的格式: {path.suffix}")
        
        if path.stat().st_size > self.MAX_SIZE:
            raise ValueError(f"文件过大: {path.stat().st_size / 1024 / 1024:.1f}MB")
    
    def analyze(self, image_path: str) -> AnalysisResult:
        """执行完整分析"""
        self._validate(image_path)
        
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        path = Path(image_path)
        
        result = AnalysisResult(
            file_path=image_path,
            width=img.shape[1],
            height=img.shape[0],
            format=path.suffix[1:].lower(),
            size_bytes=path.stat().st_size
        )
        
        # 直方图分析
        result.histogram = self.analyze_histogram(img)
        
        # 动态范围
        result.dynamic_range = self.analyze_dynamic_range(img)
        
        # 噪声分析
        result.noise_level = self.analyze_noise(img)
        
        # 色彩分析
        result.color_analysis = self.analyze_color(img)
        
        logger.info(f"图像分析完成: {result.width}x{result.height}")
        return result
    
    def analyze_histogram(self, img: np.ndarray) -> Dict:
        """分析直方图"""
        result = {}
        for i, color in enumerate(['B', 'G', 'R']):
            hist = cv2.calcHist([img], [i], None, [256], [0, 256])
            hist_data = hist.flatten().tolist()
            # 简化数据，只保留关键统计
            result[color] = {
                'mean': float(np.mean(hist_data)),
                'std': float(np.std(hist_data)),
                'min': int(np.min(hist_data)),
                'max': int(np.max(hist_data))
            }
        return result
    
    def analyze_dynamic_range(self, img: np.ndarray) -> Dict:
        """分析动态范围"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return {
            'min': int(gray.min()),
            'max': int(gray.max()),
            'range': int(gray.max() - gray.min()),
            'mean': float(gray.mean()),
            'std': float(gray.std())
        }
    
    def analyze_noise(self, img: np.ndarray) -> float:
        """估计噪声水平"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 使用局部标准差估计噪声
        h, w = gray.shape
        if h > 20 and w > 20:
            # 分块计算标准差
            blocks = []
            for i in range(0, h-10, 10):
                for j in range(0, w-10, 10):
                    block = gray[i:i+10, j:j+10]
                    blocks.append(np.std(block))
            sigma = np.median(blocks)
        else:
            sigma = np.std(gray)
        return float(sigma)
    
    def analyze_color(self, img: np.ndarray) -> Dict:
        """分析色彩"""
        b, g, r = cv2.split(img)
        
        b_mean, g_mean, r_mean = float(b.mean()), float(g.mean()), float(r.mean())
        
        # 判断白平衡
        max_mean = max(b_mean, g_mean, r_mean)
        min_mean = min(b_mean, g_mean, r_mean)
        
        if max_mean - min_mean < 15:
            wb_status = "OK - 良好"
        elif r_mean > g_mean and r_mean > b_mean:
            wb_status = "偏红"
        elif b_mean > r_mean and b_mean > g_mean:
            wb_status = "偏蓝"
        elif g_mean > r_mean and g_mean > b_mean:
            wb_status = "偏绿"
        else:
            wb_status = "异常"
        
        return {
            'B_mean': b_mean,
            'G_mean': g_mean,
            'R_mean': r_mean,
            'white_balance': wb_status,
            'contrast': float(gray.std()) if 'gray' in locals() else 0
        }
    
    def process(self, image_path: str, method: str, **params) -> str:
        """图像处理"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        output_path = Path(image_path).stem + f"_{method}.jpg"
        
        if method == 'denoise':
            # 降噪处理
            strength = params.get('strength', 10)
            dst = cv2.fastNlMeansDenoisingColored(img, None, strength, strength, 7, 21)
            cv2.imwrite(output_path, dst)
        
        elif method == 'sharpen':
            # 锐化处理
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            dst = cv2.filter2D(img, -1, kernel)
            cv2.imwrite(output_path, dst)
        
        elif method == 'brightness':
            # 亮度调整
            alpha = params.get('alpha', 1.2)  # 1.0-2.0
            beta = params.get('beta', 10)     # 亮度偏移
            dst = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
            cv2.imwrite(output_path, dst)
        
        elif method == 'contrast':
            # 对比度调整
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            dst = cv2.merge([l, a, b])
            dst = cv2.cvtColor(dst, cv2.COLOR_LAB2BGR)
            cv2.imwrite(output_path, dst)
        
        else:
            raise ValueError(f"未知的处理方法: {method}")
        
        logger.info(f"图像处理完成: {method} -> {output_path}")
        return output_path
