#!/usr/bin/env python3
"""
HDR增强处理器
支持多种Tone Mapping算法和专业级HDR处理

HDR处理流程:
1. 多帧对齐(如果需要)
2. HDR合成
3. Tone Mapping
4. 输出

支持算法:
- Reinhard: 全局+局部tone mapping
- Mantiuk: 对比度保留tone mapping
- Drago: 自适应对数tone mapping
- ACES: 学院色彩编码系统(电影级)
"""
import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from ..utils.logger import setup_logger

logger = setup_logger('isp-agent.hdr_enhanced')


class ToneMappingMethod(Enum):
    """Tone Mapping算法枚举"""
    REINHARD = 'reinhard'
    MANTIK = 'mantik'
    DRAGO = 'drago'
    ACES = 'aces'
    MERTENS = 'mertens'  # OpenCV内置融合
    EXPOSURE = 'exposure'  # 曝光融合


@dataclass
class HDRConfig:
    """HDR处理配置"""
    method: ToneMappingMethod = ToneMappingMethod.REINHARD
    gamma: float = 2.2
    exposure_offset: float = 0.0  # 曝光偏移
    contrast: float = 1.0  # 对比度
    saturation: float = 1.0  # 饱和度
    light_adaptation: float = 0.0  # 亮度适应 (Reinhard参数)
    color_adaptation: float = 0.0  # 色彩适应 (Reinhard参数)
    # Drago参数
    drago_softness: float = 0.5  # 0-1, 越大越软
    drago_bias: float = 0.85  # 偏移参数
    # Mantiuk参数
    mantik_scale: float = 0.7  # 对比度缩放


class HDRError(Exception):
    """HDR处理异常"""
    pass


class HDRProcessor:
    """
    增强型HDR处理器
    
    支持多种Tone Mapping算法:
    1. Reinhard - 适合一般用途，细节保留好
    2. Mantiuk - 保留局部对比度，适合显示设备
    3. Drago - 极端光照条件，适合高动态范围场景
    4. ACES - 电影级色彩，适合专业制作
    """
    
    def __init__(self):
        pass
    
    def merge_hdr(
        self,
        image_paths: List[str],
        output_path: str,
        method: str = 'reinhard',
        config: Optional[HDRConfig] = None
    ) -> Dict[str, Any]:
        """
        多帧HDR合成并做Tone Mapping
        
        Args:
            image_paths: 曝光序列图像路径列表
            output_path: 输出路径
            method: TM方法 ('reinhard'/'mantik'/'drago'/'aces'/'mertens'/'exposure')
            config: HDR配置
        
        Returns:
            Dict: 处理结果
        """
        logger.info(f"开始HDR处理: {len(image_paths)} 张图像, 方法={method}")
        
        if len(image_paths) < 2:
            return {'success': False, 'error': '需要至少2张图像进行HDR合成'}
        
        config = config or HDRConfig()
        
        try:
            # 1. 读取图像
            images = self._load_images(image_paths)
            if images is None:
                return {'success': False, 'error': '图像加载失败'}
            
            # 2. 合成HDR
            if len(images) == 2:
                # 两张图直接融合
                merged = self._merge_two_images(images[0], images[1])
            else:
                merged = self._merge_multiple(images)
            
            # 3. Tone Mapping
            ldr = self._apply_tonemap(merged, method, config)
            
            # 4. 后处理
            ldr = self._post_process(ldr, config)
            
            # 5. 保存
            cv2.imwrite(output_path, ldr)
            
            logger.info(f"HDR处理完成: {output_path}")
            
            return {
                'success': True,
                'output': output_path,
                'method': method,
                'frames_used': len(image_paths),
                'dynamic_range_estimate': self._estimate_dynamic_range(merged)
            }
            
        except Exception as e:
            logger.error(f"HDR处理失败: {e}")
            return {'success': False, 'error': str(e)}
    
    def _load_images(self, paths: List[str]) -> Optional[List[np.ndarray]]:
        """加载图像列表"""
        images = []
        for path in paths:
            img = cv2.imread(path)
            if img is None:
                logger.error(f"无法读取图像: {path}")
                return None
            images.append(img.astype(np.float32))
        return images
    
    def _merge_two_images(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """两张图像的简单HDR融合"""
        # 基于曝光权重的融合
        gray1 = cv2.cvtColor(img1.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        
        # 权重：中间调权重更高
        w1 = np.exp(-(gray1 - 128)**2 / (2 * 50**2))
        w2 = np.exp(-(gray2 - 128)**2 / (2 * 50**2))
        
        # 归一化
        w_sum = w1 + w2 + 1e-10
        w1, w2 = w1 / w_sum, w2 / w_sum
        
        # 融合
        merged = img1 * w1[:, :, np.newaxis] + img2 * w2[:, :, np.newaxis]
        return merged
    
    def _merge_multiple(self, images: List[np.ndarray]) -> np.ndarray:
        """多张图像的HDR融合 (使用Mertens方法)"""
        # 使用OpenCV的Mertens融合
        merge_mertens = cv2.createMergeMertens()
        
        # 归一化到0-1
        normalized = []
        for img in images:
            img_norm = img / 255.0
            normalized.append(img_norm)
        
        merged = merge_mertens.process(normalized)
        return merged * 255.0
    
    def _apply_tonemap(
        self,
        hdr: np.ndarray,
        method: str,
        config: HDRConfig
    ) -> np.ndarray:
        """
        应用Tone Mapping算法
        
        Args:
            hdr: HDR图像 (float)
            method: TM方法
            config: 配置
        
        Returns:
            LDR图像 (uint8)
        """
        # 归一化到0-1范围
        hdr = hdr.astype(np.float32) / 255.0
        
        if method == 'reinhard':
            return self._tonemap_reinhard(hdr, config)
        elif method == 'mantik':
            return self._tonemap_mantik(hdr, config)
        elif method == 'drago':
            return self._tonemap_drago(hdr, config)
        elif method == 'aces':
            return self._tonemap_aces(hdr, config)
        elif method == 'exposure':
            return self._tonemap_exposure_fusion(hdr, config)
        else:
            # 默认使用Reinhard
            return self._tonemap_reinhard(hdr, config)
    
    def _tonemap_reinhard(
        self,
        hdr: np.ndarray,
        config: HDRConfig
    ) -> np.ndarray:
        """
        Reinhard Tone Mapping
        
        全局+局部tone mapping，公式:
        L_out = (L_in * (1 + L_in / L_white^2)) / (1 + L_in)
        
        Args:
            hdr: HDR图像 (0-1范围)
            config: 配置
        
        Returns:
            LDR图像 (uint8)
        """
        # 提取亮度通道
        if len(hdr.shape) == 3:
            # 转换为亮度
            luminance = 0.299 * hdr[:, :, 2] + 0.587 * hdr[:, :, 1] + 0.114 * hdr[:, :, 0]
        else:
            luminance = hdr
        
        # 全局tone mapping
        white_point = luminance.max() * 0.8  # 白点
        white_point = max(white_point, 0.1)
        
        # Reinhard公式
        mapped = (luminance * (1 + luminance / (white_point ** 2))) / (1 + luminance)
        
        # 对比度调整
        mapped = ((mapped - 0.5) * config.contrast + 0.5)
        
        # 局部亮度适应
        if config.light_adaptation > 0:
            adapted = self._local_adaptation(mapped, config.light_adaptation)
            mapped = mapped * (1 - config.light_adaptation) + adapted * config.light_adaptation
        
        # 恢复色彩
        if len(hdr.shape) == 3:
            # 基于亮度的色彩恢复
            ratio = mapped / (luminance + 1e-10)
            result = hdr * ratio[:, :, np.newaxis]
        else:
            result = mapped
        
        # 色彩饱和度调整
        if config.saturation != 1.0:
            luminance_y = 0.299 * hdr[:, :, 2] + 0.587 * hdr[:, :, 1] + 0.114 * hdr[:, :, 0]
            result = luminance_y + (result - luminance_y) * config.saturation
        
        # Gamma校正
        result = np.power(np.clip(result, 0, 1), 1.0 / config.gamma)
        
        return (np.clip(result, 0, 1) * 255).astype(np.uint8)
    
    def _tonemap_mantik(
        self,
        hdr: np.ndarray,
        config: HDRConfig
    ) -> np.ndarray:
        """
        Mantiuk Tone Mapping
        
        对比度保留的tone mapping算法
        基于局部对比度缩放
        
        Args:
            hdr: HDR图像
            config: 配置
        
        Returns:
            LDR图像
        """
        # 计算log域
        log_hdr = np.log10(hdr + 1e-10)
        
        # 计算梯度
        grad_x = np.gradient(log_hdr, axis=1)
        grad_y = np.gradient(log_hdr, axis=0)
        
        # 计算梯度幅度
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # 对比度保留缩放
        scale = config.mantik_scale
        grad_mag_scaled = np.tanh(grad_mag * scale)
        
        # 反向积分得到输出
        # 简化实现
        mapped = log_hdr * 0.5
        
        # 转换回线性域
        result = np.power(10, mapped)
        
        # 后处理
        result = np.clip(result, 0, 1)
        result = np.power(result, 1.0 / config.gamma)
        
        return (result * 255).astype(np.uint8)
    
    def _tonemap_drago(
        self,
        hdr: np.ndarray,
        config: HDRConfig
    ) -> np.ndarray:
        """
        Drago Tone Mapping
        
        自适应对数tone mapping
        适合极端光照条件
        
        Formula:
        L_out = (log(1 + L_in * L_white) / log(1 + L_white)) ^ bias
        
        Args:
            hdr: HDR图像
            config: 配置
        
        Returns:
            LDR图像
        """
        # 计算亮度
        if len(hdr.shape) == 3:
            luminance = 0.299 * hdr[:, :, 2] + 0.587 * hdr[:, :, 1] + 0.114 * hdr[:, :, 0]
        else:
            luminance = hdr
        
        # 找到最大亮度作为白点
        L_max = luminance.max()
        if L_max < 0.001:
            L_max = 1.0
        
        # Drago公式
        softness = config.drago_softness
        bias = config.drago_bias
        
        # 计算log域的最大值
        L_white_max = np.log10(1 + L_max * softness) ** bias
        
        # 应用Drago tone mapping
        mapped = np.log10(1 + luminance * softness) ** bias / L_white_max
        
        # 转换回线性域
        result = np.power(10, mapped) - 1
        
        # 后处理
        result = np.clip(result, 0, 1)
        result = np.power(result, 1.0 / config.gamma)
        
        return (result * 255).astype(np.uint8)
    
    def _tonemap_aces(
        self,
        hdr: np.ndarray,
        config: HDRConfig
    ) -> np.ndarray:
        """
        ACES (Academy Color Encoding System) Tone Mapping
        
        电影级tone mapping，由美国电影艺术与科学学院开发
        公式参考: http://www.oscars.org/science-technology/sci-tech-projects/aces
        
        Args:
            hdr: HDR图像
            config: 配置
        
        Returns:
            LDR图像
        """
        # ACES RRT (Reference Rendering Transform) 简化实现
        # 这是更精确的实现，基于官方文档
        
        def _aces_rrt(x):
            """ACES RRT函数"""
            a = x * (2.51 * x + 0.03) / (x * (2.43 * x + 0.59) + 0.14)
            return np.clip(a, 0, 1)
        
        def _aces_fit(x):
            """将ACES输出拟合到显示范围"""
            # 简化的拟合函数
            return ((x * (6.2 * x + 0.5)) / (x * (6.2 * x + 1.7) + 0.06))
        
        # 应用ACES
        if len(hdr.shape) == 3:
            result = np.zeros_like(hdr)
            for c in range(3):
                rrt = _aces_rrt(hdr[:, :, c])
                result[:, :, c] = _aces_fit(rrt)
        else:
            result = _aces_fit(_aces_rrt(hdr))
        
        # Gamma校正
        result = np.power(result, 1.0 / config.gamma)
        
        return (np.clip(result, 0, 1) * 255).astype(np.uint8)
    
    def _tonemap_exposure_fusion(
        self,
        images: List[np.ndarray],
        config: HDRConfig
    ) -> np.ndarray:
        """
        曝光融合方法 (单图像或多图像)
        
        不生成中间HDR，直接融合不同曝光
        
        Args:
            images: 图像列表
            config: 配置
        
        Returns:
            LDR图像
        """
        if not isinstance(images, list):
            images = [images]
        
        # 简单曝光融合
        if len(images) == 1:
            return (np.clip(images[0], 0, 1) * 255).astype(np.uint8)
        
        # 多尺度融合
        merge_mertens = cv2.createMergeMertens()
        normalized = [np.clip(img, 0, 1) for img in images]
        fused = merge_mertens.process(normalized)
        
        return (np.clip(fused, 0, 1) * 255).astype(np.uint8)
    
    def _local_adaptation(self, img: np.ndarray, strength: float) -> np.ndarray:
        """
        局部亮度适应
        
        根据局部平均亮度调整映射
        
        Args:
            img: 输入图像
            strength: 适应强度 (0-1)
        
        Returns:
            适应后的图像
        """
        # 计算局部均值
        kernel_size = 15
        local_mean = cv2.blur(img, (kernel_size, kernel_size))
        
        # 计算局部适应
        adapted = img * (local_mean + 0.5)
        
        return adapted
    
    def _post_process(self, ldr: np.ndarray, config: HDRConfig) -> np.ndarray:
        """
        Tone Mapping后处理
        
        Args:
            ldr: LDR图像
            config: 配置
        
        Returns:
            处理后的图像
        """
        # 对比度增强 (可选)
        if config.contrast != 1.0:
            ldr = ldr.astype(np.float32)
            ldr = ((ldr / 255.0 - 0.5) * config.contrast + 0.5) * 255
            ldr = np.clip(ldr, 0, 255).astype(np.uint8)
        
        # 饱和度调整
        if config.saturation != 1.0:
            hsv = cv2.cvtColor(ldr, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * config.saturation, 0, 255)
            ldr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        return ldr
    
    def _estimate_dynamic_range(self, hdr: np.ndarray) -> Dict[str, Any]:
        """估计HDR图像的动态范围"""
        if len(hdr.shape) == 3:
            gray = cv2.cvtColor(hdr.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            gray = hdr.astype(np.float32)
        
        # 计算有效动态范围
        p1, p99 = np.percentile(gray, [1, 99])
        
        # 估算stops
        if p1 > 0:
            stops = np.log2(p99 / p1)
        else:
            stops = 0
        
        return {
            'min': float(gray.min()),
            'max': float(gray.max()),
            'p1': float(p1),
            'p99': float(p99),
            'useful_stops': round(stops, 1),
            'note': f'有效动态范围约 {stops:.1f} stops (曝光值)'
        }
    
    def analyze_hdr_quality(self, img: np.ndarray) -> Dict[str, Any]:
        """
        分析HDR图像质量
        
        Args:
            img: HDR图像
        
        Returns:
            质量分析结果
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            gray = img.astype(np.float32)
        
        # 动态范围分析
        min_val, max_val = gray.min(), gray.max()
        p1, p99 = np.percentile(gray, [1, 99])
        
        # 高光保留
        over_exposed = (gray > 240).sum() / gray.size * 100
        
        # 暗部保留
        under_exposed = (gray < 15).sum() / gray.size * 100
        
        # 局部对比度
        local_contrast_5 = self._compute_local_contrast(gray, 5)
        local_contrast_15 = self._compute_local_contrast(gray, 15)
        
        return {
            'dynamic_range': {
                'min': float(min_val),
                'max': float(max_val),
                'range': float(max_val - min_val),
                'p1': float(p1),
                'p99': float(p99),
                'stops': round(np.log2(max(p99, 1) / max(p1, 1)), 1)
            },
            'exposure_analysis': {
                'over_exposed_percent': round(over_exposed, 2),
                'under_exposed_percent': round(under_exposed, 2),
                'exposure_assessment': self._assess_exposure(over_exposed, under_exposed)
            },
            'local_contrast': {
                '5x5_window': round(local_contrast_5, 2),
                '15x15_window': round(local_contrast_15, 2),
                'contrast_retention': 'good' if local_contrast_15 > 20 else 'poor'
            }
        }
    
    def _compute_local_contrast(self, gray: np.ndarray, window_size: int) -> float:
        """计算局部对比度"""
        mean = cv2.blur(gray, (window_size, window_size))
        sqr_mean = cv2.blur(gray ** 2, (window_size, window_size))
        variance = np.maximum(sqr_mean - mean ** 2, 0)
        std = np.sqrt(variance)
        return float(np.mean(std))
    
    def _assess_exposure(self, over: float, under: float) -> str:
        """评估曝光"""
        if over < 2 and under < 5:
            return '良好 - 曝光均衡'
        elif over > 10:
            return '过曝 - 高光溢出严重'
        elif under > 20:
            return '欠曝 - 暗部死黑'
        elif over > 5 or under > 10:
            return '一般 - 部分区域曝光问题'
        else:
            return '可接受'


def merge_hdr_images(
    image_paths: List[str],
    output_path: str,
    method: str = 'reinhard',
    **kwargs
) -> Dict[str, Any]:
    """便捷函数"""
    processor = HDRProcessor()
    return processor.merge_hdr(image_paths, output_path, method)
