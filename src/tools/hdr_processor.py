#!/usr/bin/env python3
"""
HDR/多帧合成处理器
支持多帧HDR合成、多帧降噪
"""
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import cv2
try:
    from ..utils.logger import setup_logger
except ImportError:
    import logging
    def setup_logger(name):
        logging.basicConfig(level=20, format='%(message)s')
        return logging.getLogger(name)

logger = setup_logger('isp-agent.hdr')


# =============================================================================
# Tone Mapping 算法 (纯 NumPy 实现)
# =============================================================================

def tone_mapping_reinhard(
    img: np.ndarray,
    key: float = 0.18,
    phi: float = 1.0,
    radius: float = 1.0
) -> np.ndarray:
    """
    Reinhard Tone Mapping (全局 + 局部)

    基于 "Photographic Tone Reproduction for Digital Images" (Reinhard et al., 2002)

    全局算子:
        L_out = L_in / (1 + L_in)

    局部算子 (基于DoG滤波器检测边缘/高光区域):
        retval = L_out * (1 + L_out / L_white^2) / (1 + L_out)

    Args:
        img: HDR图像，float任意尺度 (单通道或3通道 BGR/ RGB)
        key: 亮度键值 (key value)，默认 0.18 (中性场景)
        phi: 对比度增强参数，值越大对比度越强
        radius: 局部算子半径 (高斯模糊核大小 = 2*radius+1)

    Returns:
        LDR图像，uint8，范围 [0, 255]

    Reference:
        Reinhard et al., "Photographic Tone Reproduction for Digital Images",
        ACM Transactions on Graphics (TOG), 2002
    """
    # 提取亮度通道 (BT.709)
    if len(img.shape) == 3:
        luminance = 0.2126 * img[:, :, 2] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 0]
    else:
        luminance = img.copy()

    # ---------- 全局算子 (Global Operator) ----------
    # 将任意范围的亮度映射到0-1之间
    # key 值用于归一化: 场景平均亮度 → key
    # 公式: Lg = key * L / (1 + L)

    # 计算全局平均亮度 (对数加权，避免极端值影响)
    log_lum = np.log(luminance + 1e-10)
    avg_log_lum = np.exp(np.mean(log_lum))  # 几何平均

    # 归一化亮度
    L_norm = (key / (avg_log_lum + 1e-10)) * luminance

    # 全局 Reinhard 映射
    L_global = L_norm / (1.0 + L_norm)

    # ---------- 局部算子 (Local Operator) ----------
    # 使用高斯差 (Difference of Gaussian) 检测局部高光区域
    # DoG = 低通(L_global) - 更低通(L_global)
    # 边缘区域 DoG ≈ 0 → 使用全局映射；平坦区域使用增强对比度

    # 高斯核尺寸
    sigma1 = max(radius * 0.5, 0.5)
    sigma2 = radius * 1.6

    # 高斯模糊 (使用 2D 滑动窗口实现)
    L_blur1 = _gaussian_blur_2d(L_global, sigma1)
    L_blur2 = _gaussian_blur_2d(L_global, sigma2)

    # DoG (Difference of Gaussian)
    # 使用绝对值，检测亮区和暗区
    dog = np.abs(L_blur1 - L_blur2)

    # 计算局部对比度阈值 (phi 控制对比度增强强度)
    # 公式参考: epsilon = 1 + phi * DoG
    epsilon = 1.0 + phi * dog

    # 局部增强
    # 公式: L_local = L_global * epsilon / (1 + epsilon)
    L_local = L_global * epsilon / (1.0 + epsilon)

    # ---------- 白点自适应 (White Point Adaptation) ----------
    # 避免高光溢出，同时增强暗部细节
    L_white = np.max(L_global)  # 白点 = 最大亮度
    L_white = max(L_white, 0.001)  # 防止除零

    # 最终映射
    L_final = L_local * (1.0 + L_local / (L_white ** 2)) / (1.0 + L_local)

    # ---------- 伽马校正 + 输出 ----------
    # 转换为 LDR (0-255)
    LDR = np.clip(L_final, 0, 1)
    LDR = np.power(LDR, 1.0 / 2.2)  # 伽马校正 (sRGB gamma)

    LDR_uint8 = (LDR * 255).astype(np.uint8)

    # 如果是彩色图像，恢复颜色
    if len(img.shape) == 3:
        # 计算亮度 ratio 来恢复颜色
        ratio = (L_final + 1e-10) / (L_global + 1e-10)
        ratio = np.clip(ratio, 0.5, 2.0)  # 防止颜色过度偏移
        # 将 ratio 扩展到 3 通道
        ratio_3ch = np.stack([ratio, ratio, ratio], axis=-1)
        # 颜色恢复 (保持原始比例感)
        result = img.copy()
        # 基于原始亮度比例
        orig_lum = luminance / (luminance.max() + 1e-10)
        final_lum = L_final / (L_final.max() + 1e-10)
        color_ratio = np.clip(final_lum / (orig_lum + 1e-10), 0.5, 2.0)
        result = result * color_ratio[:, :, np.newaxis]
        result = np.clip(result, 0, 1)
        result = np.power(result, 1.0 / 2.2)
        return (result * 255).astype(np.uint8)
    else:
        return LDR_uint8


def tone_mapping_aces(img: np.ndarray) -> np.ndarray:
    """
    ACES (Academy Color Encoding System) Tone Mapping

    基于 SMPTE ST 2084 (PQ) 标准曲线和 ACES RRT (Reference Rendering Transform)

    公式:
        1. x = max(0, input - 0.004)
        2. y = (x * (6.2 * x + 0.5)) / (x * (6.2 * x + 1.7) + 0.06)
        3. 输出 = y ^ (1/2.2)

    这是基于 "Academy Color Encoding System" 的简化实现，
    适用于将 HDR 转换到可显示的 LDR 范围。

    Reference:
        ACES 1.0 Specification, Academy of Motion Picture Arts and Sciences

    Args:
        img: HDR图像，float任意尺度 (单通道或3通道 BGR/RGB)

    Returns:
        LDR图像，uint8，范围 [0, 255]
    """
    # 归一化到 [0, 1] (基于最大亮度)
    img_max = np.max(img) if np.max(img) > 0 else 1.0
    img_norm = img.astype(np.float64) / img_max

    def _aces_rtt(x: np.ndarray) -> np.ndarray:
        """
        ACES Reference Rendering Transform (RRT)

        将 scene-referred 图像转换为 display-referred
        """
        # 防止负值 (CLIP)
        x = np.maximum(x, 0.0)

        # 参数 (ACES 官方参数)
        a = 2.51
        b = 0.03
        c = 2.43
        d = 0.59
        e = 0.14

        # RRT 公式
        # x = (x * (a * x + b)) / (x * (c * x + d) + e)
        result = (x * (a * x + b)) / (x * (c * x + d) + e)
        return np.clip(result, 0, 1)

    if len(img.shape) == 3:
        # 对每个通道独立应用 RRT
        result = np.zeros_like(img_norm, dtype=np.float64)
        for c in range(3):
            result[:, :, c] = _aces_rtt(img_norm[:, :, c])
    else:
        result = _aces_rtt(img_norm)

    # 色调映射后，进行边缘锐化以恢复一些细节
    # (可选，略微增加清晰度)
    # result = _unsharp_mask(result, amount=0.2)

    # 伽马校正 (转换为输出 gamma)
    result = np.power(result, 1.0 / 2.2)

    # 输出 uint8
    return (np.clip(result, 0, 1) * 255).astype(np.uint8)


def tone_mapping_mantiuk(
    img: np.ndarray,
    contrast: float = 1.0
) -> np.ndarray:
    """
    Mantiuk Tone Mapping

    基于 "A Perceptual Framework for Contrast Processing of High Dynamic Range Images"
    (Mantiuk et al., 2005)

    特点: 保留局部对比度，适合显示设备

    简化实现 (不使用完整的梯度域保角映射):
        1. 将亮度转换到 log domain
        2. 对比度缩放
        3. 转换回线性域

    Args:
        img: HDR图像，float任意尺度
        contrast: 对比度缩放因子 (0.5-2.0)，越大对比度越高

    Returns:
        LDR图像，uint8，范围 [0, 255]
    """
    # 提取亮度通道
    if len(img.shape) == 3:
        luminance = 0.2126 * img[:, :, 2] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 0]
    else:
        luminance = img.copy()

    # 防止 log(0)
    luminance = np.maximum(luminance, 1e-10)

    # 转换到 log domain
    log_lum = np.log10(luminance)

    # 计算局部梯度 (对比度的空间变化)
    grad_x, grad_y = np.gradient(log_lum)

    # 计算梯度幅度
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # Mantiuk 对比度保留: 使用 tanh 压缩大梯度
    # 公式: grad_scaled = contrast * tanh(grad_mag / contrast)
    grad_scaled = contrast * np.tanh(grad_mag / contrast)

    # 梯度方向保持不变
    grad_direction = np.arctan2(grad_y, grad_x)
    grad_x_scaled = grad_scaled * np.cos(grad_direction)
    grad_y_scaled = grad_scaled * np.sin(grad_direction)

    # 从梯度重建图像 (积分)
    # 简化: 直接对梯度进行累加积分
    # 使用傅里叶方法重建 (无旋度场积分)
    reconstructed = _gradient_integrate(grad_x_scaled, grad_y_scaled, log_lum)

    # 转换回线性域
    result_linear = np.power(10, reconstructed)

    # 归一化到 [0, 1]
    result_norm = (result_linear - result_linear.min()) / (
        result_linear.max() - result_linear.min() + 1e-10
    )

    # 伽马校正
    result_gamma = np.power(result_norm, 1.0 / 2.2)

    # 如果是彩色图像
    if len(img.shape) == 3:
        # 计算颜色比例并恢复
        orig_lum_max = luminance.max() + 1e-10
        color_ratio = luminance / orig_lum_max
        color_ratio = np.clip(color_ratio, 0, 1)
        result = img / (luminance[:, :, np.newaxis] + 1e-10) * result_linear[:, :, np.newaxis]
        result = np.clip(result, 0, 1)
        result = np.power(result, 1.0 / 2.2)
        return (result * 255).astype(np.uint8)

    return (result_gamma * 255).astype(np.uint8)


def tone_mapping_drago(
    img: np.ndarray,
    adaptation: float = 1.0
) -> np.ndarray:
    """
    Drago Tone Mapping

    基于 "Adaptive Logarithmic Mapping For Displaying High Contrast Scenes"
    (Drago et al., 2003)

    公式:
        L_out = (log(1 + L_in * L_max * softness)) /
                (log(1 + L_max * softness) ^ bias)

    特点: 自适应对数映射，适合极端光照条件

    Args:
        img: HDR图像，float任意尺度
        adaptation: 适应参数 (0.5-2.0)，控制对比度和细节保留

    Returns:
        LDR图像，uint8，范围 [0, 255]
    """
    # 提取亮度通道
    if len(img.shape) == 3:
        luminance = 0.2126 * img[:, :, 2] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 0]
    else:
        luminance = img.copy()

    # 找到最大亮度 (白点)
    L_max = np.max(luminance)
    if L_max < 0.001:
        L_max = 1.0

    # Drago 参数
    softness = 0.5  # 控制曲线软硬
    bias = 0.6 + 0.4 * adaptation  # bias 范围 [0.6, 1.0]

    # 计算饱和度参数
    # L_white_max = log10(1 + L_max * softness) ^ bias
    L_white_max = (np.log10(1.0 + L_max * softness) ** bias)

    # 防止除零
    L_white_max = max(L_white_max, 1e-10)

    # 应用 Drago tone mapping
    # numerator = log10(1 + luminance * softness * adaptation)
    numerator = np.log10(1.0 + luminance * softness * adaptation) ** bias
    mapped = numerator / L_white_max

    # 转换回线性域
    result = np.power(10, mapped) - 1e-10
    result = np.clip(result, 0, 1)

    # 伽马校正
    result = np.power(result, 1.0 / 2.2)

    # 彩色图像处理
    if len(img.shape) == 3:
        # 保持色调的彩色恢复
        lum_norm = luminance / (L_max + 1e-10)
        lum_norm = np.clip(lum_norm, 0, 1)
        color_scale = mapped / (lum_norm + 1e-10)
        color_scale = np.clip(color_scale, 0.5, 2.0)
        result_color = img / (luminance[:, :, np.newaxis] + 1e-10) * result[:, :, np.newaxis]
        result_color = np.clip(result_color, 0, 1)
        result_color = np.power(result_color, 1.0 / 2.2)
        return (result_color * 255).astype(np.uint8)

    return (result * 255).astype(np.uint8)


# =============================================================================
# 辅助函数
# =============================================================================

def _gaussian_blur_2d(img: np.ndarray, sigma: float) -> np.ndarray:
    """
    纯 NumPy 实现的高斯模糊 (2D)

    使用分离的 1D 高斯核进行卷积

    Args:
        img: 2D 图像 (float)
        sigma: 高斯标准差

    Returns:
        模糊后的图像
    """
    # 高斯核半径 (确保覆盖 3-sigma 范围)
    radius = int(3 * sigma + 0.5)
    size = 2 * radius + 1

    # 创建 1D 高斯核
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(x ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()

    # 分离卷积: 先水平，再垂直
    # 水平卷积
    img_h = _convolve1d(img.astype(np.float64), kernel, axis=1, mode='reflect')
    # 垂直卷积
    img_blurred = _convolve1d(img_h, kernel, axis=0, mode='reflect')

    return img_blurred


def _convolve1d(
    img: np.ndarray,
    kernel: np.ndarray,
    axis: int = 0,
    mode: str = 'reflect'
) -> np.ndarray:
    """
    沿指定轴进行 1D 卷积

    Args:
        img: 输入图像 (2D 或 3D)
        kernel: 1D 卷积核
        axis: 卷积轴 (0=垂直, 1=水平)
        mode: 填充模式

    Returns:
        卷积后的图像
    """
    if axis == 0:
        # 垂直卷积
        img_t = img.T
        result = np.zeros_like(img_t, dtype=np.float64)
        k_len = len(kernel)
        k_half = k_len // 2

        for i in range(img_t.shape[0]):
            row = img_t[i]
            padded = np.pad(row, (k_half, k_half), mode=mode)
            conv = np.convolve(padded, kernel, mode='valid')
            result[i] = conv
        return result.T

    else:
        # 水平卷积
        result = np.zeros_like(img, dtype=np.float64)
        k_len = len(kernel)
        k_half = k_len // 2

        for i in range(img.shape[0]):
            row = img[i]
            padded = np.pad(row, (k_half, k_half), mode=mode)
            conv = np.convolve(padded, kernel, mode='valid')
            result[i] = conv

        return result


def _gradient_integrate(
    grad_x: np.ndarray,
    grad_y: np.ndarray,
    base: np.ndarray
) -> np.ndarray:
    """
    从梯度场重建图像 (泊松积分的简化实现)

    使用傅里叶域的无旋度场积分

    Args:
        grad_x: x方向梯度
        grad_y: y方向梯度
        base: 基础图像 (用于恢复绝对值)

    Returns:
        重建后的图像
    """
    # 使用傅里叶方法求解泊松方程
    # ∇²u = ∂x(gx) + ∂y(gy)
    # 在频域中: U = G / (kx² + ky²)

    h, w = grad_x.shape

    # 计算散度 (divergence)
    div = np.gradient(grad_x, axis=1) + np.gradient(grad_y, axis=0)

    # 傅里叶变换
    div_fft = np.fft.fft2(div)

    # 频率网格
    kx = np.fft.fftfreq(w).reshape(1, w) * w
    ky = np.fft.fftfreq(h).reshape(h, 1) * h
    k_squared = kx ** 2 + ky ** 2

    # 避免除零 (直流分量设为 1)
    k_squared[0, 0] = 1.0

    # 频域积分
    u_fft = div_fft / k_squared

    # 逆变换
    u = np.fft.ifft2(u_fft).real

    # 归一化到与 base 相同的范围
    u = u - u.mean() + base.mean()

    return u


# =============================================================================
# HDR 检测与分析
# =============================================================================

class HDRDetect:
    """
    HDR图像检测器

    检测给定图像是否为 HDR 内容，或者分析图像的动态范围
    """

    @staticmethod
    def is_hdr(img: np.ndarray, threshold_stops: float = 3.0) -> Tuple[bool, Dict[str, Any]]:
        """
        检测图像是否为 HDR

        Args:
            img: 输入图像 (uint8 或 float)
            threshold_stops: HDR判定阈值(曝光值)，默认3.0 stops
                             (2 stops ≈ 4x亮度差)

        Returns:
            (is_hdr, info): 是否为HDR，详细信息
        """
        info = {}

        # 转换到 float 并提取亮度
        if img.dtype == np.uint8:
            img_f = img.astype(np.float32) / 255.0
        else:
            img_f = img.astype(np.float32)

        if len(img_f.shape) == 3:
            gray = 0.299 * img_f[:, :, 0] + 0.587 * img_f[:, :, 1] + 0.114 * img_f[:, :, 2]
        else:
            gray = img_f

        # 计算动态范围
        gray_flat = gray.flatten()
        p0_1, p99_9 = np.percentile(gray_flat, [0.1, 99.9])
        min_val = np.min(gray_flat)
        max_val = np.max(gray_flat)

        info['min_value'] = float(min_val)
        info['max_value'] = float(max_val)
        info['p0_1'] = float(p0_1)
        info['p99_9'] = float(p99_9)
        info['range_raw'] = float(max_val - min_val)

        # 计算有效动态范围 (stops)
        if p0_1 > 1e-8:
            stops = np.log2(p99_9 / p0_1)
        else:
            stops = 0

        info['effective_stops'] = round(stops, 1)
        info['threshold_stops'] = threshold_stops

        is_hdr = stops >= threshold_stops

        # 额外检测: 是否存在过曝或欠曝区域
        over_exposed = np.sum(gray_flat > 250) / len(gray_flat) * 100
        under_exposed = np.sum(gray_flat < 5) / len(gray_flat) * 100
        info['over_exposed_percent'] = round(over_exposed, 2)
        info['under_exposed_percent'] = round(under_exposed, 2)
        info['has_extreme_values'] = over_exposed > 1.0 or under_exposed > 1.0

        return is_hdr, info

    @staticmethod
    def detect_hdr_file(path: str, threshold_stops: float = 3.0) -> Tuple[bool, Dict[str, Any]]:
        """
        检测图像文件是否为 HDR

        Args:
            path: 图像文件路径
            threshold_stops: HDR判定阈值

        Returns:
            (is_hdr, info)
        """
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"无法读取图像: {path}")
        return HDRDetect.is_hdr(img, threshold_stops)


class HDRAnalyzer:
    """
    HDR图像分析器

    分析 HDR 图像的动态范围、曝光分布、高光/暗部细节保留情况
    """

    def __init__(self, img: np.ndarray):
        """
        初始化分析器

        Args:
            img: HDR图像 (float，任意范围)
        """
        if img.dtype == np.uint8:
            self.img = img.astype(np.float32)
        else:
            self.img = img.astype(np.float32)

        # 提取亮度通道
        if len(self.img.shape) == 3:
            self.luminance = (
                0.2126 * self.img[:, :, 2] +
                0.7152 * self.img[:, :, 1] +
                0.0722 * self.img[:, :, 0]
            )
        else:
            self.luminance = self.img.copy()

    def analyze(self) -> Dict[str, Any]:
        """
        全面分析 HDR 图像

        Returns:
            分析结果字典
        """
        return {
            'dynamic_range': self._analyze_dynamic_range(),
            'exposure_distribution': self._analyze_exposure_distribution(),
            'highlight_shadow': self._analyze_highlight_shadow(),
            'local_contrast': self._analyze_local_contrast(),
        }

    def _analyze_dynamic_range(self) -> Dict[str, Any]:
        """分析动态范围"""
        lum = self.luminance.flatten()
        min_val, max_val = np.min(lum), np.max(lum)
        mean_val = np.mean(lum)
        median_val = np.median(lum)

        p1, p5, p95, p99 = np.percentile(lum, [1, 5, 95, 99])

        # 有效动态范围 (stops)
        if p1 > 1e-8:
            stops = np.log2(p99 / p1)
        else:
            stops = 0.0

        return {
            'min': float(min_val),
            'max': float(max_val),
            'mean': float(mean_val),
            'median': float(median_val),
            'p1': float(p1),
            'p5': float(p5),
            'p95': float(p95),
            'p99': float(p99),
            'stops': round(stops, 1),
            'range': float(max_val - min_val),
        }

    def _analyze_exposure_distribution(self) -> Dict[str, Any]:
        """分析曝光分布"""
        lum = self.luminance.flatten()
        total = len(lum)

        # 曝光区间统计
        bins = [
            (0, 0.1, '深黑'),
            (0.1, 0.3, '暗部'),
            (0.3, 0.5, '暗中间调'),
            (0.5, 0.7, '亮中间调'),
            (0.7, 0.9, '亮部'),
            (0.9, 1.0, '高光'),
            (1.0, float('inf'), '过曝'),
        ]

        distribution = {}
        for low, high, name in bins:
            count = np.sum((lum >= low) & (lum < high))
            distribution[name] = round(count / total * 100, 2)

        return distribution

    def _analyze_highlight_shadow(self) -> Dict[str, Any]:
        """分析高光和暗部细节保留"""
        lum = self.luminance.flatten()

        over_exposed = np.sum(lum >= 250) / len(lum) * 100
        under_exposed = np.sum(lum <= 5) / len(lum) * 100
        nearly_over = np.sum(lum >= 240) / len(lum) * 100
        nearly_under = np.sum(lum <= 15) / len(lum) * 100

        # 高光细节评估
        if over_exposed > 5:
            highlight_quality = '差 - 高光溢出严重'
        elif nearly_over > 5:
            highlight_quality = '一般 - 部分高光溢出'
        else:
            highlight_quality = '良好 - 高光细节保留'

        # 暗部细节评估
        if under_exposed > 20:
            shadow_quality = '差 - 暗部死黑'
        elif nearly_under > 20:
            shadow_quality = '一般 - 部分暗部丢失'
        else:
            shadow_quality = '良好 - 暗部细节保留'

        return {
            'over_exposed_percent': round(over_exposed, 2),
            'under_exposed_percent': round(under_exposed, 2),
            'nearly_over_exposed_percent': round(nearly_over, 2),
            'nearly_under_exposed_percent': round(nearly_under, 2),
            'highlight_quality': highlight_quality,
            'shadow_quality': shadow_quality,
        }

    def _analyze_local_contrast(self) -> Dict[str, Any]:
        """分析局部对比度"""
        # 在不同窗口尺寸下计算局部对比度
        results = {}
        for window in [5, 15, 31]:
            sigma = window / 6.0
            local_mean = _gaussian_blur_2d(self.luminance, sigma)
            local_sqr_mean = _gaussian_blur_2d(self.luminance ** 2, sigma)
            local_var = np.maximum(local_sqr_mean - local_mean ** 2, 0)
            local_std = np.sqrt(local_var)
            results[f'{window}x{window}'] = round(float(np.mean(local_std)), 4)

        return results

    def recommend_tone_mapping(self) -> str:
        """
        根据分析结果推荐最适合的 Tone Mapping 算法

        Returns:
            推荐的算法名称
        """
        dyn_range = self._analyze_dynamic_range()
        highlight_shadow = self._analyze_highlight_shadow()

        stops = dyn_range['stops']
        over_pct = highlight_shadow['over_exposed_percent']
        under_pct = highlight_shadow['under_exposed_percent']

        # 根据动态范围推荐
        if stops >= 6:
            # 极高动态范围 → Drago (对数映射最适合极端光照)
            return 'drago'
        elif stops >= 4:
            # 高动态范围 → ACES (电影级色调映射)
            if over_pct > 3 or under_pct > 10:
                return 'drago'
            return 'aces'
        elif stops >= 3:
            # 中等动态范围 → Reinhard (通用)
            return 'reinhard'
        else:
            # 低动态范围 → Reinhard 轻量版或不需要 TM
            return 'reinhard'

    # =============================================================================
    # Phase 2.2: HDR 质量评估
    # =============================================================================

    def calculate_dynamic_range(self, img: Optional[np.ndarray] = None) -> float:
        """
        计算图像动态范围 (单位: stops/EV)

        Dynamic Range (stops) = log2(max / min)

        使用1%~99%百分位而非绝对最值，以避免极端噪声影响

        Args:
            img: 输入图像 (float, 若为None则使用初始化时的图像)
                 支持 uint8 (自动转float) 或任意范围 float

        Returns:
            动态范围 (stops/EV)
        """
        if img is None:
            lum = self.luminance
        else:
            if img.dtype == np.uint8:
                lum = img.astype(np.float32) / 255.0
            else:
                lum = img.astype(np.float32)

            if len(lum.shape) == 3:
                lum = (0.2126 * lum[:, :, 2] +
                       0.7152 * lum[:, :, 1] +
                       0.0722 * lum[:, :, 0])
            else:
                lum = lum.copy()

        lum_flat = lum.flatten()

        # 使用1%~99%百分位计算有效动态范围
        p_low = np.percentile(lum_flat, 1)
        p_high = np.percentile(lum_flat, 99)

        eps = 1e-10
        if p_low < eps:
            # 最暗部被截断，动态范围受限
            stops = np.log2((p_high + eps) / eps)
        else:
            stops = np.log2(p_high / p_low)

        return round(max(stops, 0.0), 2)

    def evaluate_tone_mapping(
        self,
        original: np.ndarray,
        tone_mapped: np.ndarray
    ) -> Dict[str, Any]:
        """
        评估 Tone Mapping 质量

        对比原始HDR图像与Tone Mapping后的LDR图像，分析:
        1. 有效动态范围
        2. 高光保留评估 (高光溢出检测)
        3. 暗部保留评估 (阴影细节检测)
        4. Tone mapping曲线分析 (对比度保留率)

        Args:
            original: 原始HDR图像 (float, 线性空间)
            tone_mapped: Tone Mapping后的LDR图像 (uint8 [0,255] 或 float [0,1])

        Returns:
            Dict: 质量评估报告，包含以下键:
              - dynamic_range_original: 原始HDR动态范围 (stops)
              - dynamic_range_tm: TM后动态范围 (stops)
              - dr_preservation_rate: 动态范围保留率 (%)
              - highlight_clipping: 高光溢出分析
                  - clipped_pixels: 溢出像素数量
                  - clipped_percent: 溢出比例 (%)
                  - assessment: 评估结果 (良好/一般/差)
              - shadow_detail: 暗部细节分析
                  - shadow_pixels: 暗部(<5%)像素数量
                  - shadow_percent: 暗部比例 (%)
                  - assessment: 评估结果
              - contrast_preservation: 对比度保留分析
                  - original_contrast: 原始对比度 (RMS)
                  - tm_contrast: TM后对比度 (RMS)
                  - preservation_rate: 保留率 (%)
                  - assessment: 评估结果
              - overall_score: 综合评分 (0-100)
              - recommendations: 改进建议 (List[str])
        """
        # ---------- 预处理 ----------
        if original.dtype == np.uint8:
            orig_f = original.astype(np.float32) / 255.0
        else:
            orig_f = original.astype(np.float32)

        if tone_mapped.dtype == np.uint8:
            tm_f = tone_mapped.astype(np.float32) / 255.0
        else:
            tm_f = tone_mapped.astype(np.float32)

        # 亮度提取
        if len(orig_f.shape) == 3:
            orig_lum = (0.2126 * orig_f[:, :, 2] +
                        0.7152 * orig_f[:, :, 1] +
                        0.0722 * orig_f[:, :, 0])
        else:
            orig_lum = orig_f.copy()

        if len(tm_f.shape) == 3:
            tm_lum = (0.2126 * tm_f[:, :, 2] +
                      0.7152 * tm_f[:, :, 1] +
                      0.0722 * tm_f[:, :, 0])
        else:
            tm_lum = tm_f.copy()

        orig_flat = orig_lum.flatten()
        tm_flat = tm_lum.flatten()
        total_pixels = len(tm_flat)

        # ---------- 1. 动态范围分析 ----------
        dr_orig = self.calculate_dynamic_range(orig_lum)
        dr_tm = self.calculate_dynamic_range(tm_lum)
        dr_preservation = round(
            min(dr_tm / (dr_orig + 1e-10) * 100, 100.0), 1
        )

        # ---------- 2. 高光保留评估 ----------
        # TM后高光区域 (tm_lum >= 0.95) 映射到 tm_lum == 1.0 的比例
        highlight_threshold = 0.95
        tm_highlight = tm_flat >= highlight_threshold
        orig_in_highlight = orig_flat[tm_highlight] if tm_highlight.any() else np.array([])
        # 原始HDR中该区域是否有变化 (说明高光被保留而非截断)
        if len(orig_in_highlight) > 0:
            orig_highlight_std = np.std(orig_in_highlight)
            # 如果原始高光区域变化大，但TM后都聚到1.0，说明溢出
            highlight_variance_loss = 1.0 - min(orig_highlight_std / (np.std(orig_flat) + 1e-10), 1.0)
        else:
            highlight_variance_loss = 0.0

        clipped_count = int(np.sum(tm_flat >= 0.99))
        clipped_percent = round(clipped_count / total_pixels * 100, 3)

        if clipped_percent > 10:
            highlight_assessment = '差 - 高光溢出严重'
        elif clipped_percent > 3:
            highlight_assessment = '一般 - 部分高光溢出'
        elif highlight_variance_loss > 0.5:
            highlight_assessment = '一般 - 高光细节损失'
        else:
            highlight_assessment = '良好 - 高光细节保留'

        # ---------- 3. 暗部保留评估 ----------
        shadow_threshold_low = 0.05
        shadow_threshold_high = 0.2
        # 统计TM后暗部区域
        tm_shadow = (tm_flat >= shadow_threshold_low) & (tm_flat <= shadow_threshold_high)
        shadow_pixels = int(np.sum(tm_shadow))
        shadow_percent = round(shadow_pixels / total_pixels * 100, 2)

        # 原始暗部在TM后的分布是否分散 (分散=细节保留)
        orig_in_shadow = orig_flat[tm_shadow] if tm_shadow.any() else np.array([])
        if len(orig_in_shadow) > 0:
            shadow_detail_score = round(np.std(orig_in_shadow) * 100, 3)
        else:
            shadow_detail_score = 0.0

        if shadow_percent > 40:
            shadow_assessment = '良好 - 暗部细节丰富'
        elif shadow_percent > 20:
            shadow_assessment = '一般 - 暗部细节一般'
        else:
            shadow_assessment = '差 - 暗部死黑/细节丢失'

        # ---------- 4. 对比度保留率 (RMS对比度) ----------
        # 全局对比度: 亮度RMS
        orig_contrast = float(np.sqrt(np.mean((orig_flat - orig_flat.mean()) ** 2)))
        tm_contrast = float(np.sqrt(np.mean((tm_flat - tm_flat.mean()) ** 2)))
        contrast_preservation = round(
            min(tm_contrast / (orig_contrast + 1e-10) * 100, 100.0), 1
        )

        if contrast_preservation >= 70:
            contrast_assessment = '良好 - 对比度保留完整'
        elif contrast_preservation >= 50:
            contrast_assessment = '一般 - 对比度部分压缩'
        else:
            contrast_assessment = '差 - 对比度严重压缩'

        # ---------- 5. 综合评分 (0-100) ----------
        # 加权平均: DR保留40%, 高光30%, 暗部20%, 对比度10%
        score_dr = min(dr_preservation, 100.0)
        score_highlight = max(0, 100 - clipped_percent * 10)  # 溢出越多扣分越多
        score_shadow = min(shadow_percent * 2, 100) if shadow_percent < 50 else 100
        score_contrast = contrast_preservation

        overall_score = round(
            score_dr * 0.4 +
            score_highlight * 0.3 +
            score_shadow * 0.2 +
            score_contrast * 0.1, 1
        )

        # ---------- 6. 改进建议 ----------
        recommendations = []
        if dr_preservation < 60:
            recommendations.append(
                f"动态范围保留率仅{dr_preservation}%，建议使用保留范围更强的TM算法(如ACES)"
            )
        if clipped_percent > 5:
            recommendations.append(
                f"高光溢出比例{clipped_percent}%，建议调整曝光或使用更好的TM"
            )
        if shadow_percent < 15:
            recommendations.append(
                f"暗部细节比例{shadow_percent}%偏低，建议增强暗部提亮"
            )
        if contrast_preservation < 50:
            recommendations.append(
                f"对比度保留率仅{contrast_preservation}%，结果可能偏平淡"
            )
        if not recommendations:
            recommendations.append("整体质量良好，各项指标正常")

        return {
            'dynamic_range_original': dr_orig,
            'dynamic_range_tm': dr_tm,
            'dr_preservation_rate': dr_preservation,
            'highlight_clipping': {
                'clipped_pixels': clipped_count,
                'clipped_percent': clipped_percent,
                'assessment': highlight_assessment,
            },
            'shadow_detail': {
                'shadow_pixels': shadow_pixels,
                'shadow_percent': shadow_percent,
                'detail_score': shadow_detail_score,
                'assessment': shadow_assessment,
            },
            'contrast_preservation': {
                'original_contrast': round(orig_contrast, 4),
                'tm_contrast': round(tm_contrast, 4),
                'preservation_rate': contrast_preservation,
                'assessment': contrast_assessment,
            },
            'overall_score': overall_score,
            'recommendations': recommendations,
        }


# =============================================================================
# 原有的 HDRProcessor 类 (保持向后兼容)
# =============================================================================

class HDRProcessor:
    """HDR/多帧合成处理器"""

    def __init__(self):
        pass

    def merge_hdr(
        self,
        image_paths: List[str],
        output_path: str,
        method: str = 'exposure'
    ) -> Dict[str, Any]:
        """
        多帧HDR合成

        Args:
            image_paths: 曝光序列图像路径列表（至少3张）
            output_path: 输出路径
            method: 合成方法 ('exposure' / 'mtb' / 'render')

        Returns:
            Dict: 处理结果
        """
        logger.info(f"开始HDR合成: {len(image_paths)} 张图像")

        if len(image_paths) < 2:
            return {'success': False, 'error': '需要至少2张图像'}

        try:
            if method == 'exposure':
                result = self._merge_by_exposure(image_paths, output_path)
            elif method == 'mtb':
                result = self._merge_by_mtb(image_paths, output_path)
            elif method == 'render':
                result = self._merge_and_render(image_paths, output_path)
            else:
                return {'success': False, 'error': f'不支持的方法: {method}'}

            return {'success': True, 'output': output_path, 'method': method}

        except Exception as e:
            logger.error(f"HDR合成失败: {e}")
            return {'success': False, 'error': str(e)}

    def _merge_by_exposure(self, image_paths: List[str], output_path: str) -> None:
        """基于曝光合成"""
        images = []
        for path in image_paths:
            img = cv2.imread(path)
            if img is None:
                raise ValueError(f"无法读取图像: {path}")
            images.append(img)

        imgs = [np.float32(img) for img in images]

        merge_mertens = cv2.createMergeMertens()
        merged = merge_mertens.process(imgs)

        merged = cv2.normalize(merged, None, 0, 255, cv2.NORM_MINMAX)
        merged = np.uint8(merged)

        cv2.imwrite(output_path, merged)
        logger.info(f"HDR合成完成: {output_path}")

    def _merge_by_mtb(self, image_paths: List[str], output_path: str) -> None:
        """基于MTB(中值阈值分箱)合成"""
        base = cv2.imread(image_paths[0])
        if base is None:
            raise ValueError(f"无法读取基准图像: {image_paths[0]}")

        gray_base = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)

        _, mtb = cv2.threshold(gray_base, 127, 255, cv2.THRESH_BINARY)

        result = base
        for path in image_paths[1:]:
            img = cv2.imread(path)
            if img is not None:
                result = cv2.addWeighted(result, 0.7, img, 0.3, 0)

        cv2.imwrite(output_path, result)

    def _merge_and_render(self, image_paths: List[str], output_path: str) -> None:
        """HDR渲染合成"""
        images = []
        exposures = []

        for path in image_paths:
            img = cv2.imread(path)
            if img is not None:
                img = np.float32(img) / 255.0
                images.append(img)
                exposures.append(1.0)

        if len(images) < 2:
            raise ValueError("需要至少2张图像")

        merge_mertens = cv2.createMergeMertens()
        hdr = merge_mertens.process(images)

        # 使用纯 NumPy 的 Reinhard (替代 OpenCV 版本)
        # 先归一化 HDR 到合理范围
        hdr_max = np.max(hdr)
        if hdr_max > 0:
            hdr_normalized = hdr * (1.0 / hdr_max) * 2.0  # 缩放到约 0-2 范围
        else:
            hdr_normalized = hdr

        ldr = tone_mapping_reinhard(hdr_normalized, key=0.18, phi=1.0, radius=1.0)

        cv2.imwrite(output_path, ldr)
        logger.info(f"HDR渲染完成: {output_path}")

    def multi_frame_denoise(
        self,
        image_paths: List[str],
        output_path: str
    ) -> Dict[str, Any]:
        """多帧降噪"""
        logger.info(f"开始多帧降噪: {len(image_paths)} 张图像")

        if len(image_paths) < 2:
            return {'success': False, 'error': '需要至少2张图像进行多帧降噪'}

        try:
            images = []
            for path in image_paths:
                img = cv2.imread(path)
                if img is None:
                    raise ValueError(f"无法读取图像: {path}")
                images.append(img)

            stack = np.stack(images, axis=0)
            denoised = np.median(stack, axis=0).astype(np.uint8)

            cv2.imwrite(output_path, denoised)

            logger.info(f"多帧降噪完成: {output_path}")
            return {'success': True, 'output': output_path, 'frames_used': len(images)}

        except Exception as e:
            logger.error(f"多帧降噪失败: {e}")
            return {'success': False, 'error': str(e)}

    def align_images(
        self,
        reference_path: str,
        image_paths: List[str],
        output_dir: str
    ) -> List[str]:
        """图像对齐（用于手持HDR拍摄）"""
        logger.info("开始图像对齐")

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        ref = cv2.imread(reference_path)
        if ref is None:
            raise ValueError(f"无法读取参考图像: {reference_path}")

        ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)

        sift = cv2.SIFT_create()
        kp_ref, des_ref = sift.detectAndCompute(ref_gray, None)

        output_paths = []

        for i, path in enumerate(image_paths):
            img = cv2.imread(path)
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kp, des = sift.detectAndCompute(gray, None)

            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des, des_ref, k=2)

            good = []
            for m_n in matches:
                if len(m_n) == 2:
                    if m_n[0].distance < 0.75 * m_n[1].distance:
                        good.append(m_n[0])

            if len(good) > 10:
                src_pts =                src_pts = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_ref[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                aligned = cv2.warpPerspective(img, M, (ref.shape[1], ref.shape[0]))

                output_path = os.path.join(output_dir, f"aligned_{i}.jpg")
                cv2.imwrite(output_path, aligned)
                output_paths.append(output_path)
            else:
                output_path = os.path.join(output_dir, f"aligned_{i}.jpg")
                cv2.imwrite(output_path, img)
                output_paths.append(output_path)

        logger.info(f"图像对齐完成: {len(output_paths)} 张")
        return output_paths


def merge_hdr_images(image_paths: List[str], output_path: str, method: str = 'exposure') -> Dict[str, Any]:
    """便捷函数"""
    processor = HDRProcessor()
    return processor.merge_hdr(image_paths, output_path, method)


def denoise_multi_frame(image_paths: List[str], output_path: str) -> Dict[str, Any]:
    """便捷函数"""
    processor = HDRProcessor()
    return processor.multi_frame_denoise(image_paths, output_path)
