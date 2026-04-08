#!/usr/bin/env python3
"""
ISO 16505 图像质量指标模块
实现ISO 16505标准中定义的色彩准确性(ΔE)、锐度和噪声水平指标

ISO 16505 (Road vehicles - Camera monitor systems) 定义了车载摄像头系统的最低性能要求:
- 色彩准确性: ΔE2000 < 14 (可接受), < 10 (良好)
- 锐度: MTF > 0.25 @ Nyquist (最低要求)
- 噪声水平: SNR > 30dB (最低), > 40dB (良好)

Author: ISP Team
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
from enum import Enum

try:
    from ..utils.logger import setup_logger
except ImportError:
    import logging
    def setup_logger(name):
        logging.basicConfig(level=20, format='%(message)s')
        return logging.getLogger(name)

logger = setup_logger('isp-agent.iso16505')


# =============================================================================
# 色彩空间转换工具
# =============================================================================

def rgb_to_lab(image: np.ndarray) -> np.ndarray:
    """
    将RGB图像转换到CIE Lab色彩空间

    Args:
        image: RGB图像，uint8 [0,255] 或 float [0,1]

    Returns:
        Lab图像，float (L: 0-100, a: -128-127, b: -128-127)
    """
    if image.dtype == np.uint8:
        img = image.astype(np.float32) / 255.0
    else:
        img = image.astype(np.float32)

    def _gamma_correct(rgb):
        mask = rgb > 0.04045
        return np.where(mask, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)

    r = _gamma_correct(img[:, :, 0])
    g = _gamma_correct(img[:, :, 1])
    b = _gamma_correct(img[:, :, 2])

    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

    xn, yn, zn = 0.95047, 1.0, 1.08883

    def _f(t):
        delta = 6.0 / 29.0
        return np.where(t > delta ** 3, t ** (1.0 / 3.0), t / (3 * delta ** 2) + 4.0 / 29.0)

    fx = _f(x / xn)
    fy = _f(y / yn)
    fz = _f(z / zn)

    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b_lab = 200.0 * (fy - fz)

    return np.stack([L, a, b_lab], axis=-1)


class DeltaEScale(Enum):
    """ΔE色差标准"""
    DE76 = 'delta_e_76'
    DE94 = 'delta_e_94'
    DE00 = 'delta_e_00'


def delta_e_76(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """CIE 1976 ΔE*ab 色差"""
    def _to_float(lab):
        if lab.dtype == np.uint8:
            L = lab[:, :, 0].astype(np.float32) / 2.55
            a = lab[:, :, 1].astype(np.float32) - 128.0
            b = lab[:, :, 2].astype(np.float32) - 128.0
        else:
            L = lab[:, :, 0].astype(np.float32)
            a = lab[:, :, 1].astype(np.float32)
            b = lab[:, :, 2].astype(np.float32)
        return np.stack([L, a, b], axis=-1)

    lab1_f = _to_float(lab1)
    lab2_f = _to_float(lab2)
    diff = lab1_f - lab2_f
    return np.sqrt(np.sum(diff ** 2, axis=-1))


def delta_e_94(lab1: np.ndarray, lab2: np.ndarray, kL=1.0, kC=1.0, kH=1.0) -> np.ndarray:
    """CIE 1994 ΔE*94 色差"""
    def _to_flat(lab):
        if lab.dtype == np.uint8:
            L = lab[:, :, 0].astype(np.float32) / 2.55
            a = lab[:, :, 1].astype(np.float32) - 128.0
            b = lab[:, :, 2].astype(np.float32) - 128.0
        else:
            L, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
        return L.astype(np.float32), a.astype(np.float32), b.astype(np.float32)

    L1, a1, b1 = _to_flat(lab1)
    L2, a2, b2 = _to_flat(lab2)

    delta_L = L1 - L2
    C1 = np.sqrt(a1 ** 2 + b1 ** 2)
    C2 = np.sqrt(a2 ** 2 + b2 ** 2)
    delta_C = C1 - C2
    delta_H_sq = np.maximum(a1 ** 2 + b1 ** 2 - C1 ** 2 - (a2 ** 2 + b2 ** 2 - C2 ** 2), 0)
    delta_H = np.sqrt(delta_H_sq)

    SL, SC, SH = 1.0, 1.0 + 0.045 * C1, 1.0 + 0.015 * C1

    return np.sqrt(
        (delta_L / (kL * SL)) ** 2 +
        (delta_C / (kC * SC)) ** 2 +
        (delta_H / (kH * SH)) ** 2
    )


def delta_e_00(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """CIEDE2000 ΔE*00 色差 (最符合人眼感知)"""
    def _to_flat(lab):
        if lab.dtype == np.uint8:
            L = lab[:, :, 0].astype(np.float32) / 2.55
            a = lab[:, :, 1].astype(np.float32) - 128.0
            b = lab[:, :, 2].astype(np.float32) - 128.0
        else:
            L, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
        return L.astype(np.float32), a.astype(np.float32), b.astype(np.float32)

    L1, a1, b1 = _to_flat(lab1)
    L2, a2, b2 = _to_flat(lab2)

    C1 = np.sqrt(a1 ** 2 + b1 ** 2)
    C2 = np.sqrt(a2 ** 2 + b2 ** 2)
    C_mean = (C1 + C2) / 2.0
    C7 = C_mean ** 7
    G = 0.5 * (1.0 - np.sqrt(C7 / (C7 + 25 ** 7)))

    a1p = a1 * (1.0 + G)
    a2p = a2 * (1.0 + G)
    C1p = np.sqrt(a1p ** 2 + b1 ** 2)
    C2p = np.sqrt(a2p ** 2 + b2 ** 2)

    h1p = np.arctan2(b1, a1p) * 180.0 / np.pi
    h1p = np.where(h1p < 0, h1p + 360.0, h1p)
    h2p = np.arctan2(b2, a2p) * 180.0 / np.pi
    h2p = np.where(h2p < 0, h2p + 360.0, h2p)

    deltaLp = L2 - L1
    deltaCp = C2p - C1p

    diff_a = a1p - a2p
    diff_b = b1 - b2
    deltaHp_sq = np.maximum(diff_a ** 2 + diff_b ** 2 - deltaCp ** 2, 0)
    deltaHp = np.where(
        np.abs(diff_a) + np.abs(diff_b) < 1e-10,
        0.0,
        np.where(np.abs(h2p - h1p) <= 180.0,
                 h2p - h1p,
                 np.where(h2p - h1p > 180.0, h2p - h1p - 360.0, h2p - h1p + 360.0))
    )
    deltaHp = 2.0 * np.sqrt(C1p * C2p) * np.sin(np.pi * deltaHp / 360.0)

    L1_bar = (L1 + L2) / 2.0
    C1_bar = (C1p + C2p) / 2.0

    h_bar = np.where(np.abs(h1p - h2p) > 180.0,
                     (h1p + h2p + 360.0) / 2.0,
                     (h1p + h2p) / 2.0)

    T = (1.0 - 0.17 * np.cos(np.pi * (h_bar - 30) / 180.0)
            + 0.2 * np.cos(np.pi * 7.0 * h_bar / 180.0)
            - 0.4 * np.cos(np.pi * (h_bar + 6) / 180.0)
            + 0.2 * np.cos(np.pi * (12.0 * h_bar + 63) / 180.0))

    L50_sq = (L1_bar - 50) ** 2
    SL = 1.0 + 0.015 * L50_sq / np.sqrt(20.0 + L50_sq)
    SC = 1.0 + 0.045 * C1_bar
    SH = 1.0 + 0.015 * C1_bar * T

    C7_bar = C1_bar ** 7
    RC = 2.0 * np.sqrt(C7_bar / (C7_bar + 25 ** 7))
    delta_theta = 30.0 * np.exp(-((h_bar - 275) / 25) ** 2)
    RT = -np.sin(np.pi * 2 * delta_theta / 180.0) * RC

    de00 = np.sqrt(
        (deltaLp / SL) ** 2 +
        (deltaCp / SC) ** 2 +
        (deltaHp / SH) ** 2 +
        RT * (deltaCp / SC) * (deltaHp / SH)
    )
    return np.clip(de00, 0, 100)


# =============================================================================
# 锐度 / MTF 计算
# =============================================================================

def compute_mtf(image: np.ndarray, direction: str = 'horizontal',
                roi: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
    """计算MTF曲线 (Slanted Edge Method)"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    if gray.dtype == np.uint8:
        gray = gray.astype(np.float64) / 255.0

    if roi:
        x, y, w, h = roi
        gray = gray[y:y+h, x:x+w]

    # 确保profile是1D数组
    profile = np.mean(gray, axis=0) if direction == 'horizontal' else np.mean(gray, axis=1)
    profile = np.asarray(profile).flatten()

    lsf = np.abs(np.diff(profile))
    lsf_max = lsf.max() + 1e-10
    lsf = lsf / lsf_max

    mtf = np.abs(np.fft.fft(lsf))
    mtf = mtf[:len(mtf) // 2]
    mtf_norm = mtf[0] + 1e-10
    mtf = mtf / mtf_norm
    return mtf


def compute_mtf50(mtf: np.ndarray) -> Tuple[float, float]:
    """MTF@50: 找到MTF下降到0.5的频率"""
    if len(mtf) < 2:
        return 0.0, 0.0
    idx = np.where(mtf < 0.5)[0]
    if len(idx) == 0:
        return 0.5, mtf[-1]
    freq = np.linspace(0, 0.5, len(mtf))
    return freq[idx[0]], 0.5


def compute_sharpness_mtf(image: np.ndarray,
                           roi: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, Any]:
    """ISO 16505兼容的锐度MTF指标"""
    mtf_h = compute_mtf(image, 'horizontal', roi)
    mtf_v = compute_mtf(image, 'vertical', roi)
    # 取相同长度(非方形图像水平/垂直方向长度不同)
    min_len = min(len(mtf_h), len(mtf_v))
    mtf_avg = (mtf_h[:min_len] + mtf_v[:min_len]) / 2.0

    mtf50_freq, _ = compute_mtf50(mtf_avg)
    n = min(int(0.5 * len(mtf_avg)), len(mtf_avg) - 1)
    mtf_nyquist = float(mtf_avg[n]) if n > 0 else 0.0

    iso_min, iso_rec = 0.25, 0.4
    if mtf_nyquist >= 0.6:
        acuity_score, iso_pass, level = 100.0, True, 'excellent'
    elif mtf_nyquist >= iso_rec:
        acuity_score = 50.0 + 50.0 * (mtf_nyquist - iso_rec) / (0.6 - iso_rec)
        iso_pass, level = True, 'good'
    elif mtf_nyquist >= iso_min:
        acuity_score = 50.0 + 50.0 * (mtf_nyquist - iso_min) / (iso_rec - iso_min)
        iso_pass, level = True, 'acceptable'
    else:
        acuity_score = 50.0 * mtf_nyquist / iso_min
        iso_pass, level = False, 'poor'

    return {
        'mtf_50': round(mtf50_freq, 4),
        'mtf_nyquist': round(mtf_nyquist, 4),
        'acuity_score': round(acuity_score, 1),
        'pass_iso16505': iso_pass,
        'iso16505_level': level
    }


# =============================================================================
# 噪声水平计算
# =============================================================================

def compute_snr(image: np.ndarray, roi: Optional[Tuple[int, int, int, int]] = None) -> float:
    """计算SNR (dB)"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    if roi:
        x, y, w, h = roi
        gray = gray[y:y+h, x:x+w]

    if gray.dtype == np.uint8:
        gray = gray.astype(np.float64) / 255.0

    blur = cv2.blur(gray, (5, 5))
    noise = gray - blur

    sig_pow = np.mean(blur ** 2)
    noise_pow = np.mean(noise ** 2)

    if noise_pow < 1e-10:
        return 100.0
    return round(10.0 * np.log10(sig_pow / noise_pow), 2)


def compute_noise_iso16505(image: np.ndarray,
                           roi: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, Any]:
    """ISO 16505兼容的噪声指标"""
    snr = compute_snr(image, roi)

    if snr >= 50.0:
        score, level, iso_pass = 100.0, 'excellent', True
    elif snr >= 40.0:
        score = 50.0 + 50.0 * (snr - 40.0) / 10.0
        level, iso_pass = 'good', True
    elif snr >= 30.0:
        score = 50.0 * (snr - 30.0) / 10.0
        level, iso_pass = 'acceptable', True
    else:
        score = max(0.0, 50.0 * snr / 30.0)
        level, iso_pass = 'poor', False

    return {
        'snr_db': snr,
        'noise_score': round(score, 1),
        'pass_iso16505': iso_pass,
        'iso16505_level': level
    }


# =============================================================================
# ISO 16505 综合评估
# =============================================================================

@dataclass
class ISO16505Result:
    """ISO 16505 评估结果"""
    color_accuracy: Dict[str, Any] = field(default_factory=dict)
    sharpness: Dict[str, Any] = field(default_factory=dict)
    noise: Dict[str, Any] = field(default_factory=dict)
    overall_score: float = 0.0
    compliant: bool = False
    level: str = 'non_compliant'


class ISO16505Evaluator:
    """
    ISO 16505 车载摄像头图像质量评估器

    符合 ISO 16505:2015 Road vehicles - Camera monitor systems 要求

    主要指标:
    - 色彩准确性 (Colorimetric accuracy): ΔE2000
    - 锐度 (Sharpness): MTF @ Nyquist
    - 噪声水平 (Noise): SNR (dB)
    """

    THRESHOLDS = {
        'delta_e_acceptable': 14.0,
        'delta_e_good': 10.0,
        'delta_e_excellent': 6.0,
        'snr_minimum': 30.0,
        'snr_good': 40.0,
        'snr_excellent': 50.0,
        'mtf_nyquist_minimum': 0.25,
        'mtf_nyquist_good': 0.40,
        'mtf_nyquist_excellent': 0.60,
    }

    def __init__(self, reference_image: Optional[np.ndarray] = None):
        self.reference = reference_image

    def assess_color_accuracy(
        self,
        test_image: np.ndarray,
        reference: Optional[np.ndarray] = None,
        scale: DeltaEScale = DeltaEScale.DE00
    ) -> Dict[str, Any]:
        """评估色彩准确性 (ΔE)"""
        ref = reference if reference is not None else (self.reference)
        if ref is None:
            raise ValueError("需要提供参考图像才能评估色彩准确性")

        if test_image.shape != ref.shape:
            ref = cv2.resize(ref, (test_image.shape[1], test_image.shape[0]))

        lab_test = rgb_to_lab(test_image)
        lab_ref = rgb_to_lab(ref)

        if scale == DeltaEScale.DE76:
            de_map = delta_e_76(lab_test, lab_ref)
        elif scale == DeltaEScale.DE94:
            de_map = delta_e_94(lab_test, lab_ref)
        else:
            de_map = delta_e_00(lab_test, lab_ref)

        mean_de = float(np.mean(de_map))
        max_de = float(np.max(de_map))
        median_de = float(np.median(de_map))
        p95_de = float(np.percentile(de_map, 95))

        t = self.THRESHOLDS
        if mean_de <= t['delta_e_excellent']:
            score, level, iso_pass = 100.0, 'excellent', True
        elif mean_de <= t['delta_e_good']:
            score = 50.0 + 50.0 * (t['delta_e_good'] - mean_de) / (t['delta_e_good'] - t['delta_e_excellent'])
            level, iso_pass = 'good', True
        elif mean_de <= t['delta_e_acceptable']:
            score = 50.0 * (t['delta_e_acceptable'] - mean_de) / (t['delta_e_acceptable'] - t['delta_e_good'])
            level, iso_pass = 'acceptable', True
        else:
            score = max(0.0, 50.0 * (20.0 - mean_de) / (20.0 - t['delta_e_acceptable']))
            level, iso_pass = 'poor', False

        return {
            'scale': scale.value,
            'mean_de': round(mean_de, 2),
            'max_de': round(max_de, 2),
            'median_de': round(median_de, 2),
            'p95_de': round(p95_de, 2),
            'score': round(score, 1),
            'level': level,
            'pass_iso16505': iso_pass,
        }

    def assess_sharpness(self, image: np.ndarray,
                          roi: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, Any]:
        """评估锐度 (MTF)"""
        return compute_sharpness_mtf(image, roi)

    def assess_noise(self, image: np.ndarray,
                     roi: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, Any]:
        """评估噪声水平"""
        return compute_noise_iso16505(image, roi)

    def evaluate(self, test_image: np.ndarray) -> ISO16505Result:
        """完整ISO 16505评估"""
        result = ISO16505Result()

        if self.reference is not None:
            result.color_accuracy = self.assess_color_accuracy(test_image, self.reference)
        else:
            lab = rgb_to_lab(test_image)
            L_mean = float(np.mean(lab[:, :, 0]))
            a_mean = float(np.mean(lab[:, :, 1]))
            b_mean = float(np.mean(lab[:, :, 2]))
            chroma = float(np.sqrt(a_mean ** 2 + b_mean ** 2))
            de_fake = min(chroma * 2, 30)
            score = max(0.0, 100 - de_fake * 5)
            result.color_accuracy = {
                'mean_de': round(de_fake, 2),
                'neutral_chroma': round(chroma, 2),
                'score': round(score, 1),
                'level': 'good' if score > 80 else 'acceptable' if score > 50 else 'poor',
                'pass_iso16505': score > 50,
                'note': '无参考图像，使用色度中性评估'
            }

        result.sharpness = self.assess_sharpness(test_image)
        result.noise = self.assess_noise(test_image)

        c = result.color_accuracy.get('score', 0)
        s = result.sharpness.get('acuity_score', 0)
        n = result.noise.get('noise_score', 0)
        result.overall_score = round(c * 0.3 + s * 0.4 + n * 0.3, 1)

        result.compliant = (
            result.color_accuracy.get('pass_iso16505', False)
            and result.sharpness.get('pass_iso16505', False)
            and result.noise.get('pass_iso16505', False)
        )

        if result.overall_score >= 90:
            result.level = 'excellent'
        elif result.overall_score >= 75:
            result.level = 'good'
        elif result.overall_score >= 50:
            result.level = 'acceptable'
        else:
            result.level = 'poor'

        return result

    def to_dict(self, result: ISO16505Result) -> Dict[str, Any]:
        """将结果转换为字典"""
        return {
            'overall_score': result.overall_score,
            'compliant': result.compliant,
            'level': result.level,
            'color_accuracy': result.color_accuracy,
            'sharpness': result.sharpness,
            'noise': result.noise,
        }


# =============================================================================
# CLI 便捷函数
# =============================================================================

def assess_iso16505(
    test_image_path: str,
    reference_image_path: Optional[str] = None,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """便捷函数：评估单张图像的ISO 16505合规性"""
    import json

    test_img = cv2.imread(test_image_path)
    if test_img is None:
        raise ValueError(f"无法读取测试图像: {test_image_path}")

    ref_img = None
    if reference_image_path:
        ref_img = cv2.imread(reference_image_path)
        if ref_img is None:
            raise ValueError(f"无法读取参考图像: {reference_image_path}")

    evaluator = ISO16505Evaluator(ref_img)
    result = evaluator.evaluate(test_img)
    result_dict = evaluator.to_dict(result)

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)
        logger.info(f"ISO 16505评估结果已保存: {output_path}")

    return result_dict
