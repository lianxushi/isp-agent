#!/usr/bin/env python3
"""
车载场景图像质量评估
针对自动驾驶和智能驾驶舱场景的图像质量分析
"""
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional
from ..utils.logger import setup_logger

logger = setup_logger('isp-agent.automotive')


@dataclass
class AutomotiveQualityResult:
    """车载图像质量评估结果"""
    overall_score: float  # 0-100
    resolution_assessment: str
    framerate_assessment: str
    fov_assessment: str
    night_vision_score: float  # 0-100
    hdr_score: float  # 0-100
    motion_blur_score: float  # 0-100
    recommendations: list


class AutomotiveQualityAnalyzer:
    """
    车载图像质量分析器
    
    评估场景：
    - 前视摄像头（ADAS/自动驾驶）
    - 环视摄像头（泊车辅助）
    - 驾驶员监控（DMS）
    - 座舱交互摄像头
    """
    
    # 车载摄像头规格标准
    ADAS_STANDARDS = {
        'min_resolution': (1920, 1080),  # 1080P minimum
        'preferred_resolution': (3840, 2160),  # 4K preferred
        'min_fps': 30,
        'preferred_fps': 60,
        'min_fov': 100,  # 水平FOV
        'max_fov': 180,
    }
    
    NIGHT_VISION_THRESHOLDS = {
        'brightness_min': 80,
        'noise_max': 50,
        'dynamic_range_min': 150,
    }
    
    def __init__(self):
        pass
    
    def analyze(
        self,
        image_path: str,
        scene_type: str = 'adas_front',
        resolution: tuple = None,
        fps: float = None,
        fov: float = None
    ) -> AutomotiveQualityResult:
        """
        车载图像质量分析
        
        Args:
            image_path: 图像路径
            scene_type: 场景类型:
                - 'adas_front': 前视ADAS
                - 'adas_rear': 后视ADAS
                - 'surround': 环视
                - 'dms': 驾驶员监控
                - 'cockpit': 座舱交互
            resolution: 图像分辨率 (width, height)
            fps: 帧率
            fov: 水平视场角(度)
        
        Returns:
            AutomotiveQualityResult: 评估结果
        """
        logger.info(f"开始车载场景分析: {scene_type}")
        
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        height, width = img.shape[:2]
        
        # 各维度评分
        resolution_score = self._evaluate_resolution(width, height, scene_type)
        framerate_score = self._evaluate_framerate(fps, scene_type)
        fov_score = self._evaluate_fov(fov, scene_type)
        night_score = self._evaluate_night_vision(img, scene_type)
        hdr_score = self._evaluate_hdr(img)
        motion_score = self._evaluate_motion_blur(img)
        
        # 综合评分
        weights = {
            'resolution': 0.15,
            'framerate': 0.15,
            'fov': 0.10,
            'night': 0.25,
            'hdr': 0.20,
            'motion': 0.15,
        }
        
        overall = (
            resolution_score * weights['resolution'] +
            framerate_score * weights['framerate'] +
            fov_score * weights['fov'] +
            night_score * weights['night'] +
            hdr_score * weights['hdr'] +
            motion_score * weights['motion']
        )
        
        # 生成建议
        recommendations = self._generate_recommendations(
            resolution_score, framerate_score, fov_score,
            night_score, hdr_score, motion_score, scene_type
        )
        
        return AutomotiveQualityResult(
            overall_score=round(overall, 1),
            resolution_assessment=self._get_resolution_comment(resolution_score),
            framerate_assessment=self._get_framerate_comment(framerate_score),
            fov_assessment=self._get_fov_comment(fov_score),
            night_vision_score=round(night_score, 1),
            hdr_score=round(hdr_score, 1),
            motion_blur_score=round(motion_score, 1),
            recommendations=recommendations
        )
    
    def _evaluate_resolution(self, width: int, height: int, scene_type: str) -> float:
        """评估分辨率是否满足要求"""
        min_w, min_h = self.ADAS_STANDARDS['min_resolution']
        pref_w, pref_h = self.ADAS_STANDARDS['preferred_resolution']
        
        pixel_count = width * height
        min_pixels = min_w * min_h
        pref_pixels = pref_w * pref_h
        
        if pixel_count >= pref_pixels:
            return 100.0
        elif pixel_count >= min_pixels:
            return 70.0 + 30 * (pixel_count - min_pixels) / (pref_pixels - min_pixels)
        else:
            return 50 * pixel_count / min_pixels
    
    def _evaluate_framerate(self, fps: float, scene_type: str) -> float:
        """评估帧率是否满足要求"""
        if fps is None:
            return 50.0  # 未知
        
        min_fps = self.ADAS_STANDARDS['min_fps']
        pref_fps = self.ADAS_STANDARDS['preferred_fps']
        
        if fps >= pref_fps:
            return 100.0
        elif fps >= min_fps:
            return 70.0 + 30 * (fps - min_fps) / (pref_fps - min_fps)
        else:
            return 50 * fps / min_fps
    
    def _evaluate_fov(self, fov: float, scene_type: str) -> float:
        """评估视场角是否满足要求"""
        if fov is None:
            return 50.0  # 未知
        
        min_fov = self.ADAS_STANDARDS['min_fov']
        max_fov = self.ADAS_STANDARDS['max_fov']
        
        if fov < min_fov:
            return 50 * fov / min_fov
        elif fov > max_fov:
            # FOV过大可能导致边缘畸变
            return max(0, 100 - 10 * (fov - max_fov))
        else:
            # 最佳范围
            ideal = 120  # 最佳FOV
            return 100 - abs(fov - ideal) * 0.5
    
    def _evaluate_night_vision(self, img, scene_type: str) -> float:
        """评估夜间成像质量"""
        # 转换为灰度
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 亮度评估
        brightness = gray.mean()
        
        # 噪声评估
        noise = self._estimate_noise(gray)
        
        # 动态范围
        dynamic_range = gray.max() - gray.min()
        
        # 评分
        score = 100.0
        
        # 亮度评分
        if brightness < self.NIGHT_VISION_THRESHOLDS['brightness_min']:
            score -= (self.NIGHT_VISION_THRESHOLDS['brightness_min'] - brightness) * 0.5
        
        # 噪声评分
        if noise > self.NIGHT_VISION_THRESHOLDS['noise_max']:
            score -= (noise - self.NIGHT_VISION_THRESHOLDS['noise_max']) * 0.5
        
        # 动态范围评分
        if dynamic_range < self.NIGHT_VISION_THRESHOLDS['dynamic_range_min']:
            score -= (self.NIGHT_VISION_THRESHOLDS['dynamic_range_min'] - dynamic_range) * 0.3
        
        return max(0, score)
    
    def _estimate_noise(self, gray: np.ndarray) -> float:
        """估计图像噪声"""
        # 使用LAV方法
        kernel_size = 7
        mean = cv2.blur(gray.astype(np.float32), (kernel_size, kernel_size))
        variance = cv2.blur((gray.astype(np.float32) - mean) ** 2, (kernel_size, kernel_size))
        noise = float(np.sqrt(np.median(variance)))
        return noise
    
    def _evaluate_hdr(self, img) -> float:
        """评估HDR效果"""
        # 分析动态范围
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        p1, p99 = np.percentile(gray, [1, 99])
        useful_range = p99 - p1
        
        # 理想情况下应该有较大的有用动态范围
        if useful_range >= 200:
            return 100.0
        elif useful_range >= 150:
            return 70.0 + 30 * (useful_range - 150) / 50
        elif useful_range >= 100:
            return 40 + 30 * (useful_range - 100) / 50
        else:
            return 40 * useful_range / 100
    
    def _evaluate_motion_blur(self, img) -> float:
        """评估运动模糊"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        
        # 使用Laplacian方差评估清晰度
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        
        # 高方差 = 清晰，低方差 = 模糊
        if variance >= 500:
            return 100.0
        elif variance >= 100:
            return 50 + 50 * (variance - 100) / 400
        else:
            return 50 * variance / 100
    
    def _get_resolution_comment(self, score: float) -> str:
        if score >= 90:
            return "优秀 - 4K及以上分辨率"
        elif score >= 70:
            return "良好 - 1080P分辨率"
        elif score >= 50:
            return "一般 - 分辨率偏低"
        else:
            return "不足 - 建议升级摄像头"
    
    def _get_framerate_comment(self, score: float) -> str:
        if score >= 90:
            return "优秀 - 60fps及以上"
        elif score >= 70:
            return "良好 - 30fps"
        elif score >= 50:
            return "一般 - 帧率偏低"
        else:
            return "不足 - 建议提升帧率"
    
    def _get_fov_comment(self, score: float) -> str:
        if score >= 90:
            return "优秀 - 视场角合适"
        elif score >= 70:
            return "良好 - 视场角可接受"
        elif score >= 50:
            return "一般 - 视场角偏小/偏大"
        else:
            return "不足 - 建议调整"
    
    def _generate_recommendations(
        self, res_score, fps_score, fov_score,
        night_score, hdr_score, motion_score, scene_type
    ) -> list:
        """生成优化建议"""
        recs = []
        
        if res_score < 70:
            recs.append("建议升级到至少1080P分辨率，前视摄像头推荐4K")
        
        if fps_score < 70:
            recs.append("建议提升帧率到30fps以上，高速场景推荐60fps")
        
        if fov_score < 70:
            recs.append("建议调整视场角，前视推荐120°-150°")
        
        if night_score < 60:
            recs.append("夜间成像质量不足，建议：增加补光灯、调整曝光参数或更换更大光圈镜头")
        
        if hdr_score < 60:
            recs.append("动态范围不足，建议启用HDR模式或多帧合成")
        
        if motion_score < 60:
            recs.append("存在运动模糊，建议：提升快门速度、启用电子防抖(EIS)")
        
        if not recs:
            recs.append("图像质量良好，满足车载使用要求")
        
        return recs
    
    def quick_check(
        self,
        image_path: str,
        resolution: tuple = None,
        fps: float = None
    ) -> Dict[str, Any]:
        """
        快速检查（不需要FOV信息）
        
        Args:
            image_path: 图像路径
            resolution: 分辨率 (w, h)
            fps: 帧率
        
        Returns:
            Dict: 检查结果
        """
        img = cv2.imread(image_path)
        if img is None:
            return {'error': f'无法读取图像: {image_path}'}
        
        h, w = img.shape[:2]
        actual_res = (w, h)
        
        result = self.analyze(
            image_path,
            resolution=resolution or actual_res,
            fps=fps,
            fov=None  # 未知
        )
        
        return {
            'resolution': actual_res,
            'resolution_status': result.resolution_assessment,
            'night_vision_score': result.night_vision_score,
            'hdr_score': result.hdr_score,
            'motion_blur_score': result.motion_blur_score,
            'overall_score': result.overall_score,
            'recommendations': result.recommendations
        }


def analyze_automotive_quality(
    image_path: str,
    scene_type: str = 'adas_front',
    **kwargs
) -> AutomotiveQualityResult:
    """便捷函数"""
    analyzer = AutomotiveQualityAnalyzer()
    return analyzer.analyze(image_path, scene_type, **kwargs)
