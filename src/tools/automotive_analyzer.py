#!/usr/bin/env python3
"""
车载场景图像质量评估
针对自动驾驶和智能驾驶舱场景的图像质量分析

符合标准:
- ISO 16505 (Automotive camera systems)
- AEC-Q100 (Automotive electronics)
- IEEE 802.3 (Automotive ethernet)
"""
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
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
    # 扩展字段
    iso16505_compliance: Dict[str, Any] = field(default_factory=dict)
    scene_type: str = 'generic'


class ISO16505Standard:
    """
    ISO 16505 车载摄像头系统标准参数
    
    ISO 16505定义了车载摄像头系统的最低性能要求:
    - 最低分辨率: 1280x960
    - 最低帧率: 25fps
    - 最低动态范围: 100dB
    - 最低水平FOV: 70°
    """
    
    # ISO 16505 最低要求
    MIN_REQUIREMENTS = {
        'resolution': (1280, 960),  # 最低分辨率
        'fps': 25,                   # 最低帧率
        'dynamic_range_db': 100,     # 最低动态范围(dB)
        'horizontal_fov': 70,        # 最低水平FOV(度)
    }
    
    # 推荐要求
    RECOMMENDED = {
        'resolution': (1920, 1080), # 推荐分辨率
        'fps': 30,                   # 推荐帧率
        'dynamic_range_db': 120,     # 推荐动态范围(dB)
        'horizontal_fov': 120,       # 推荐水平FOV(度)
    }
    
    # 优质要求
    PREMIUM = {
        'resolution': (3840, 2160), # 4K
        'fps': 60,                   # 60fps
        'dynamic_range_db': 140,     # 优秀动态范围(dB)
        'horizontal_fov': 150,       # 宽视场角
    }
    
    @classmethod
    def check_compliance(
        cls,
        resolution: Tuple[int, int],
        fps: float,
        dynamic_range_db: float,
        fov: float
    ) -> Dict[str, Any]:
        """
        检查是否符合ISO 16505标准
        
        Args:
            resolution: 分辨率 (w, h)
            fps: 帧率
            dynamic_range_db: 动态范围(dB)
            fov: 水平视场角(度)
        
        Returns:
            合规性检查结果
        """
        results = {}
        
        # 分辨率
        min_res = cls.MIN_REQUIREMENTS['resolution']
        if resolution[0] >= min_res[0] and resolution[1] >= min_res[1]:
            results['resolution'] = {
                'status': 'pass',
                'level': 'minimum',
                'actual': resolution,
                'required': min_res
            }
        elif resolution[0] >= cls.RECOMMENDED['resolution'][0]:
            results['resolution'] = {
                'status': 'pass',
                'level': 'premium',
                'actual': resolution,
                'required': cls.RECOMMENDED['resolution']
            }
        else:
            results['resolution'] = {
                'status': 'fail',
                'level': 'below_minimum',
                'actual': resolution,
                'required': min_res
            }
        
        # 帧率
        min_fps = cls.MIN_REQUIREMENTS['fps']
        if fps >= cls.PREMIUM['fps']:
            results['fps'] = {'status': 'pass', 'level': 'premium', 'actual': fps}
        elif fps >= min_fps:
            results['fps'] = {'status': 'pass', 'level': 'minimum', 'actual': fps}
        else:
            results['fps'] = {'status': 'fail', 'level': 'below_minimum', 'actual': fps}
        
        # 动态范围
        min_dr = cls.MIN_REQUIREMENTS['dynamic_range_db']
        if dynamic_range_db >= cls.PREMIUM['dynamic_range_db']:
            results['dynamic_range'] = {'status': 'pass', 'level': 'premium', 'actual_db': dynamic_range_db}
        elif dynamic_range_db >= min_dr:
            results['dynamic_range'] = {'status': 'pass', 'level': 'minimum', 'actual_db': dynamic_range_db}
        else:
            results['dynamic_range'] = {'status': 'fail', 'level': 'below_minimum', 'actual_db': dynamic_range_db}
        
        # FOV
        min_fov = cls.MIN_REQUIREMENTS['horizontal_fov']
        if fov >= cls.RECOMMENDED['horizontal_fov']:
            results['fov'] = {'status': 'pass', 'level': 'premium', 'actual': fov}
        elif fov >= min_fov:
            results['fov'] = {'status': 'pass', 'level': 'minimum', 'actual': fov}
        else:
            results['fov'] = {'status': 'fail', 'level': 'below_minimum', 'actual': fov}
        
        # 总体评估
        all_pass = all(r['status'] == 'pass' for r in results.values())
        premium_count = sum(1 for r in results.values() if r.get('level') == 'premium')
        
        overall_level = 'premium' if premium_count == 4 else 'standard' if all_pass else 'non_compliant'
        
        return {
            'compliant': all_pass,
            'overall_level': overall_level,
            'premium_count': premium_count,
            'details': results
        }


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
    
    def analyze_with_iso_compliance(
        self,
        image_path: str,
        scene_type: str = 'adas_front',
        resolution: tuple = None,
        fps: float = None,
        fov: float = None
    ) -> Dict[str, Any]:
        """
        增强的车载图像质量分析 (ISO 16505合规版)
        
        Args:
            image_path: 图像路径
            scene_type: 场景类型
            resolution: 分辨率 (w, h)
            fps: 帧率
            fov: 水平视场角(度)
        
        Returns:
            Dict: 包含ISO 16505合规性检查的完整分析结果
        """
        logger.info(f"开始ISO 16505合规分析: {scene_type}")
        
        # 基础分析
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        height, width = img.shape[:2]
        actual_resolution = (width, height)
        
        # 如果未提供分辨率，使用图像实际分辨率
        if resolution is None:
            resolution = actual_resolution
        
        # 评估各项指标
        resolution_score = self._evaluate_resolution(width, height, scene_type)
        framerate_score = self._evaluate_framerate(fps, scene_type)
        fov_score = self._evaluate_fov(fov, scene_type)
        night_score = self._evaluate_night_vision(img, scene_type)
        hdr_score = self._evaluate_hdr(img)
        motion_score = self._evaluate_motion_blur(img)
        
        # 计算动态范围(dB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        p1, p99 = np.percentile(gray, [1, 99])
        # stops = log2(max/min) -> dB = stops * 6 (近似)
        if p1 > 0:
            stops = np.log2(p99 / max(p1, 1))
            dynamic_range_db = stops * 6.02  # 转换为dB
        else:
            dynamic_range_db = 0
        
        # ISO 16505合规性检查
        iso_compliance = ISO16505Standard.check_compliance(
            resolution,
            fps or 30,  # 默认30fps
            dynamic_range_db,
            fov or 120  # 默认120°FOV
        )
        
        # 综合评分 (权重根据场景调整)
        if scene_type == 'adas_front':
            weights = {
                'resolution': 0.20, 'framerate': 0.15, 'fov': 0.10,
                'night': 0.20, 'hdr': 0.25, 'motion': 0.10,
            }
        elif scene_type == 'dms':
            weights = {
                'resolution': 0.25, 'framerate': 0.15, 'fov': 0.05,
                'night': 0.25, 'hdr': 0.10, 'motion': 0.20,
            }
        elif scene_type == 'surround':
            weights = {
                'resolution': 0.15, 'framerate': 0.10, 'fov': 0.20,
                'night': 0.20, 'hdr': 0.20, 'motion': 0.15,
            }
        else:
            weights = {
                'resolution': 0.15, 'framerate': 0.15, 'fov': 0.10,
                'night': 0.25, 'hdr': 0.20, 'motion': 0.15,
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
        
        # 添加ISO合规建议
        if not iso_compliance['compliant']:
            for item, details in iso_compliance['details'].items():
                if details['status'] == 'fail':
                    recommendations.append(
                        f"ISO 16505不合规 - {item}: "
                        f"当前{details.get('actual', details.get('actual_db', 'N/A'))}, "
                        f"需要{details['required'] if 'required' in details else ISO16505Standard.MIN_REQUIREMENTS.get(item, 'N/A')}"
                    )
        elif iso_compliance['overall_level'] == 'premium':
            recommendations.insert(0, "✅ 符合ISO 16505 Premium级别")
        else:
            recommendations.insert(0, "✅ 符合ISO 16505最低要求")
        
        return {
            'overall_score': round(overall, 1),
            'scene_type': scene_type,
            'resolution': {
                'score': round(resolution_score, 1),
                'assessment': self._get_resolution_comment(resolution_score),
                'actual': actual_resolution
            },
            'framerate': {
                'score': round(framerate_score, 1),
                'assessment': self._get_framerate_comment(framerate_score),
                'actual': fps
            },
            'fov': {
                'score': round(fov_score, 1),
                'assessment': self._get_fov_comment(fov_score),
                'actual': fov
            },
            'night_vision': {
                'score': round(night_score, 1),
                'brightness': float(gray.mean()),
                'dynamic_range_estimate_db': round(dynamic_range_db, 1)
            },
            'hdr': {
                'score': round(hdr_score, 1),
                'exposure_range_stops': round(np.log2(max(p99/p1, 1.01)), 1) if p1 > 0 else 0
            },
            'motion_blur': {
                'score': round(motion_score, 1),
                'sharpness_laplacian': round(cv2.Laplacian(gray, cv2.CV_64F).var(), 1)
            },
            'iso16505_compliance': iso_compliance,
            'recommendations': recommendations
        }
    
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


    # ========================================================================
    # Phase 3: 车载场景深度分析功能
    # ========================================================================

    def check_iso_16505_compliance(
        self,
        image: np.ndarray,
        metadata: Dict = None
    ) -> Dict:
        """
        ISO 16505 合规检查 - 纯NumPy实现
        
        检查项:
        - 分辨率 (最低1280x960)
        - 帧率 (最低25fps，在元数据中检查)
        - 水平FOV (≥70°)
        - 动态范围 (≥100dB)
        
        Args:
            image: numpy数组图像 (H, W, C) 或 (H, W)
            metadata: 元数据字典，包含:
                - fps: 帧率
                - fov: 水平视场角(度)
                - sensor_info: 传感器信息
            
        Returns:
            Dict: 结构化合规检查报告
        """
        logger.info("执行ISO 16505合规检查")
        
        h, w = image.shape[:2]
        
        # 1. 分辨率检查
        min_resolution = (1280, 960)
        resolution_pass = (w >= min_resolution[0] and h >= min_resolution[1])
        resolution_score = min(100, 100 * min(w / min_resolution[0], h / min_resolution[1]))
        
        # 2. 帧率检查 (从metadata获取)
        fps = metadata.get('fps', 30) if metadata else 30
        fps_pass = fps >= 25
        fps_score = min(100, 100 * fps / 25)
        
        # 3. 水平FOV检查 (从metadata获取)
        fov = metadata.get('fov', 120) if metadata else 120
        fov_pass = fov >= 70
        fov_score = min(100, 100 * fov / 70) if fov >= 70 else max(0, 50 * fov / 70)
        
        # 4. 动态范围检查 (纯NumPy计算)
        if len(image.shape) == 3:
            # 彩色图像，转灰度计算
            gray = np.mean(image, axis=2).astype(np.float32)
        else:
            gray = image.astype(np.float32)
        
        # 计算动态范围(dB)
        p1, p99 = np.percentile(gray, [1, 99])
        if p1 > 0:
            stops = np.log2(p99 / max(p1, 1))
            dynamic_range_db = stops * 6.02
        else:
            dynamic_range_db = 0
        
        dr_pass = dynamic_range_db >= 100
        dr_score = min(100, 100 * dynamic_range_db / 100)
        
        # 综合合规判定
        all_pass = resolution_pass and fps_pass and fov_pass and dr_pass
        pass_count = sum([resolution_pass, fps_pass, fov_pass, dr_pass])
        
        if all_pass:
            overall_level = 'compliant_minimum'
            if resolution_score >= 90 and dr_score >= 90:
                overall_level = 'compliant_premium'
        else:
            overall_level = 'non_compliant'
        
        return {
            'compliant': all_pass,
            'overall_level': overall_level,
            'checks': {
                'resolution': {
                    'status': 'pass' if resolution_pass else 'fail',
                    'actual': (w, h),
                    'required': min_resolution,
                    'score': round(resolution_score, 1),
                    'detail': f"{w}x{h} {'>=' if resolution_pass else '<'} {min_resolution[0]}x{min_resolution[1]}"
                },
                'framerate': {
                    'status': 'pass' if fps_pass else 'fail',
                    'actual_fps': fps,
                    'required_fps': 25,
                    'score': round(fps_score, 1),
                    'detail': f"{fps}fps {'>=' if fps_pass else '<'} 25fps"
                },
                'horizontal_fov': {
                    'status': 'pass' if fov_pass else 'fail',
                    'actual_fov': fov,
                    'required_fov': 70,
                    'unit': 'degrees',
                    'score': round(fov_score, 1),
                    'detail': f"{fov}° {'>=' if fov_pass else '<'} 70°"
                },
                'dynamic_range': {
                    'status': 'pass' if dr_pass else 'fail',
                    'actual_db': round(dynamic_range_db, 1),
                    'required_db': 100,
                    'unit': 'dB',
                    'score': round(dr_score, 1),
                    'detail': f"{dynamic_range_db:.1f}dB {'>=' if dr_pass else '<'} 100dB"
                }
            },
            'summary': {
                'pass_count': pass_count,
                'total_checks': 4,
                'pass_rate': f"{pass_count}/4"
            }
        }

    def analyze_adas_scene(
        self,
        image: np.ndarray,
        scene_type: str
    ) -> Dict:
        """
        场景自适应分析 - 纯NumPy实现
        
        场景类型:
        - forward_adas: 前视ADAS，重点评估车道线清晰度
        - surround_view: 环视，畸变校正质量
        - dms: 驾驶员监控，红外响应/面部清晰度
        - cabin: 座舱，色彩还原准确性
        
        Args:
            image: numpy数组图像
            scene_type: 场景类型字符串
        
        Returns:
            Dict: 场景分析报告
        """
        logger.info(f"执行{scene_type}场景分析")
        
        h, w = image.shape[:2]
        
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2).astype(np.uint8)
            bgr = image
        else:
            gray = image
            bgr = np.stack([image, image, image], axis=2)
        
        result = {
            'scene_type': scene_type,
            'image_size': (h, w),
            'metrics': {},
            'scores': {},
            'recommendations': []
        }
        
        if scene_type == 'forward_adas':
            # 前视ADAS: 车道线清晰度评估
            lane_score = self._evaluate_lane_clarity(gray)
            edge_score = self._evaluate_edge_sharpness(gray)
            contrast_score = self._evaluate_scene_contrast(gray)
            
            overall = 0.4 * lane_score + 0.35 * edge_score + 0.25 * contrast_score
            
            result['metrics'] = {
                'lane_clarity': round(lane_score, 2),
                'edge_sharpness': round(edge_score, 2),
                'scene_contrast': round(contrast_score, 2)
            }
            result['scores'] = {
                'overall': round(overall, 1),
                'grade': self._score_to_grade(overall)
            }
            
            if lane_score < 60:
                result['recommendations'].append("车道线清晰度不足，建议增强边缘增强滤波")
            if edge_score < 60:
                result['recommendations'].append("图像锐度不足，建议调整 sharpening 参数")
            if overall >= 80:
                result['recommendations'].insert(0, "✅ 前视ADAS场景图像质量良好")
                
        elif scene_type == 'surround_view':
            # 环视: 畸变校正质量评估
            distortion_score = self._evaluate_distortion_quality(gray, w, h)
            uniformity_score = self._evaluate_parking_view_uniformity(bgr)
            edge_distortion_score = self._evaluate_edge_distortion(gray, w, h)
            
            overall = 0.4 * distortion_score + 0.3 * uniformity_score + 0.3 * edge_distortion_score
            
            result['metrics'] = {
                'distortion_quality': round(distortion_score, 2),
                'view_uniformity': round(uniformity_score, 2),
                'edge_distortion': round(edge_distortion_score, 2)
            }
            result['scores'] = {
                'overall': round(overall, 1),
                'grade': self._score_to_grade(overall)
            }
            
            if distortion_score < 60:
                result['recommendations'].append("畸变校正质量不佳，建议重新标定相机")
            if edge_distortion_score < 60:
                result['recommendations'].append("边缘区域畸变明显，建议检查镜头状态")
            if overall >= 80:
                result['recommendations'].insert(0, "✅ 环视场景图像质量良好")
                
        elif scene_type == 'dms':
            # 驾驶员监控: 红外响应/面部清晰度
            face_clarity = self._evaluate_face_clarity(gray)
            ir_response = self._evaluate_ir_response(bgr)
            illumination = self._evaluate_dms_illumination(gray)
            
            overall = 0.4 * face_clarity + 0.35 * ir_response + 0.25 * illumination
            
            result['metrics'] = {
                'face_clarity': round(face_clarity, 2),
                'ir_response': round(ir_response, 2),
                'illumination_evenness': round(illumination, 2)
            }
            result['scores'] = {
                'overall': round(overall, 1),
                'grade': self._score_to_grade(overall)
            }
            
            if face_clarity < 60:
                result['recommendations'].append("面部清晰度不足，建议使用近红外补光灯")
            if ir_response < 60:
                result['recommendations'].append("红外响应不足，建议检查IR-Cut滤波器状态")
            if overall >= 80:
                result['recommendations'].insert(0, "✅ DMS场景图像质量良好")
                
        elif scene_type == 'cabin':
            # 座舱: 色彩还原准确性
            color_accuracy = self._evaluate_color_accuracy(bgr)
            skin_tone = self._evaluate_skin_tone(bgr)
            white_balance = self._evaluate_white_balance(gray)
            
            overall = 0.4 * color_accuracy + 0.35 * skin_tone + 0.25 * white_balance
            
            result['metrics'] = {
                'color_accuracy': round(color_accuracy, 2),
                'skin_tone_fidelity': round(skin_tone, 2),
                'white_balance': round(white_balance, 2)
            }
            result['scores'] = {
                'overall': round(overall, 1),
                'grade': self._score_to_grade(overall)
            }
            
            if color_accuracy < 60:
                result['recommendations'].append("色彩还原偏差较大，建议重新校准色彩矩阵")
            if skin_tone < 60:
                result['recommendations'].append("肤色还原不准确，建议调整色彩空间转换参数")
            if overall >= 80:
                result['recommendations'].insert(0, "✅ 座舱场景图像质量良好")
        else:
            result['error'] = f"未知场景类型: {scene_type}"
        
        return result

    def _evaluate_lane_clarity(self, gray: np.ndarray) -> float:
        """评估车道线清晰度 - 纯NumPy"""
        # 使用边缘检测+车道线特征提取
        h, w = gray.shape
        
        # Sobel边缘检测
        sobel_x = np.abs(np.diff(gray.astype(np.float32), axis=1))
        sobel_y = np.abs(np.diff(gray.astype(np.float32), axis=0))
        
        # 扩展到相同尺寸
        sobel_x = np.pad(sobel_x, ((0, 0), (0, 1)), mode='edge')
        sobel_y = np.pad(sobel_y, ((0, 1), (0, 0)), mode='edge')
        edges = sobel_x + sobel_y
        
        # 车道线通常在图像下方1/3到2/3区域，接近水平
        roi_y_start = int(h * 0.5)
        roi_y_end = int(h * 0.9)
        roi = edges[roi_y_start:roi_y_end, :]
        
        # 计算边缘密度
        edge_density = np.mean(roi > 20)
        
        # 车道线清晰度评分
        score = min(100, edge_density * 500)
        return max(0, score)

    def _evaluate_edge_sharpness(self, gray: np.ndarray) -> float:
        """评估边缘锐度 - 纯NumPy Laplacian变种"""
        # Laplacian方差
        laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
        
        # 卷积操作
        padded = np.pad(gray.astype(np.float32), 1, mode='edge')
        laplacian = np.zeros_like(gray, dtype=np.float32)
        
        for i in range(1, gray.shape[0] + 1):
            for j in range(1, gray.shape[1] + 1):
                region = padded[i-1:i+2, j-1:j+2]
                laplacian[i-1, j-1] = np.sum(region * laplacian_kernel)
        
        variance = np.var(laplacian)
        
        # 方差越高越清晰
        if variance >= 500:
            return 100.0
        elif variance >= 100:
            return 50 + 50 * (variance - 100) / 400
        else:
            return 50 * variance / 100

    def _evaluate_scene_contrast(self, gray: np.ndarray) -> float:
        """评估场景对比度 - 纯NumPy"""
        p1, p99 = np.percentile(gray, [1, 99])
        contrast = (p99 - p1) / 255.0 * 100
        
        return min(100, contrast * 1.2)

    def _evaluate_distortion_quality(self, gray: np.ndarray, w: int, h: int) -> float:
        """评估畸变校正质量 - 纯NumPy"""
        # 简化的畸变评估：检查边缘区域与中心区域的清晰度差异
        center_y, center_x = h // 2, w // 2
        radius_inner = min(w, h) // 4
        radius_outer = min(w, h) // 2
        
        y_coords, x_coords = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        
        # 中心区域
        center_mask = dist_from_center < radius_inner
        center_score = np.mean(gray[center_mask])
        
        # 边缘区域
        edge_mask = (dist_from_center > radius_inner) & (dist_from_center < radius_outer)
        edge_score = np.mean(gray[edge_mask])
        
        # 理想情况下中心与边缘亮度应该接近
        ratio = min(center_score, edge_score) / max(center_score, edge_score + 1e-6)
        
        return ratio * 100

    def _evaluate_parking_view_uniformity(self, bgr: np.ndarray) -> float:
        """评估环视视图均匀性 - 纯NumPy"""
        # 转换为灰度
        gray = np.mean(bgr, axis=2)
        
        # 分区域计算亮度
        h, w = gray.shape
        n_regions = 3
        region_h, region_w = h // n_regions, w // n_regions
        
        brightnesses = []
        for i in range(n_regions):
            for j in range(n_regions):
                region = gray[i*region_h:(i+1)*region_h, j*region_w:(j+1)*region_w]
                brightnesses.append(np.mean(region))
        
        # 计算亮度均匀性
        brightnesses = np.array(brightnesses)
        std = np.std(brightnesses)
        mean = np.mean(brightnesses)
        
        if mean > 0:
            cv = std / mean  # 变异系数
            uniformity = max(0, 100 - cv * 100)
        else:
            uniformity = 0
        
        return uniformity

    def _evaluate_edge_distortion(self, gray: np.ndarray, w: int, h: int) -> float:
        """评估边缘畸变 - 纯NumPy"""
        # 检查四个角的清晰度
        corners = [
            gray[:h//4, :w//4],           # 左上
            gray[:h//4, -w//4:],          # 右上
            gray[-h//4:, :w//4],          # 左下
            gray[-h//4:, -w//4:],         # 右下
        ]
        
        corner_scores = []
        for corner in corners:
            if corner.size > 0:
                # 计算局部方差作为清晰度指标
                variance = np.var(corner)
                corner_scores.append(min(100, variance / 10))
        
        avg_corner_score = np.mean(corner_scores) if corner_scores else 50
        
        # 中心清晰度
        center = gray[h//4:3*h//4, w//4:3*w//4]
        center_variance = np.var(center)
        center_score = min(100, center_variance / 10)
        
        # 边缘与中心清晰度比值
        ratio = avg_corner_score / (center_score + 1e-6)
        
        return min(100, ratio * 80)

    def _evaluate_face_clarity(self, gray: np.ndarray) -> float:
        """评估面部清晰度 - 纯NumPy"""
        h, w = gray.shape
        
        # 假设面部在图像中心区域
        face_roi = gray[h//3:2*h//3, w//3:2*w//3]
        
        # 使用Laplacian变种评估清晰度
        sobel_x = np.abs(np.diff(face_roi.astype(np.float32), axis=1))
        sobel_y = np.abs(np.diff(face_roi.astype(np.float32), axis=0))
        
        sobel_x = np.pad(sobel_x, ((0, 0), (0, 1)), mode='edge')
        sobel_y = np.pad(sobel_y, ((0, 1), (0, 0)), mode='edge')
        edges = sobel_x + sobel_y
        
        edge_strength = np.mean(edges)
        score = min(100, edge_strength * 3)
        
        return max(0, score)

    def _evaluate_ir_response(self, bgr: np.ndarray) -> float:
        """评估红外响应 - 纯NumPy"""
        # 红外图像特点：R和B通道响应较低，Gr和Gb通道响应较高
        if bgr.shape[2] == 3:
            b, g, r = bgr[:,:,0], bgr[:,:,1], bgr[:,:,2]
            
            # 计算Gr/(R+B)比值作为红外响应指标
            ir_ratio = np.mean(g) / (np.mean(r) + np.mean(b) + 1e-6)
            
            # 正常可见光图像比值约1:1，红外增强后Gr占比更高
            # 归一化到0-100
            score = min(100, ir_ratio * 50)
        else:
            score = 50
        
        return score

    def _evaluate_dms_illumination(self, gray: np.ndarray) -> float:
        """评估DMS照明均匀性 - 纯NumPy"""
        h, w = gray.shape
        
        # 分9个区域评估
        region_h, region_w = h // 3, w // 3
        brightnesses = []
        
        for i in range(3):
            for j in range(3):
                region = gray[i*region_h:(i+1)*region_h, j*region_w:(j+1)*region_w]
                brightnesses.append(np.mean(region))
        
        brightnesses = np.array(brightnesses)
        
        # 中心区域应该足够亮
        center_brightness = brightnesses[4]
        
        # 均匀性
        std = np.std(brightnesses)
        uniformity = max(0, 100 - std * 2)
        
        # 亮度足够性
        brightness_score = min(100, center_brightness * 0.8)
        
        return (uniformity + brightness_score) / 2

    def _evaluate_color_accuracy(self, bgr: np.ndarray) -> float:
        """评估色彩准确性 - 纯NumPy"""
        # 简化的色彩准确性评估：检查色彩分布
        h, w, c = bgr.shape
        
        # 期望值：自然场景中R、G、B应该有一定的分布
        r_mean, g_mean, b_mean = np.mean(bgr[:,:,2]), np.mean(bgr[:,:,1]), np.mean(bgr[:,:,0])
        
        # 计算色彩平衡
        total = r_mean + g_mean + b_mean + 1e-6
        r_ratio, g_ratio, b_ratio = r_mean/total, g_mean/total, b_mean/total
        
        # 期望接近自然光比例（约0.3:0.4:0.3）
        balance_score = 100 - abs(r_ratio - 0.3) * 100 - abs(g_ratio - 0.4) * 100 - abs(b_ratio - 0.3) * 100
        
        # 色彩饱和度评估
        hsv = self._rgb_to_hsv(bgr)
        saturation = np.mean(hsv[:,:,1])
        saturation_score = 100 - abs(saturation - 0.5) * 100  # 期望饱和度约0.5
        
        return max(0, min(100, (balance_score + saturation_score) / 2))

    def _rgb_to_hsv(self, rgb: np.ndarray) -> np.ndarray:
        """RGB转HSV - 纯NumPy"""
        r, g, b = rgb[:,:,2].astype(np.float32), rgb[:,:,1].astype(np.float32), rgb[:,:,0].astype(np.float32)
        
        maxc = np.maximum(np.maximum(r, g), b)
        minc = np.minimum(np.minimum(r, g), b)
        v = maxc
        
        deltac = maxc - minc
        s = np.where(maxc > 0, deltac / (maxc + 1e-10), 0)
        
        h = np.zeros_like(maxc)
        
        mask_r = (maxc == r) & (deltac > 0)
        mask_g = (maxc == g) & (deltac > 0) & ~mask_r
        mask_b = (maxc == b) & (deltac > 0) & ~mask_r & ~mask_g
        
        h[mask_r] = 60 * (((g[mask_r] - b[mask_r]) / (deltac[mask_r] + 1e-10)) % 6)
        h[mask_g] = 60 * ((b[mask_g] - r[mask_g]) / (deltac[mask_g] + 1e-10) + 2)
        h[mask_b] = 60 * ((r[mask_b] - g[mask_b]) / (deltac[mask_b] + 1e-10) + 4)
        
        h = h / 360.0  # 归一化到0-1
        
        return np.stack([h, s, v/255.0], axis=2)

    def _evaluate_skin_tone(self, bgr: np.ndarray) -> float:
        """评估肤色保真度 - 纯NumPy"""
        # 简化的肤色检测：YCbCr空间
        r, g, b = bgr[:,:,2].astype(np.float32), bgr[:,:,1].astype(np.float32), bgr[:,:,0].astype(np.float32)
        
        # YCbCr转换（简化）
        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
        cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
        
        # 肤色范围检测 (Cb: 77-127, Cr: 133-173)
        skin_mask = (cb >= 77) & (cb <= 127) & (cr >= 133) & (cr <= 173)
        
        if np.sum(skin_mask) > 100:
            # 皮肤区域色彩准确性
            skin_pixels = bgr[skin_mask]
            skin_r, skin_g, skin_b = skin_pixels[:,2], skin_pixels[:,1], skin_pixels[:,0]
            
            # 期望肤色偏向红润
            skin_balance = (np.mean(skin_r) - np.mean(skin_g)) / (np.mean(skin_g) + 1e-6)
            skin_score = min(100, max(0, 70 + skin_balance * 100))
        else:
            skin_score = 50  # 未检测到肤色区域
        
        return skin_score

    def _evaluate_white_balance(self, gray: np.ndarray) -> float:
        """评估白平衡 - 纯NumPy"""
        # 灰度世界假设：平均灰度值应该接近中间灰(128)
        mean_gray = np.mean(gray)
        
        # 偏离128越多，白平衡越差
        deviation = abs(mean_gray - 128) / 128 * 100
        
        score = max(0, 100 - deviation)
        
        return score

    def _score_to_grade(self, score: float) -> str:
        """评分转等级"""
        if score >= 90:
            return 'A (Excellent)'
        elif score >= 80:
            return 'B (Good)'
        elif score >= 70:
            return 'C (Fair)'
        elif score >= 60:
            return 'D (Poor)'
        else:
            return 'F (Fail)'

    def analyze_low_light(self, image: np.ndarray) -> Dict:
        """
        夜间成像专项分析 - 纯NumPy实现
        
        - 低光检测
        - 噪声水平评估
        - 暗部增益分析
        
        Args:
            image: numpy数组图像
        
        Returns:
            Dict: 夜间成像分析报告
        """
        logger.info("执行夜间成像分析")
        
        h, w = image.shape[:2]
        
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2).astype(np.float32)
            bgr = image.astype(np.float32)
        else:
            gray = image.astype(np.float32)
            bgr = np.stack([image, image, image], axis=2)
        
        result = {
            'overall_score': 0,
            'metrics': {},
            'analysis': {},
            'recommendations': []
        }
        
        # 1. 低光检测
        brightness = np.mean(gray)
        brightness_score = self._evaluate_low_light_brightness(brightness)
        
        # 2. 噪声水平评估
        noise_level = self._estimate_noise_numPy(gray)
        noise_score = self._evaluate_noise_score(noise_level)
        
        # 3. 暗部增益分析
        shadow_gain = self._analyze_shadow_gain(gray)
        shadow_score = self._evaluate_shadow_score(shadow_gain)
        
        # HDR/动态范围评估
        dr_score = self._evaluate_low_light_dynamic_range(gray)
        
        # 综合评分
        overall = 0.3 * brightness_score + 0.3 * noise_score + 0.25 * shadow_score + 0.15 * dr_score
        
        result['overall_score'] = round(overall, 1)
        result['metrics'] = {
            'brightness': {
                'value': round(float(brightness), 1),
                'score': round(brightness_score, 1),
                'level': self._get_brightness_level(brightness)
            },
            'noise_level': {
                'value': round(noise_level, 2),
                'score': round(noise_score, 1),
                'level': self._get_noise_level(noise_level)
            },
            'shadow_gain': {
                'value': round(shadow_gain, 2),
                'score': round(shadow_score, 1),
                'analysis': '暗部提升过强' if shadow_gain > 3 else '暗部提升适中' if shadow_gain > 1 else '暗部可能欠曝'
            },
            'dynamic_range_db': {
                'value': round(dr_score * 10, 1),
                'score': round(dr_score, 1)
            }
        }
        result['analysis'] = {
            'overall_level': self._get_night_vision_level(overall),
            'low_light_detected': brightness < 60,
            'high_noise_detected': noise_level > 30,
            'shadow_boost_applied': shadow_gain > 1.5
        }
        
        # 生成建议
        if brightness_score < 60:
            result['recommendations'].append("亮度不足，建议增加曝光时间或使用大光圈镜头")
        if noise_score < 60:
            result['recommendations'].append(f"噪声水平较高({noise_level:.1f})，建议启用3D降噪或增加照明")
        if shadow_gain > 3:
            result['recommendations'].append("暗部增益过高可能导致噪声放大，建议适当控制")
        if dr_score < 40:
            result['recommendations'].append("动态范围不足，建议启用HDR或多帧合成")
        if overall >= 80:
            result['recommendations'].insert(0, "✅ 夜间成像质量良好")
        
        return result

    def _evaluate_low_light_brightness(self, brightness: float) -> float:
        """评估低光场景下的亮度评分"""
        if brightness >= 80:
            return 100.0
        elif brightness >= 60:
            return 80 + 20 * (brightness - 60) / 20
        elif brightness >= 40:
            return 50 + 30 * (brightness - 40) / 20
        else:
            return max(0, 50 * brightness / 40)

    def _estimate_noise_numPy(self, gray: np.ndarray) -> float:
        """纯NumPy噪声估计 - 使用LAV方法"""
        h, w = gray.shape
        
        # 计算局部方差
        variances = []
        block_size = 7
        
        for i in range(block_size, h - block_size, block_size):
            for j in range(block_size, w - block_size, block_size):
                block = gray[i-block_size:i, j-block_size:j]
                mean = np.mean(block)
                var = np.mean((block - mean) ** 2)
                variances.append(var)
        
        if variances:
            median_var = np.median(variances)
            noise = np.sqrt(median_var)
        else:
            noise = 0
        
        return noise

    def _evaluate_noise_score(self, noise: float) -> float:
        """噪声评分"""
        if noise <= 10:
            return 100.0
        elif noise <= 20:
            return 80 + 20 * (20 - noise) / 10
        elif noise <= 40:
            return 50 + 30 * (40 - noise) / 20
        else:
            return max(0, 50 * 40 / noise)

    def _analyze_shadow_gain(self, gray: np.ndarray) -> float:
        """分析暗部增益 - 纯NumPy"""
        # 暗部定义：亮度低于中值的区域
        median = np.median(gray)
        shadow_mask = gray < median
        
        if np.sum(shadow_mask) > 0:
            shadow_mean = np.mean(gray[shadow_mask])
            global_mean = np.mean(gray)
            
            # 增益比
            if shadow_mean > 0:
                gain_ratio = global_mean / shadow_mean
            else:
                gain_ratio = 1.0
        else:
            gain_ratio = 1.0
        
        return gain_ratio

    def _evaluate_shadow_score(self, shadow_gain: float) -> float:
        """暗部评分"""
        if 1.0 <= shadow_gain <= 2.5:
            return 100.0
        elif shadow_gain > 2.5:
            return max(0, 100 - (shadow_gain - 2.5) * 20)
        else:
            return max(0, 50 + 50 * shadow_gain)

    def _evaluate_low_light_dynamic_range(self, gray: np.ndarray) -> float:
        """低光动态范围评估 - 纯NumPy"""
        p1, p5, p95, p99 = np.percentile(gray, [1, 5, 95, 99])
        
        # 可用动态范围
        useful_range = p95 - p5
        
        # 暗部细节（p1-p5的暗部范围）
        shadow_range = p5 - p1
        
        # HDR能力评分
        if useful_range >= 150:
            return 100.0
        elif useful_range >= 100:
            return 70 + 30 * (useful_range - 100) / 50
        elif useful_range >= 50:
            return 40 + 30 * (useful_range - 50) / 50
        else:
            return 40 * useful_range / 50

    def _get_brightness_level(self, brightness: float) -> str:
        """亮度等级描述"""
        if brightness >= 80:
            return '明亮'
        elif brightness >= 60:
            return '适中'
        elif brightness >= 40:
            return '偏暗'
        else:
            return '低光'

    def _get_noise_level(self, noise: float) -> str:
        """噪声等级描述"""
        if noise <= 10:
            return '极低'
        elif noise <= 20:
            return '良好'
        elif noise <= 40:
            return '中等'
        else:
            return '较高'

    def _get_night_vision_level(self, score: float) -> str:
        """夜间成像等级"""
        if score >= 90:
            return '优秀'
        elif score >= 80:
            return '良好'
        elif score >= 70:
            return '一般'
        elif score >= 60:
            return '较差'
        else:
            return '不合格'


def analyze_automotive_quality(
    image_path: str,
    scene_type: str = 'adas_front',
    **kwargs
) -> AutomotiveQualityResult:
    """便捷函数"""
    analyzer = AutomotiveQualityAnalyzer()
    return analyzer.analyze(image_path, scene_type, **kwargs)
