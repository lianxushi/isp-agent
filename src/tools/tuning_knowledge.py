#!/usr/bin/env python3
"""
ISP调优知识库
基于规则和经验的调优建议生成
"""
from typing import Dict, Any, List, Optional
import numpy as np
from scipy import ndimage
from ..utils.logger import setup_logger

logger = setup_logger('isp-agent.knowledge')


class ISPTuningKnowledge:
    """
    ISP调优知识库
    
    包含：
    - 场景识别与参数推荐
    - 常见问题诊断
    - 调优建议生成
    """
    
    # 场景参数模板
    SCENE_PRESETS = {
        'landscape': {
            'name': '风景',
            'iso': 100,
            'saturation': 1.2,
            'contrast': 1.1,
            'sharpness': 1.0,
            'denoise': 0.8,
        },
        'portrait': {
            'name': '人像',
            'iso': 200,
            'saturation': 0.9,
            'contrast': 1.0,
            'sharpness': 0.8,
            'denoise': 1.0,
        },
        'night': {
            'name': '夜景',
            'iso': 1600,
            'saturation': 1.1,
            'contrast': 1.2,
            'sharpness': 1.1,
            'denoise': 1.5,
        },
        'sports': {
            'name': '运动',
            'iso': 800,
            'saturation': 1.0,
            'contrast': 1.1,
            'sharpness': 1.3,
            'denoise': 0.7,
        },
        'indoor': {
            'name': '室内',
            'iso': 400,
            'saturation': 1.0,
            'contrast': 1.05,
            'sharpness': 1.0,
            'denoise': 1.2,
        },
        'automotive': {
            'name': '车载',
            'iso': 600,
            'saturation': 1.1,
            'contrast': 1.15,
            'sharpness': 1.2,
            'denoise': 1.0,
        },
    }
    
    # 问题症状与解决方案
    PROBLEM_SOLUTIONS = {
        'too_dark': {
            'symptoms': ['画面偏暗', '暗部细节丢失'],
            'causes': ['曝光不足', 'ISO过低', '快门过快'],
            'solutions': [
                '增加曝光补偿 +0.5~1.0 EV',
                '提高ISO (建议不超过1600)',
                '降低快门速度',
                '启用HDR或多帧合成',
            ]
        },
        'too_bright': {
            'symptoms': ['画面过曝', '高光溢出'],
            'causes': ['曝光过度', 'ISO过高', '快门过慢'],
            'solutions': [
                '减少曝光补偿 -0.5~1.0 EV',
                '降低ISO',
                '提高快门速度',
                '启用自动曝光锁定',
            ]
        },
        'noise': {
            'symptoms': ['噪点明显', '颗粒感强'],
            'causes': ['ISO过高', '降噪不足', '暗光拍摄'],
            'solutions': [
                '降低ISO',
                '增强降噪强度 (LDC 1.2~1.5)',
                '使用多帧降噪',
                '增加环境光照',
            ]
        },
        'blur': {
            'symptoms': ['画面模糊', '细节丢失'],
            'causes': ['抖动', '对焦失败', '快门过慢'],
            'solutions': [
                '启用EIS/OIS防抖',
                '提高快门速度 (1/快门 > 1/焦距*2)',
                '检查对焦系统',
                '使用三脚架',
            ]
        },
        'color_cast': {
            'symptoms': ['色彩偏色', '白平衡不准'],
            'causes': ['白平衡设置错误', '光源色温不匹配'],
            'solutions': [
                '调整白平衡模式 (自动/日光/阴天/钨丝灯)',
                '手动调整色温 (K值)',
                '使用灰卡校准',
                '调整R/B增益',
            ]
        },
        'low_saturation': {
            'symptoms': ['色彩暗淡', '不够鲜艳'],
            'causes': ['饱和度设置低', '拍摄场景本身色彩低'],
            'solutions': [
                '提高饱和度 +0.2~+0.5',
                '提高自然饱和度',
                '启用场景模式',
            ]
        },
        'high_saturation': {
            'symptoms': ['色彩过艳', '颜色溢出'],
            'causes': ['饱和度过高', '色彩空间设置错误'],
            'solutions': [
                '降低饱和度 -0.2~-0.5',
                '检查色彩空间 (sRGB/Rec.709)',
            ]
        },
        'artifact': {
            'symptoms': ['块效应', '振铃', '摩尔纹'],
            'causes': ['压缩过度', '锐化过度', '传感器混叠'],
            'solutions': [
                '降低压缩率',
                '调整锐化强度',
                '使用抗混叠滤镜',
                '安装光学低通滤镜',
            ]
        },
    }
    
    # 自动驾驶场景特定建议
    AUTOMOTIVE_RECOMMENDATIONS = {
        'adas_front': {
            'description': '前视ADAS摄像头',
            'priority': ['sharpness', 'dynamic_range', 'framerate'],
            'suggestions': [
                '推荐分辨率: 1920x1080 或更高',
                '帧率: 30fps (高速场景60fps)',
                '启用HDR以应对逆光',
                '锐化适度 (避免边缘伪影)',
                '色彩: 自然还原，无需艺术处理',
            ]
        },
        'adas_rear': {
            'description': '后视ADAS摄像头',
            'priority': ['exposure', 'dynamic_range'],
            'suggestions': [
                '宽动态范围 (100dB+)',
                '应对车库/隧道等大光比场景',
                '帧率: 30fps',
                '夜间补光建议',
            ]
        },
        'surround': {
            'description': '环视摄像头',
            'priority': ['distortion', 'stitching', 'brightness'],
            'suggestions': [
                '低畸变镜头 (桶形畸变<10%)',
                '亮度均匀性',
                '拼接边缘融合优化',
                '帧率: 15-30fps',
            ]
        },
        'dms': {
            'description': '驾驶员监控摄像头',
            'priority': ['sharpness', 'ir_response', 'low_light'],
            'suggestions': [
                '近红外响应 (940nm)',
                '高锐度用于面部识别',
                '逆光补偿',
                '帧率: 25-30fps',
                '隐私保护处理',
            ]
        },
    }
    
    def __init__(self):
        pass
    
    def get_preset(self, scene_type: str) -> Optional[Dict[str, Any]]:
        """获取场景参数模板"""
        return self.SCENE_PRESETS.get(scene_type)
    
    def diagnose(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        基于分析结果诊断问题
        
        Args:
            analysis_result: 图像分析结果 (来自ImageAnalyzer)
        
        Returns:
            Dict: 诊断结果和建议
        """
        issues = []
        solutions = []
        
        # 亮度问题
        brightness = analysis_result.get('brightness')
        if brightness:
            if brightness < 50:
                issues.append('too_dark')
            elif brightness > 200:
                issues.append('too_bright')
        
        # 噪声问题
        noise = analysis_result.get('noise_level')
        if noise and noise > 30:
            issues.append('noise')
        
        # 清晰度问题
        contrast = analysis_result.get('contrast')
        if contrast and contrast < 30:
            issues.append('blur')
        
        # 色彩问题
        color_analysis = analysis_result.get('color_analysis')
        if color_analysis:
            wb = color_analysis.get('white_balance', '')
            if '偏' in wb:
                issues.append('color_cast')
            
            sat = color_analysis.get('saturation', 100)
            if sat < 40:
                issues.append('low_saturation')
            elif sat > 180:
                issues.append('high_saturation')
        
        # 动态范围
        dynamic_range = analysis_result.get('dynamic_range')
        if dynamic_range:
            useful_range = dynamic_range.get('useful_range', 0)
            if useful_range < 100:
                issues.append('low_dynamic_range')
        
        # 生成建议
        for issue in issues:
            if issue in self.PROBLEM_SOLUTIONS:
                sol = self.PROBLEM_SOLUTIONS[issue]
                solutions.append({
                    'issue': sol['symptoms'][0],
                    'causes': sol['causes'],
                    'solutions': sol['solutions']
                })
        
        return {
            'issues_found': len(issues),
            'issues': issues,
            'recommendations': solutions,
        }
    
    def get_automotive_recommendations(
        self,
        scene_type: str,
        quality_scores: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """获取车载场景特定建议"""
        if scene_type not in self.AUTOMOTIVE_RECOMMENDATIONS:
            return {'error': f'未知场景类型: {scene_type}'}
        
        rec = self.AUTOMOTIVE_RECOMMENDATIONS[scene_type]
        
        # 如果有质量评分，生成针对性建议
        specific = []
        if quality_scores:
            scores = quality_scores
            
            if scores.get('sharpness', 100) < 70:
                specific.append('⚠️ 清晰度不足，建议增加锐化强度')
            
            if scores.get('noise', 100) < 60:
                specific.append('⚠️ 噪声明显，建议增强降噪或增加照明')
            
            if scores.get('dynamic_range', 100) < 60:
                specific.append('⚠️ 动态范围不足，建议启用HDR模式')
        
        return {
            'scene_type': scene_type,
            'description': rec['description'],
            'priority': rec['priority'],
            'general_suggestions': rec['suggestions'],
            'specific_suggestions': specific,
        }
    
    def generate_tuning_suggestions(
        self,
        analysis_result: Dict[str, Any],
        scene_type: str = 'auto'
    ) -> Dict[str, Any]:
        """
        生成综合调优建议
        
        Args:
            analysis_result: 图像分析结果
            scene_type: 场景类型
        
        Returns:
            Dict: 调优建议
        """
        # 诊断问题
        diagnosis = self.diagnose(analysis_result)
        
        # 获取场景预设
        preset = self.get_preset(scene_type) if scene_type != 'auto' else None
        
        # 生成参数建议
        params_suggestion = {}
        
        if preset:
            params_suggestion = {
                'iso': preset.get('iso'),
                'saturation': preset.get('saturation'),
                'contrast': preset.get('contrast'),
                'sharpness': preset.get('sharpness'),
                'denoise': preset.get('denoise'),
            }
        
        return {
            'diagnosis': diagnosis,
            'scene_preset': preset,
            'suggested_params': params_suggestion,
            'overall_assessment': self._generate_assessment(diagnosis),
        }
    
    def _generate_assessment(self, diagnosis: Dict[str, Any]) -> str:
        """生成总体评估"""
        issues_count = diagnosis.get('issues_found', 0)
        
        if issues_count == 0:
            return "图像质量良好，无需重大调整"
        elif issues_count == 1:
            return "发现1个问题，建议按上述方案调整"
        elif issues_count <= 3:
            return f"发现{issues_count}个问题，建议逐一排查调整"
        else:
            return f"发现{issues_count}个问题，建议系统性地检查ISP pipeline各模块"

    # ------------------------------------------------------------------
    # Phase 4: Bayer / Demosaic 问题诊断
    # ------------------------------------------------------------------
    def diagnose_bayer_issues(self, image: np.ndarray) -> Dict[str, Any]:
        """
        诊断Bayer阵列/demosaic相关的图像问题
        
        检测项:
        - 摩尔纹 (Moiré): 高频图案与传感器网格混叠
        - 伪彩色 (False Color): demosaic错误导致的异常色彩
        - 拉链效应 (Zipper Artifacts): 边缘处交替的亮暗像素
        - demosaic精度评估: 基于局部一致性和边缘保真度
        
        Args:
            image: RGB图像，shape (H, W, 3)，dtype任意，值范围[0,255]或[0,1]
        
        Returns:
            Dict: 结构化诊断报告
        """
        # 归一化到[0,1]
        img = image.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0
        
        h, w = img.shape[:2]
        report = {
            'moire': self._detect_moire(img),
            'false_color': self._detect_false_color(img),
            'zipper_artifacts': self._detect_zipper_artifacts(img),
            'demosaic_accuracy': self._assess_demosaic_accuracy(img),
        }
        
        # 汇总
        issues = [k for k, v in report.items()
                  if isinstance(v, dict) and v.get('severity', 0) >= 2]
        report['summary'] = {
            'issues_found': len(issues),
            'issues': issues,
            'overall_severity': max([v.get('severity', 0) for v in report.values()
                                     if isinstance(v, dict)]) if report else 0,
        }
        return report

    def _detect_moire(self, img: np.ndarray) -> Dict[str, Any]:
        """检测摩尔纹: 使用频域分析检测高频混叠图案"""
        gray = np.mean(img, axis=2)
        
        # 简化的FFT方法：计算高频能量比
        # 对图像分块，在每个块上计算高频分量
        from scipy.fft import fft2, fftshift
        
        # 使用较小的块避免整幅图像的巨大FFT
        block_size = min(256, gray.shape[0], gray.shape[1])
        if gray.shape[0] < block_size or gray.shape[1] < block_size:
            return {'severity': 0, 'score': 0.0, 'detail': '图像太小，跳过摩尔纹检测'}
        
        # 取中心块
        r0, c0 = gray.shape[0] // 2, gray.shape[1] // 2
        block = gray[r0 - block_size // 2:r0 + block_size // 2,
                     c0 - block_size // 2:c0 + block_size // 2]
        
        fft = fftshift(fft2(block))
        mag = np.abs(fft)
        
        # 计算总能量和中心附近低频能量
        total_energy = np.sum(mag ** 2)
        h, w = mag.shape
        cy, cx = h // 2, w // 2
        
        # 定义高频区域（排除中心3x3的低频区域）
        mask = np.ones_like(mag, dtype=bool)
        mask[cy - 1:cy + 2, cx - 1:cx + 2] = False
        hf_energy = np.sum(mag[mask] ** 2)
        lf_energy = np.sum(mag[~mask] ** 2)
        
        hf_ratio = hf_energy / (total_energy + 1e-10)
        
        # 摩尔纹通常在中高频段有尖峰；检测是否为周期性模式
        # 方法：计算径向功率谱的方差
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)
        max_r = min(cx, cy)
        radial_profile = np.zeros(max_r)
        for ri in range(max_r):
            radial_profile[ri] = np.mean(mag[r == ri]) if np.any(r == ri) else 0
        
        # 径向功率谱的方差（周期性=摩尔纹）
        var = np.var(radial_profile[1:])
        mean_val = np.mean(radial_profile[1:]) + 1e-10
        periodicity_score = var / (mean_val ** 2 + 1e-10)
        
        # 合并评分：高频能量比+周期性
        moire_score = float(np.clip((hf_ratio * 10 + periodicity_score * 0.5), 0, 10))
        severity = 0
        if moire_score > 4:
            severity = 3
        elif moire_score > 2.5:
            severity = 2
        elif moire_score > 1.5:
            severity = 1
        
        return {
            'severity': severity,
            'score': round(moire_score, 3),
            'hf_ratio': round(float(hf_ratio), 4),
            'periodicity': round(float(periodicity_score), 4),
            'detail': '摩尔纹明显，建议使用光学低通滤镜或调整传感器配置'
                       if severity >= 2 else '未检测到明显摩尔纹' if severity == 0
                       else '检测到轻微摩尔纹痕迹',
        }

    def _detect_false_color(self, img: np.ndarray) -> Dict[str, Any]:
        """检测伪彩色: demosaic错误导致的异常色偏"""
        # 方法：检测局部色度通道的极端值
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        
        # 计算色度 Cb=blue-Y, Cr=red-Y (简化版)
        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = b - y
        cr = r - y
        
        # 在平坦区域检测色度噪声（demosaic错误在平坦区域最明显）
        # 使用梯度检测平坦区域
        gy, gx = np.gradient(y)
        grad_mag = np.sqrt(gy ** 2 + gx ** 2)
        flat_mask = grad_mag < np.percentile(grad_mag, 30)
        
        if not np.any(flat_mask):
            flat_mask = np.ones_like(grad_mag, dtype=bool)
        
        cb_flat = cb[flat_mask]
        cr_flat = cr[flat_mask]
        
        # 计算色度统计
        cb_std = float(np.std(cb_flat))
        cr_std = float(np.std(cr_flat))
        cb_mean = float(np.mean(np.abs(cb_flat)))
        cr_mean = float(np.mean(np.abs(cr_flat)))
        
        # 色度标准差过大 → 伪彩色
        chroma_score = (cb_std + cr_std) * 5
        severity = 0
        if chroma_score > 3:
            severity = 3
        elif chroma_score > 1.5:
            severity = 2
        elif chroma_score > 0.8:
            severity = 1
        
        return {
            'severity': severity,
            'score': round(chroma_score, 3),
            'cb_std': round(cb_std, 4),
            'cr_std': round(cr_std, 4),
            'detail': '伪彩色明显，建议检查demosaic算法或色彩校正矩阵'
                       if severity >= 2 else '色度正常' if severity == 0
                       else '检测到轻微伪彩色',
        }

    def _detect_zipper_artifacts(self, img: np.ndarray) -> Dict[str, Any]:
        """检测拉链效应: demosaic在边缘产生的交替亮暗像素"""
        gray = np.mean(img, axis=2)
        
        # 在边缘处检测相邻像素差异的交替模式
        # 计算水平梯度
        dx = np.diff(gray, axis=1)  # shape (H, W-1)
        
        # 统计正负梯度的交替频率
        sign_changes = np.diff(np.sign(dx), axis=1)  # 检测sign flip
        # sign_changes != 0 表示交替
        flip_rate = np.mean(np.abs(sign_changes) > 0)
        
        # 也检测垂直方向
        dy = np.diff(gray, axis=0)
        sign_changes_v = np.diff(np.sign(dy), axis=0)
        flip_rate_v = np.mean(np.abs(sign_changes_v) > 0)
        
        avg_flip_rate = (flip_rate + flip_rate_v) / 2
        
        # 拉链效应特征：flip_rate在某个中等范围（太低=正常，太高=噪声）
        # 正常图像的flip_rate约0.2-0.4，过高≈0.7+可能是拉链效应
        zipper_score = float(np.clip((avg_flip_rate - 0.3) * 10, 0, 10))
        
        severity = 0
        if zipper_score > 4:
            severity = 3
        elif zipper_score > 2.5:
            severity = 2
        elif zipper_score > 1.5:
            severity = 1
        
        return {
            'severity': severity,
            'score': round(zipper_score, 3),
            'h_flip_rate': round(float(flip_rate), 4),
            'v_flip_rate': round(float(flip_rate_v), 4),
            'detail': '拉链效应明显，建议使用高级demosaic算法(如AHAT)'
                       if severity >= 2 else '未检测到明显拉链效应' if severity == 0
                       else '检测到轻微拉链效应',
        }

    def _assess_demosaic_accuracy(self, img: np.ndarray) -> Dict[str, Any]:
        """评估demosaic整体精度"""
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        
        # 方法1: 检测边缘处RGB通道的不一致性
        # 边缘通常在绿色通道最清晰（人眼对绿色最敏感，demosaic常优先绿色）
        # 计算RGB通道边缘响应差异
        grad_r = np.sqrt(ndimage.sobel(r, mode='reflect') ** 2 +
                         ndimage.sobel(r, axis=1, mode='reflect') ** 2)
        grad_g = np.sqrt(ndimage.sobel(g, mode='reflect') ** 2 +
                         ndimage.sobel(g, axis=1, mode='reflect') ** 2)
        grad_b = np.sqrt(ndimage.sobel(b, mode='reflect') ** 2 +
                         ndimage.sobel(b, axis=1, mode='reflect') ** 2)
        
        # 边缘处G与R/B差异（理想情况下三者相近）
        edge_mask = (grad_g > np.percentile(grad_g, 70))
        if np.any(edge_mask):
            rg_diff = float(np.mean(np.abs(grad_r[edge_mask] - grad_g[edge_mask])))
            bg_diff = float(np.mean(np.abs(grad_b[edge_mask] - grad_g[edge_mask])))
            channel_inconsistency = float((rg_diff + bg_diff) / 2)
        else:
            channel_inconsistency = 0.0
        
        # 方法2: 局部色度平滑度（平坦区色度应平滑）
        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = b - y
        cr = r - y
        chroma_laplacian = np.abs(ndimage.laplace(cb)) + np.abs(ndimage.laplace(cr))
        chroma_smoothness = float(np.mean(chroma_laplacian))
        
        # 综合评分（越低越好）
        accuracy_score = float(np.clip(
            channel_inconsistency * 3 + chroma_smoothness * 2, 0, 10))
        
        severity = 0
        if accuracy_score > 4:
            severity = 3
        elif accuracy_score > 2.5:
            severity = 2
        elif accuracy_score > 1.5:
            severity = 1
        
        return {
            'severity': severity,
            'score': round(accuracy_score, 3),
            'channel_inconsistency': round(channel_inconsistency, 4),
            'chroma_smoothness': round(chroma_smoothness, 4),
            'detail': 'demosaic精度较低，建议更换算法或检查Bayer CFA配置'
                       if severity >= 2 else 'demosaic精度正常' if severity == 0
                       else 'demosaic精度一般',
        }

    # ------------------------------------------------------------------
    # Phase 4: 降噪参数优化建议
    # ------------------------------------------------------------------
    def suggest_denoise_params(self, image: np.ndarray) -> Dict[str, Any]:
        """
        基于图像内容推荐降噪参数
        
        分析:
        - 噪声水平评估
        - 推荐降噪强度
        - temporal vs spatial NR 策略建议
        
        Args:
            image: RGB图像，shape (H, W, 3)，dtype任意
        
        Returns:
            Dict: 降噪参数建议
        """
        img = image.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0
        
        gray = np.mean(img, axis=2)
        
        # ---- 噪声水平评估 ----
        noise_est = self._estimate_noise_level(gray)
        
        # ---- 场景复杂度（影响降噪策略）----
        scene_complexity = self._assess_scene_complexity(gray)
        
        # ---- 运动程度（影响时域vs空域选择）----
        motion_score = self._estimate_motion_content(gray)
        
        # ---- 综合降噪建议 ----
        strength = self._compute_denoise_strength(noise_est, scene_complexity)
        
        # temporal vs spatial建议
        use_temporal = motion_score < 0.3  # 静态场景用时域降噪
        spatial_strength = strength
        temporal_strength = strength * 0.8 if use_temporal else 0.0
        
        return {
            'noise_level': noise_est,
            'scene_complexity': scene_complexity,
            'motion_score': motion_score,
            'recommendations': {
                'spatial_strength': round(spatial_strength, 2),
                'temporal_strength': round(temporal_strength, 2),
                'use_temporal_nr': use_temporal,
                'spatial_nr_radius': 2 if strength < 1.5 else 3 if strength < 2.5 else 3,
                'temporal_nr_frames': 3 if use_temporal else 0,
                'luminance_strength': round(strength * 1.1, 2),
                'chroma_strength': round(strength * 0.9, 2),
            },
            'strategy': '建议使用时域降噪' if use_temporal else '建议使用空域降噪，静态区域可叠加时域',
        }

    def _estimate_noise_level(self, gray: np.ndarray) -> Dict[str, Any]:
        """估计图像噪声水平（基于平坦区域噪声方差）"""
        # 使用 MAD (Median Absolute Deviation) 估计方差，更鲁棒
        # 在平滑区域估计
        smooth = ndimage.uniform_filter(gray, size=3)
        detail = gray - smooth
        
        # MAD → 标准差估计
        mad = np.median(np.abs(detail - np.median(detail)))
        sigma_est = 1.4826 * mad  # 正态分布因子
        
        # 分通道噪声估计
        noise_db = float(20 * np.log10(sigma_est + 1e-10))
        
        severity = 0
        if sigma_est > 0.05:
            severity = 3
        elif sigma_est > 0.03:
            severity = 2
        elif sigma_est > 0.015:
            severity = 1
        
        return {
            'sigma': round(float(sigma_est), 4),
            'noise_db': round(noise_db, 2),
            'severity': severity,
            'label': '高噪声' if severity >= 3 else '中等噪声' if severity == 2
                     else '低噪声' if severity == 0 else '轻微噪声',
        }

    def _assess_scene_complexity(self, gray: np.ndarray) -> Dict[str, Any]:
        """评估场景复杂度（细节多寡）"""
        # 高通能量占比作为复杂度指标
        smooth = ndimage.uniform_filter(gray, size=5)
        detail = gray - smooth
        hf_energy = np.var(detail)
        total_var = np.var(gray)
        
        complexity_ratio = hf_energy / (total_var + 1e-10)
        
        complexity_score = float(np.clip(complexity_ratio * 50, 0, 10))
        
        return {
            'score': round(complexity_score, 3),
            'hf_energy_ratio': round(float(complexity_ratio), 4),
            'label': '高复杂度' if complexity_score > 5 else '中复杂度'
                     if complexity_score > 2 else '低复杂度',
        }

    def _estimate_motion_content(self, gray: np.ndarray) -> float:
        """估计运动程度（简化版：时域差分需要多帧，这里用空间运动估计）"""
        # 用帧内空间梯度方差近似（真实时域NR需要视频序列）
        gy, gx = np.gradient(gray)
        motion = np.sqrt(gx ** 2 + gy ** 2)
        return float(np.clip(np.mean(motion) * 10, 0, 1))

    def _compute_denoise_strength(
        self,
        noise_est: Dict[str, Any],
        complexity: Dict[str, Any]
    ) -> float:
        """计算推荐降噪强度"""
        sigma = noise_est['sigma']
        comp = complexity['score']
        
        # 基础强度 = 噪声水平
        base = sigma * 20
        
        # 复杂度补偿：细节多的场景不宜过强降噪
        detail_penalty = comp * 0.05
        
        strength = max(0.0, base - detail_penalty)
        return float(np.clip(strength, 0, 4))

    # ------------------------------------------------------------------
    # Phase 4: 锐化伪影诊断
    # ------------------------------------------------------------------
    def diagnose_sharpening_artifacts(self, image: np.ndarray) -> Dict[str, Any]:
        """
        诊断锐化相关伪影
        
        检测项:
        - 振铃效应 (Ringing): 边缘附近的震荡图案
        - 过锐化 (Over-sharpening): 整体锐化过度
        - 边缘伪影 (Edge Artifacts): 边缘处异常
        
        Args:
            image: RGB图像
        
        Returns:
            Dict: 锐化伪影诊断报告
        """
        img = image.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0
        
        gray = np.mean(img, axis=2)
        
        report = {
            'ringing': self._detect_ringing(gray),
            'over_sharpening': self._detect_over_sharpening(gray, img),
            'edge_artifacts': self._detect_edge_artifacts(gray),
        }
        
        issues = [k for k, v in report.items()
                  if isinstance(v, dict) and v.get('severity', 0) >= 2]
        report['summary'] = {
            'issues_found': len(issues),
            'issues': issues,
            'overall_severity': max([v.get('severity', 0) for v in report.values()
                                     if isinstance(v, dict)]),
        }
        return report

    def _detect_ringing(self, gray: np.ndarray) -> Dict[str, Any]:
        """检测振铃效应：边缘附近的明暗交替"""
        # 使用Laplacian检测二阶导震荡
        lap = ndimage.laplace(gray)
        
        # 在边缘附近检测Laplacian符号交替
        # 找到强边缘
        grad = np.sqrt(ndimage.sobel(gray, mode='reflect') ** 2 +
                       ndimage.sobel(gray, axis=1, mode='reflect') ** 2)
        edge_mask = grad > np.percentile(grad, 80)
        
        if not np.any(edge_mask):
            return {'severity': 0, 'score': 0.0, 'detail': '无明显边缘，跳过振铃检测'}
        
        # 在边缘附近检测Laplacian震荡
        dilated_edge = ndimage.binary_dilation(edge_mask, iterations=3)
        ring_zone = dilated_edge & ~edge_mask  # 边缘周围区域
        
        if not np.any(ring_zone):
            ring_zone = dilated_edge  # fallback
        
        lap_ring = lap[ring_zone]
        # 振铃特征：Laplacian绝对值大但均值接近0（正负交替）
        lap_mean = np.mean(np.abs(lap_ring))
        lap_signedness = np.abs(np.mean(lap_ring)) / (lap_mean + 1e-10)
        
        ringing_score = float(np.clip(lap_mean * 50 - (1 - lap_signedness) * 3, 0, 10))
        
        severity = 0
        if ringing_score > 4:
            severity = 3
        elif ringing_score > 2.5:
            severity = 2
        elif ringing_score > 1.5:
            severity = 1
        
        return {
            'severity': severity,
            'score': round(ringing_score, 3),
            'lap_mean': round(float(lap_mean), 5),
            'lap_signedness': round(float(lap_signedness), 4),
            'detail': '振铃效应明显，建议降低锐化强度或使用更平滑的锐化核'
                       if severity >= 2 else '未检测到明显振铃' if severity == 0
                       else '检测到轻微振铃',
        }

    def _detect_over_sharpening(self, gray: np.ndarray, img: np.ndarray) -> Dict[str, Any]:
        """检测过锐化"""
        # 方法1：边缘处overshoot幅度
        grad = np.sqrt(ndimage.sobel(gray, axis=0, mode='reflect') ** 2 +
                       ndimage.sobel(gray, axis=1, mode='reflect') ** 2)
        
        # 检测边缘处的overshoot：峰值前后是否有异常高值
        # 使用形态学reconstruction找局部极值
        from scipy.ndimage import maximum_filter, minimum_filter
        
        max_f = maximum_filter(gray, size=3)
        min_f = minimum_filter(gray, size=3)
        
        # overshoot ratio
        overshoot = np.maximum(gray - max_f, 0)
        undershoot = np.maximum(min_f - gray, 0)
        
        edge_mask = grad > np.percentile(grad, 70)
        if np.any(edge_mask):
            avg_overshoot = float(np.mean(overshoot[edge_mask]))
            avg_undershoot = float(np.mean(undershoot[edge_mask]))
        else:
            avg_overshoot = 0.0
            avg_undershoot = 0.0
        
        # 方法2：整体图像方差与原始细节对比
        smooth = ndimage.uniform_filter(gray, size=3)
        detail = gray - smooth
        detail_energy = np.var(detail)
        
        # 方法3：高频能量过高
        smooth5 = ndimage.uniform_filter(gray, size=5)
        hf = gray - smooth5
        hf_energy = np.var(hf)
        total_energy = np.var(gray) + 1e-10
        hf_ratio = hf_energy / total_energy
        
        oversharp_score = float(np.clip(
            avg_overshoot * 20 + hf_ratio * 30, 0, 10))
        
        severity = 0
        if oversharp_score > 5:
            severity = 3
        elif oversharp_score > 3:
            severity = 2
        elif oversharp_score > 1.5:
            severity = 1
        
        return {
            'severity': severity,
            'score': round(oversharp_score, 3),
            'avg_overshoot': round(avg_overshoot, 5),
            'avg_undershoot': round(avg_undershoot, 5),
            'hf_ratio': round(float(hf_ratio), 4),
            'detail': '过锐化严重，建议降低锐化强度20-30%'
                       if severity >= 2 else '锐化程度适中' if severity == 0
                       else '检测到轻微过锐化倾向',
        }

    def _detect_edge_artifacts(self, gray: np.ndarray) -> Dict[str, Any]:
        """检测边缘伪影"""
        # 检测边缘处的颜色分离或边缘粗细异常
        # 计算边缘宽度（理想边缘应该很细）
        grad = np.sqrt(ndimage.sobel(gray, axis=0, mode='reflect') ** 2 +
                       ndimage.sobel(gray, axis=1, mode='reflect') ** 2)
        
        # 边缘应该很窄；计算边缘有效宽度（梯度非零区域）
        edge_mask = grad > np.percentile(grad, 60)
        
        if not np.any(edge_mask):
            return {'severity': 0, 'score': 0.0, 'detail': '无明显边缘'}
        
        # 计算边缘点数 / 总点数 = 边缘密度
        edge_density = float(np.mean(edge_mask))
        
        # 边缘一致性：边缘处梯度方向的一致性
        gx = ndimage.sobel(gray, axis=1, mode='reflect')
        gy = ndimage.sobel(gray, axis=0, mode='reflect')
        angle = np.arctan2(gy, gx + 1e-10)
        edge_angles = angle[edge_mask]
        if len(edge_angles) > 10:
            # 方向熵（高熵=杂乱=可能有伪影）
            hist, _ = np.histogram(edge_angles, bins=16, range=(-np.pi, np.pi))
            hist_norm = hist / (hist.sum() + 1e-10)
            entropy = -np.sum(hist_norm * np.log(hist_norm + 1e-10))
            max_entropy = np.log(16)
            angle_entropy_ratio = entropy / (max_entropy + 1e-10)
        else:
            angle_entropy_ratio = 0.5
        
        # 边缘伪影评分
        artifact_score = float(np.clip(
            edge_density * 10 + (1 - angle_entropy_ratio) * 5, 0, 10))
        
        severity = 0
        if artifact_score > 5:
            severity = 3
        elif artifact_score > 3:
            severity = 2
        elif artifact_score > 1.5:
            severity = 1
        
        return {
            'severity': severity,
            'score': round(artifact_score, 3),
            'edge_density': round(edge_density, 4),
            'angle_entropy_ratio': round(float(angle_entropy_ratio), 4),
            'detail': '边缘伪影明显，建议检查镜头素质或后处理锐化算法'
                       if severity >= 2 else '边缘质量正常' if severity == 0
                       else '检测到轻微边缘伪影',
        }

    # ------------------------------------------------------------------
    # Phase 4: 色彩空间转换问题诊断
    # ------------------------------------------------------------------
    def diagnose_colorspace_issues(
        self,
        image: np.ndarray,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        诊断色彩空间和转换相关问题
        
        检测项:
        - sRGB vs Adobe RGB 检测
        - 色域溢出 (Gamut Overflow)
        - Gamma转换问题
        
        Args:
            image: RGB图像
            metadata: 可选元数据，包含color_space/gamma/profile等信息
        
        Returns:
            Dict: 色彩空间诊断报告
        """
        img = image.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0
        
        metadata = metadata or {}
        
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        
        report = {
            'colorspace_estimate': self._estimate_colorspace(img, metadata),
            'gamut_overflow': self._detect_gamut_overflow(img),
            'gamma_issues': self._detect_gamma_issues(img),
        }
        
        issues = [k for k, v in report.items()
                  if isinstance(v, dict) and v.get('severity', 0) >= 2]
        report['summary'] = {
            'issues_found': len(issues),
            'issues': issues,
            'overall_severity': max([v.get('severity', 0) for v in report.values()
                                     if isinstance(v, dict)]),
        }
        return report

    def _estimate_colorspace(
        self,
        img: np.ndarray,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """估算图像当前使用的色彩空间（sRGB vs Adobe RGB）"""
        # 如果metadata中有色彩空间信息，直接使用
        if metadata.get('color_space'):
            cs = metadata['color_space'].lower()
            return {
                'detected': cs,
                'confidence': 1.0,
                'detail': f'元数据指定: {cs}',
            }
        
        # 基于色彩饱和度特征推断
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        
        # Adobe RGB色域更宽，同等曝光下饱和度会更高
        # 计算最大通道饱和度分布
        max_rgb = np.maximum(np.maximum(r, g), b)
        min_rgb = np.minimum(np.minimum(r, g), b)
        saturation = (max_rgb - min_rgb) / (max_rgb + 1e-10)
        
        # Adobe RGB在纯色区域饱和度更高
        high_sat_pixels = saturation > 0.5
        if not np.any(high_sat_pixels):
            return {
                'detected': 'srgb',
                'confidence': 0.5,
                'detail': '无法明确区分，默认为sRGB',
            }
        
        avg_sat_high = float(np.mean(saturation[high_sat_pixels]))
        
        # 经验阈值：Adobe RGB的饱和度分布更宽
        if avg_sat_high > 0.85:
            detected = 'adobergb'
            confidence = 0.7
        elif avg_sat_high > 0.75:
            detected = 'dcip3'
            confidence = 0.6
        else:
            detected = 'srgb'
            confidence = 0.7
        
        return {
            'detected': detected,
            'confidence': confidence,
            'avg_saturation': round(avg_sat_high, 4),
            'detail': f'估算色彩空间: {detected} (置信度{confidence:.0%})',
        }

    def _detect_gamut_overflow(self, img: np.ndarray) -> Dict[str, Any]:
        """检测色域溢出：图像中有像素超出目标色域"""
        # 检测标准sRGB色域外的像素
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        
        # 1. 检查通道是否超出[0,1]范围（色域溢出基本指标）
        overflow_basic = np.any(r > 1.0) or np.any(g > 1.0) or np.any(b > 1.0)
        underflow_basic = np.any(r < 0.0) or np.any(g < 0.0) or np.any(b < 0.0)
        
        if overflow_basic or underflow_basic:
            overflow_count = int(np.sum(r > 1.0) + np.sum(g > 1.0) + np.sum(b > 1.0) +
                                  np.sum(r < 0.0) + np.sum(g < 0.0) + np.sum(b < 0.0))
            total_pixels = r.size
            overflow_ratio = overflow_count / (total_pixels * 3)
            severity = 3 if overflow_ratio > 0.01 else 2 if overflow_ratio > 0.001 else 1
            return {
                'severity': severity,
                'score': round(float(np.clip(overflow_ratio * 100, 0, 10)), 3),
                'overflow_ratio': round(float(overflow_ratio), 5),
                'type': 'over_range',
                'detail': '检测到严重色域溢出，建议在色彩空间转换后进行色域映射',
            }
        
        # 2. 检测纯色区域的边界伪影（HSL饱和度超限）
        max_rgb = np.maximum(np.maximum(r, g), b)
        min_rgb = np.minimum(np.minimum(r, g), b)
        l = (max_rgb + min_rgb) / 2
        s = np.where(l > 0.5,
                     (max_rgb - min_rgb) / (2 - max_rgb - min_rgb + 1e-10),
                     (max_rgb - min_rgb) / (max_rgb + min_rgb + 1e-10))
        
        # 完美饱和度=1表示可能超出sRGB色域边界
        over_sat_pixels = s > 0.98
        over_sat_ratio = float(np.mean(over_sat_pixels)) if np.any(over_sat_pixels) else 0.0
        
        severity = 0
        if over_sat_ratio > 0.05:
            severity = 3
        elif over_sat_ratio > 0.01:
            severity = 2
        elif over_sat_ratio > 0.002:
            severity = 1
        
        return {
            'severity': severity,
            'score': round(float(np.clip(over_sat_ratio * 20, 0, 10)), 3),
            'over_sat_ratio': round(over_sat_ratio, 5),
            'type': 'over_saturation',
            'detail': '检测到色域溢出（纯色超饱和），色彩空间转换配置可能有误'
                       if severity >= 2 else '色域正常' if severity == 0
                       else '检测到轻微色域超限',
        }

    def _detect_gamma_issues(self, img: np.ndarray) -> Dict[str, Any]:
        """检测Gamma转换问题"""
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        
        # 计算全局直方图和累积分布
        hist_gray = np.histogram(np.mean(img, axis=2), bins=256, range=(0, 1))[0]
        cdf = np.cumsum(hist_gray)
        cdf_norm = cdf / (cdf[-1] + 1e-10)
        
        # 估算实际Gamma：找斜率=1的点对应的输入值
        # 对于sRGB gamma曲线，输出=输入^2.2时，斜率=1对应的输出≈0.375
        target_out = 0.375
        idx = np.searchsorted(cdf_norm, target_out)
        idx = min(idx, 255)
        
        # 理想sRGB gamma=2.2，对应的输入应该是0.375^(1/2.2) ≈ 0.63
        # 用线性回归估算暗部和中部的gamma
        # 取[0.1, 0.9]区间估算幂律
        low_idx = max(1, int(0.1 * 256))
        high_idx = min(254, int(0.9 * 256))
        
        if high_idx > low_idx:
            x = np.linspace(0.0, 1.0, high_idx - low_idx + 1)
            y = cdf_norm[low_idx:high_idx + 1].astype(float)
            # 避免log(0)
            y_safe = np.clip(y, 1e-10, 1.0)
            # 幂律拟合：log(y) = gamma * log(x)，但CDF不适合幂律，直接用线性
            # 改用亮度梯度方法
            y_clip = np.clip(y, 1e-6, 1.0)
            log_x = np.log(x + 1e-10)
            log_y = np.log(y_clip)
            
            # 最小二乘拟合
            valid = np.isfinite(log_x) & np.isfinite(log_y) & (x > 0.01)
            if np.any(valid):
                coeffs = np.polyfit(log_x[valid], log_y[valid], 1)
                estimated_gamma = float(coeffs[0])
            else:
                estimated_gamma = 2.2
        else:
            estimated_gamma = 2.2
        
        # sRGB标准gamma=2.2，允许±0.3偏差
        gamma_error = abs(estimated_gamma - 2.2)
        
        severity = 0
        if gamma_error > 1.0:
            severity = 3
        elif gamma_error > 0.6:
            severity = 2
        elif gamma_error > 0.3:
            severity = 1
        
        # 额外检测：暗部溢出或截断
        black_clip = np.mean(r < 0.01) + np.mean(g < 0.01) + np.mean(b < 0.01)
        white_clip = np.mean(r > 0.99) + np.mean(g > 0.99) + np.mean(b > 0.99)
        clip_score = float((black_clip + white_clip) / 3.0)
        
        return {
            'severity': max(severity, (3 if clip_score > 0.05 else 2 if clip_score > 0.01 else 1 if clip_score > 0.002 else 0)),
            'score': round(float(np.clip(gamma_error + clip_score * 5, 0, 10)), 3),
            'estimated_gamma': round(estimated_gamma, 3),
            'gamma_error': round(gamma_error, 3),
            'black_clip_ratio': round(float(black_clip / 3.0), 5),
            'white_clip_ratio': round(float(white_clip / 3.0), 5),
            'detail': f'Gamma={estimated_gamma:.2f}，{"严重偏离sRGB标准" if severity >= 2 else "轻微偏差，建议确认输出色彩空间"}'
                       if severity >= 2 else f'Gamma≈{estimated_gamma:.2f}，符合sRGB标准',
        }


    # ------------------------------------------------------------------
    # Phase 5: 场景识别与参数推荐
    # ------------------------------------------------------------------
    def detect_scene_type(self, image: np.ndarray) -> str:
        """
        自动检测场景类型
        
        基于图像特征分析判断场景类型：
        - sunny: 晴天，高亮度、高对比度、饱和度适中
        - cloudy: 阴天，亮度均匀、对比度低、饱和度偏低
        - night: 夜景，整体偏暗、噪声特征
        - indoor: 室内，光线复杂、局部对比度差异大
        
        Args:
            image: RGB图像，shape (H, W, 3)
        
        Returns:
            str: 场景类型 ('sunny', 'cloudy', 'night', 'indoor')
        """
        img = image.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0
        
        gray = np.mean(img, axis=2)
        
        # 计算全局统计
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))
        
        # 饱和度分析
        max_rgb = np.maximum(np.maximum(img[:,:,0], img[:,:,1]), img[:,:,2])
        min_rgb = np.minimum(np.minimum(img[:,:,0], img[:,:,1]), img[:,:,2])
        saturation = np.mean((max_rgb - min_rgb) / (max_rgb + 1e-10))
        
        # 亮度分布特征
        dark_ratio = float(np.mean(gray < 0.2))  # 暗部比例
        bright_ratio = float(np.mean(gray > 0.8))  # 亮部比例
        
        # 噪声估计（夜景判断辅助）
        smooth = ndimage.uniform_filter(gray, size=5)
        noise_estimate = float(np.std(gray - smooth))
        
        # 场景判断规则
        scores = {
            'sunny': 0.0,
            'cloudy': 0.0,
            'night': 0.0,
            'indoor': 0.0,
        }
        
        # 晴天：高亮度、高对比度、饱和度适中
        scores['sunny'] = (
            brightness * 4.0 +            # 高亮度权重（增强）
            contrast * 5.0 +              # 高对比度权重（增强）
            (1.0 - abs(saturation - 0.5) * 2) * 0.5  # 饱和度适中
        )
        
        # 阴天：亮度中等、对比度低、饱和度偏低
        scores['cloudy'] = (
            (1.0 - abs(brightness - 0.5) * 3) * 1.5 +
            (1.0 - min(contrast * 2, 1.0)) * 1.5 +
            (1.0 - min(saturation * 1.5, 1.0)) * 1.0
        )
        
        # 夜景：整体暗、高噪声比
        scores['night'] = (
            (1.0 - brightness) * 3.0 +    # 低亮度
            dark_ratio * 3.0 +            # 大量暗部
            min(noise_estimate * 5.0, 3.0) +  # 噪声明显（有上限）
            (1.0 - min(bright_ratio * 3, 1.0))  # 少亮部
        )
        
        # 室内：亮度变化大、有明显局部高亮区域
        brightness_range = float(np.max(gray) - np.min(gray))
        indoor_score = (
            min(brightness_range * 4.0, 2.5) +   # 亮度分布广（增强）
            contrast * 2.5 +                      # 对比度贡献（增强）
            (1.0 - abs(brightness - 0.4) * 2) * 2.0  # 亮度适中权重增加
        )
        # 室内：不应该太暗（>0.2），且亮度范围明显（>0.3）
        if brightness > 0.2 and brightness_range > 0.3:
            scores['indoor'] = indoor_score
        
        # 选择最高分场景
        detected = max(scores, key=scores.get)
        
        logger.debug(f"场景检测: {scores} -> {detected}")
        
        return detected

    def recommend_params_for_scene(
        self,
        image: np.ndarray,
        scene_type: str = None
    ) -> Dict[str, Any]:
        """
        场景化参数推荐
        
        根据场景类型推荐ISP参数：
        - sunny: 晴天 - 高光强，建议降低对比度
        - cloudy: 阴天 - 光线均匀，建议适当提亮
        - night: 夜景 - 噪声大，建议强降噪
        - indoor: 室内 - 光线复杂，建议自动曝光优化
        
        Args:
            image: RGB图像
            scene_type: 场景类型，如为None则自动检测
        
        Returns:
            Dict: 推荐参数和建议
        """
        # 自动检测场景类型
        if scene_type is None:
            scene_type = self.detect_scene_type(image)
        
        img = image.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0
        
        gray = np.mean(img, axis=2)
        
        # 基于场景类型的参数推荐
        recommendations = {
            'sunny': {
                'description': '晴天场景',
                'params': {
                    'exposure_bias': -0.3,        # 略微降低曝光
                    'contrast': 0.85,             # 降低对比度
                    'highlight_priority': 1.2,    # 优先保护高光
                    'shadow_recovery': 1.1,       # 适度提亮暗部
                    'saturation': 1.05,           # 饱和度适中
                    'denoise': 0.7,               # 降噪可较低
                    'sharpness': 1.0,
                },
                'suggestions': [
                    '高光区域容易过曝，建议启用HDR或调整AE算法',
                    '降低对比度可避免高光溢出',
                    '建议启用局部色调映射(Tone Mapping)',
                ],
                'isp_settings': {
                    'ae_mode': 'center_weighted',  # 中央重点测光
                    'awb_mode': 'daylight',
                    'af_mode': 'continuous',
                },
            },
            'cloudy': {
                'description': '阴天场景',
                'params': {
                    'exposure_bias': 0.3,         # 略微提亮
                    'contrast': 1.1,             # 适度提高对比度
                    'highlight_priority': 1.0,
                    'shadow_recovery': 1.15,      # 适当提亮暗部
                    'saturation': 1.15,           # 略微提高饱和度
                    'denoise': 0.9,
                    'sharpness': 1.05,
                },
                'suggestions': [
                    '光线均匀，可适当提亮整体曝光',
                    '云天色温偏冷，可微调白平衡',
                    '对比度略低，适当增加可提升层次感',
                ],
                'isp_settings': {
                    'ae_mode': 'matrix',
                    'awb_mode': 'cloudy',
                    'af_mode': 'continuous',
                },
            },
            'night': {
                'description': '夜景场景',
                'params': {
                    'exposure_bias': 0.5,         # 曝光补偿
                    'contrast': 1.2,             # 提高对比度
                    'highlight_priority': 1.3,    # 强力保护高光
                    'shadow_recovery': 1.3,       # 提亮暗部
                    'saturation': 1.1,           # 适度饱和
                    'denoise': 1.8,               # 强降噪
                    'sharpness': 0.9,             # 避免噪声放大
                    'temporal_nr': 1.5,           # 时域降噪
                },
                'suggestions': [
                    '噪声明显，建议开启多帧降噪',
                    '暗部细节重要，建议启用暗部增强算法',
                    '如支持，建议使用长曝光多帧合成HDR',
                    '注意抑制彩色噪声(Chroma NR)',
                ],
                'isp_settings': {
                    'ae_mode': 'center_weighted',
                    'awb_mode': 'auto',
                    'af_mode': 'continuous',
                    'long_exposure_nr': True,
                },
            },
            'indoor': {
                'description': '室内场景',
                'params': {
                    'exposure_bias': 0.2,
                    'contrast': 1.05,
                    'highlight_priority': 1.15,
                    'shadow_recovery': 1.2,       # 室内暗部较多
                    'saturation': 1.0,
                    'denoise': 1.2,               # 中等降噪
                    'sharpness': 1.0,
                    'auto_exposure_speed': 0.7,   # 较慢的AE收敛速度
                },
                'suggestions': [
                    '室内光线复杂，建议启用自动曝光优化',
                    '注意白平衡，人工光源可能导致偏色',
                    '如有人物，注意面部曝光',
                    '考虑使用闪光灯或增加环境照明',
                ],
                'isp_settings': {
                    'ae_mode': 'spot',             # 点测光
                    'awb_mode': 'auto',
                    'af_mode': 'continuous',
                    'flicker_detection': True,   # 抑制工频闪烁
                },
            },
        }
        
        if scene_type not in recommendations:
            logger.warning(f"未知场景类型: {scene_type}，使用默认参数")
            scene_type = 'indoor'
        
        rec = recommendations[scene_type]
        
        # 基于图像分析微调参数
        brightness = float(np.mean(gray))
        if brightness < 0.3 and scene_type != 'night':
            # 过暗，微调曝光
            rec['params']['exposure_bias'] = max(rec['params']['exposure_bias'], 0.5)
            rec['suggestions'].append('图像偏暗，建议增加曝光或照明')
        elif brightness > 0.7 and scene_type != 'sunny':
            # 过亮，微调
            rec['params']['exposure_bias'] = min(rec['params']['exposure_bias'], -0.3)
            rec['suggestions'].append('图像偏亮，建议减少曝光或使用ND滤镜')
        
        return {
            'scene_type': scene_type,
            'description': rec['description'],
            'recommended_params': rec['params'],
            'suggestions': rec['suggestions'],
            'isp_settings': rec['isp_settings'],
            'scene_confidence': self._get_scene_confidence(image, scene_type),
        }

    def _get_scene_confidence(self, image: np.ndarray, scene_type: str) -> float:
        """评估场景识别的置信度"""
        img = image.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0
        
        gray = np.mean(img, axis=2)
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))
        
        # 简单置信度评估
        if scene_type == 'night' and brightness < 0.3:
            return 0.9
        elif scene_type == 'sunny' and brightness > 0.6 and contrast > 0.2:
            return 0.85
        elif scene_type == 'indoor' and 0.2 < brightness < 0.6:
            return 0.75
        elif scene_type == 'cloudy':
            return 0.7
        return 0.6

    def recommend_motion_params(
        self,
        image: np.ndarray,
        motion_blur: float = None
    ) -> Dict[str, Any]:
        """
        运动/静止场景参数推荐
        
        分析图像中的运动模糊程度，推荐：
        - 运动场景：建议快门速度、ISO限制
        - 静止场景：允许长曝光
        
        Args:
            image: RGB图像
            motion_blur: 运动模糊估计值(0-1, None则自动估计)
        
        Returns:
            Dict: 运动场景参数建议
        """
        img = image.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0
        
        gray = np.mean(img, axis=2)
        
        # 自动估计运动模糊
        if motion_blur is None:
            motion_blur = self._estimate_motion_blur(gray)
        
        # 边缘锐度作为运动模糊的辅助判断
        edges = np.sqrt(
            ndimage.sobel(gray, axis=0, mode='reflect')**2 +
            ndimage.sobel(gray, axis=1, mode='reflect')**2
        )
        edge_sharpness = float(np.percentile(edges, 90))
        
        # 判断是运动场景还是静止场景
        is_motion_scene = motion_blur > 0.3 or edge_sharpness < 0.1
        
        if is_motion_scene:
            # 运动场景参数
            shutter_speed = max(1.0 / 500, 1.0 / (500 + motion_blur * 2000))
            iso_limit = 1600 if motion_blur < 0.5 else 800
            
            return {
                'scene': 'motion',
                'motion_blur_level': motion_blur,
                'edge_sharpness': round(edge_sharpness, 4),
                'recommendations': {
                    'shutter_speed': f'1/{int(1/shutter_speed)}s 或更快',
                    'shutter_speed_value': round(shutter_speed, 4),
                    'iso_limit': iso_limit,
                    'max_iso': min(iso_limit * 2, 3200),
                    'denoise': 0.8 - motion_blur * 0.3,  # 运动场景降噪适当降低
                    'sharpness': 1.3,                     # 适度锐化弥补
                    'eis_strength': 0.8,                  # 启用电子防抖
                    'exposure_mode': 'shutter_priority',
                },
                'suggestions': [
                    f'检测到运动模糊，建议快门速度 >= 1/{int(1/shutter_speed)}s',
                    f'ISO建议限制在 {iso_limit} 以下',
                    '运动场景降噪不宜过强，避免拖影',
                    '建议使用全局快门传感器减少果冻效应',
                ],
                'freeze_motion': True,
                'allow_long_exposure': False,
            }
        else:
            # 静止场景参数
            return {
                'scene': 'static',
                'motion_blur_level': motion_blur,
                'edge_sharpness': round(edge_sharpness, 4),
                'recommendations': {
                    'shutter_speed': '1/60s 或更长',
                    'shutter_speed_min': 1.0 / 60,
                    'iso_min': 100,
                    'iso_limit': 800,
                    'denoise': 1.2,                   # 静止场景可加强降噪
                    'sharpness': 1.0,
                    'temporal_nr': 1.0,               # 可使用时域降噪
                    'exposure_mode': 'aperture_priority',
                },
                'suggestions': [
                    '静止场景，允许长曝光获取更好画质',
                    '建议使用三脚架配合长曝光',
                    '可启用多帧降噪提升暗部纯净度',
                    '如光线充足，可适当降低ISO',
                ],
                'freeze_motion': False,
                'allow_long_exposure': True,
            }

    def _estimate_motion_blur(self, gray: np.ndarray) -> float:
        """
        估计图像中的运动模糊程度
        
        基于边缘扩散和频率域分析
        """
        # 方法1：边缘宽度估计
        # 模糊图像的边缘会更宽
        edges = np.sqrt(
            ndimage.sobel(gray, axis=0, mode='reflect')**2 +
            ndimage.sobel(gray, axis=1, mode='reflect')**2
        )
        
        # 强边缘的局部梯度变化
        strong_edge_mask = edges > np.percentile(edges, 80)
        if not np.any(strong_edge_mask):
            return 0.0
        
        # 计算边缘处的局部梯度标准差
        edge_std = float(np.std(edges[strong_edge_mask]))
        edge_mean = float(np.mean(edges[strong_edge_mask]))
        
        # 边缘模糊时，梯度变化减小
        # 归一化到[0,1]
        blur_score = 1.0 - min(edge_std / (edge_mean + 1e-10), 1.0)
        
        # 方法2：高频能量比
        from scipy.fft import fft2, fftshift
        h, w = gray.shape
        cy, cx = h // 2, w // 2
        
        # 使用中心块
        block_size = min(256, h, w)
        if h >= block_size and w >= block_size:
            block = gray[cy - block_size//2:cy + block_size//2,
                         cx - block_size//2:cx + block_size//2]
        else:
            block = gray
        
        fft = fftshift(fft2(block))
        mag = np.abs(fft)
        
        # 低频 vs 高频能量比
        r = np.sqrt(np.arange(mag.shape[0])[:, None]**2 + 
                    np.arange(mag.shape[1])[None, :]**2)
        
        lf_energy = float(np.sum(mag[r < 20]**2))
        hf_energy = float(np.sum(mag[r >= 20]**2))
        hf_ratio = hf_energy / (lf_energy + hf_energy + 1e-10)
        
        # 运动模糊降低高频能量
        blur_from_freq = 1.0 - min(hf_ratio * 5, 1.0)
        
        # 综合评分
        final_blur = (blur_score * 0.6 + blur_from_freq * 0.4)
        
        return float(np.clip(final_blur, 0.0, 1.0))

    def recommend_extreme_light_params(self, image: np.ndarray) -> Dict[str, Any]:
        """
        低光/高光环境适配参数推荐
        
        分析环境光照条件，提供针对性参数建议：
        - 低光环境：暗部提亮、噪声抑制
        - 高光环境：高光压缩、动态范围扩展
        
        Args:
            image: RGB图像
        
        Returns:
            Dict: 极端光照环境参数建议
        """
        img = image.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0
        
        gray = np.mean(img, axis=2)
        
        # 分析亮度分布
        dark_ratio = float(np.mean(gray < 0.15))    # 极暗部比例
        shadow_ratio = float(np.mean((gray >= 0.15) & (gray < 0.4)))  # 暗部比例
        mid_ratio = float(np.mean((gray >= 0.4) & (gray < 0.7)))       # 中间调
        highlight_ratio = float(np.mean((gray >= 0.7) & (gray < 0.95)))  # 高光比例
        white_ratio = float(np.mean(gray >= 0.95))   # 纯白/过曝比例
        
        brightness = float(np.mean(gray))
        
        # 判断光照环境
        # 大光比场景：同时存在大量亮部和暗部
        high_contrast_scene = dark_ratio > 0.2 and highlight_ratio > 0.2
        is_low_light = (brightness < 0.35 or dark_ratio > 0.3) and not high_contrast_scene
        is_high_light = (brightness > 0.65 or white_ratio > 0.1) and not high_contrast_scene
        
        result = {
            'environment_type': None,
            'brightness': round(brightness, 4),
            'distribution': {
                'dark': round(dark_ratio, 4),
                'shadow': round(shadow_ratio, 4),
                'mid': round(mid_ratio, 4),
                'highlight': round(highlight_ratio, 4),
                'white': round(white_ratio, 4),
            },
        }
        
        if is_low_light and not is_high_light:
            result['environment_type'] = 'low_light'
            result['recommendations'] = {
                'exposure_bias': 0.5,                # 提亮
                'shadow_recovery': 1.5,               # 强力暗部提亮
                'black_level': -5,                    # 扩展暗部细节
                'denoise': 1.8,                      # 强降噪
                'temporal_nr': 1.5,                   # 时域降噪
                'spatial_nr': 1.3,
                'chroma_denoise': 1.5,               # 彩色噪声抑制
                'sharpness': 0.9,                    # 降噪同时保持细节
                'gamma': 2.0,                        # 提亮暗部
                'hdr_mode': 'multi_frame',            # 多帧HDR
                'long_exposure_gain': 0.8,            # 长曝光增益
            }
            result['suggestions'] = [
                '低光环境，建议启用多帧降噪或长曝光合成',
                '暗部细节重要，建议启用暗部增强算法',
                '噪声抑制关键，推荐使用3D降噪(MDNR)',
                '如支持，建议使用f/1.8或更大光圈',
                '注意抑制彩色噪声，可单独处理Cr/Cb通道',
                '建议增加环境照明或使用补光灯',
            ]
            result['hdr_recommended'] = True
            result['snr_improvement_target'] = 1.5
            
        elif is_high_light and not is_low_light:
            result['environment_type'] = 'high_light'
            result['recommendations'] = {
                'exposure_bias': -0.5,                # 降低曝光
                'highlight_priority': 1.5,            # 强力高光保护
                'tone_mapping': 'local_adaptive',     # 局部自适应色调映射
                'contrast': 0.9,                     # 降低全局对比度
                'dynamic_range_compression': 1.3,    # 动态范围压缩
                'highlight_recovery': 1.2,            # 高光细节恢复
                'shadow_recovery': 1.1,               # 暗部保持
                'saturation': 0.95,
                'sharpness': 1.0,
                'denoise': 0.7,
            }
            result['suggestions'] = [
                '高光环境，建议启用HDR或局部色调映射',
                '优先保护高光细节，避免过曝',
                '动态范围压缩可保留更多高光信息',
                '建议使用偏振镜减少反光',
                '适当降低饱和度避免高光区色彩溢出',
                '中央重点测光或点测光更适合此场景',
            ]
            result['hdr_recommended'] = True
            result['dynamic_range_extension'] = '120dB+'
            
        elif is_low_light and is_high_light:
            # 同时存在极端亮暗 = 大光比场景
            result['environment_type'] = 'high_contrast'
            result['recommendations'] = {
                'exposure_bias': 0.0,
                'hdr_mode': 'multi_frame_hdr',        # 必须使用HDR
                'tone_mapping': 'local_adaptive',
                'highlight_priority': 1.4,
                'shadow_recovery': 1.4,
                'dynamic_range_compression': 1.5,
                'contrast': 0.85,
                'denoise': 1.3,
                'exposure_bracketing': [-1, 0, +1],   # 包围曝光
            }
            result['suggestions'] = [
                '大光比场景，必须使用HDR模式',
                '建议使用包围曝光合成',
                '局部色调映射可同时保留亮暗细节',
                '注意ghost伪影，建议使用运动补偿',
                '如可能，调整拍摄角度避免直射光源',
            ]
            result['hdr_recommended'] = True
            result['bracketing_recommended'] = True
            
        else:
            # 正常光照
            result['environment_type'] = 'normal'
            result['recommendations'] = {
                'exposure_bias': 0.0,
                'highlight_priority': 1.0,
                'shadow_recovery': 1.0,
                'contrast': 1.0,
                'denoise': 1.0,
                'sharpness': 1.0,
            }
            result['suggestions'] = [
                '光照条件正常，保持默认参数即可',
                '如需微调，注意保持亮暗平衡',
            ]
            result['hdr_recommended'] = False
        
        return result


    # ------------------------------------------------------------------
    # Phase 6: 根因分析与症状映射
    # ------------------------------------------------------------------

    # 症状 → ISP模块映射表
    SYMPTOM_MODULE_MAP = {
        '噪声过多': {
            'primary_module': 'denoise',
            'isp_blocks': ['SpatialNR', 'TemporalNR', 'LDC', '3DNR'],
            'causes': ['ISO过高', '曝光不足', '降噪参数过低', '传感器噪声特性'],
            'root_causes': [
                'ISP降噪模块参数配置不当（降噪强度过低）',
                'AE算法曝光时间过短导致sensor gain过高',
                '低光环境下sensor噪声放大',
                'Temporal NR未启用导致时域噪声累积',
            ],
            'severity_impact': 2,
        },
        '噪点明显': {
            'primary_module': 'denoise',
            'isp_blocks': ['SpatialNR', 'TemporalNR', 'ChromaNR', 'LDC'],
            'causes': ['高ISO', '暗光环境', '降噪关闭'],
            'root_causes': [
                '降噪模块未启用或参数过低',
                '色彩噪声（chroma noise）未单独处理',
                'AE增益过高导致噪声放大',
            ],
            'severity_impact': 2,
        },
        '色彩偏蓝': {
            'primary_module': 'ccm',
            'isp_blocks': ['AWB', 'CCM', 'Saturation', 'ColorTemp'],
            'causes': ['白平衡偏冷', '色温设置错误', '光源色温不匹配'],
            'root_causes': [
                'AWB算法将非蓝天误判为光源色温',
                'CCM色彩校正矩阵未针对当前光源优化',
                '场景中大量蓝色物体干扰AWB统计',
                '阴天/背阴场景AWB偏冷偏移未补偿',
            ],
            'severity_impact': 2,
        },
        '色彩偏黄': {
            'primary_module': 'ccm',
            'isp_blocks': ['AWB', 'CCM', 'Saturation', 'ColorTemp'],
            'causes': ['白平衡偏暖', '钨丝灯光源', '夕阳场景'],
            'root_causes': [
                'AWB算法将室内钨丝灯误判为日光',
                'CCM矩阵中R/Gain过高或B/Gain过低',
                '场景以暖色调物体为主干扰AWB',
                '手动白平衡R/B偏移设置不当',
            ],
            'severity_impact': 2,
        },
        '色彩偏红': {
            'primary_module': 'ccm',
            'isp_blocks': ['AWB', 'CCM', 'Saturation'],
            'causes': ['白平衡错误', '色彩校正矩阵偏差', 'R通道增益过高'],
            'root_causes': [
                'CCM矩阵R系数设置过高',
                'AWB R/Gain偏移量设置错误',
                '场景红色物体过多干扰AWB统计',
            ],
            'severity_impact': 2,
        },
        '边缘模糊': {
            'primary_module': 'sharpening',
            'isp_blocks': ['Sharpening', 'EdgeEnhance', 'LDC', 'Demosaic'],
            'causes': ['锐化不足', '镜头聚焦偏差', '去噪过度', '去摩尔纹模糊'],
            'root_causes': [
                'Sharpening模块锐化强度过低',
                'LDC（Lens Distortion Correction）参数不当导致局部模糊',
                '降噪模块降噪过强导致细节损失',
                '去噪声算法在边缘处过度平滑',
                'Demosaic算法在高频边缘处精度不足',
            ],
            'severity_impact': 3,
        },
        '细节丢失': {
            'primary_module': 'sharpening',
            'isp_blocks': ['Sharpening', 'Demosaic', 'LDC', 'NoiseReduction'],
            'causes': ['锐化不足', '降噪过度', '镜头素质', '对焦问题'],
            'root_causes': [
                'Sharpening强度不足或算法不当',
                '降噪模块平滑过度导致高频细节损失',
                'Demosaic精度不足导致细节重建错误',
                '镜头慧差/像散导致边缘细节退化',
            ],
            'severity_impact': 3,
        },
        '摩尔纹': {
            'primary_module': 'demosaic',
            'isp_blocks': ['Demosaic', 'AntiAliasing', 'OpticalLowPass'],
            'causes': ['传感器混叠', '高频图案与Bayer网格干涉', '去马赛克算法不足'],
            'root_causes': [
                'Demosaic算法对高频图案处理能力不足',
                '传感器AA（抗混叠）滤波器过弱或缺失',
                '镜头解析力过高导致奈奎斯特频率以上信息折叠',
                '拍摄高频条纹图案（如布料、建筑物）',
            ],
            'severity_impact': 2,
        },
        '伪彩色': {
            'primary_module': 'demosaic',
            'isp_blocks': ['Demosaic', 'FalseColorSuppression', 'CCM'],
            'causes': ['Demosaic错误', '色度通道异常', '色彩插值错误'],
            'root_causes': [
                'Demosaic算法在边缘处色彩插值错误',
                'False Color抑制参数未开启或过低',
                'CCM色彩空间转换精度不足',
                '边缘处RGB通道不一致性未正确处理',
            ],
            'severity_impact': 2,
        },
        '拉链效应': {
            'primary_module': 'demosaic',
            'isp_blocks': ['Demosaic', 'EdgeAdaptiveDemosaic'],
            'causes': ['Demosaic算法缺陷', '边缘处理不当'],
            'root_causes': [
                'Demosaic算法在锐利边缘处产生交替亮暗像素',
                '未使用边缘自适应Demosaic算法（如AHAT、Variable Number of Gradients）',
                ' Bayer CFA填充模式与算法不匹配',
            ],
            'severity_impact': 2,
        },
        '过曝': {
            'primary_module': 'ae',
            'isp_blocks': ['AE', 'HDR', 'ToneMapping', 'Gamma'],
            'causes': ['曝光过度', 'AE目标过亮', '高光保护不足'],
            'root_causes': [
                'AE算法目标亮度设置过高',
                '高光区域未检测或保护不足',
                'Tone Mapping高光压缩不足',
                'HDR模式未启用导致大光比场景高光溢出',
            ],
            'severity_impact': 3,
        },
        '欠曝': {
            'primary_module': 'ae',
            'isp_blocks': ['AE', 'HDR', 'LongExposure', 'BlackLevel'],
            'causes': ['曝光不足', 'AE目标过暗', '暗部提亮不足'],
            'root_causes': [
                'AE目标亮度设置过低',
                '暗部提亮（Shadow Recovery）参数不足',
                '长曝光多帧合成未启用',
                'HDR模式未启用导致大光比场景暗部死黑',
            ],
            'severity_impact': 3,
        },
        '动态范围不足': {
            'primary_module': 'hdr',
            'isp_blocks': ['HDR', 'ToneMapping', 'LocalTM', 'Gamma'],
            'causes': ['HDR未启用', 'ToneMapping不当', '对比度过高'],
            'root_causes': [
                'HDR多帧合成未启用',
                '局部色调映射（Local Tone Mapping）参数不当',
                '全局Gamma曲线压缩过度',
                '高光恢复（Highlight Recovery）参数不足',
            ],
            'severity_impact': 3,
        },
        '对比度低': {
            'primary_module': 'gamma',
            'isp_blocks': ['Gamma', 'Contrast', 'LocalTM', 'CLAHE'],
            'causes': ['Gamma曲线过平', '对比度参数过低', '去雾算法影响'],
            'root_causes': [
                'Gamma曲线斜率设置过低（gamma值过大）',
                '全局对比度参数未适当提升',
                '暗部提升（Black Level）过度压缩',
                '去雾/增强算法对对比度的负面影响',
            ],
            'severity_impact': 1,
        },
        '对比度过高': {
            'primary_module': 'gamma',
            'isp_blocks': ['Gamma', 'Contrast', 'ToneMapping', 'HDR'],
            'causes': ['Gamma曲线过陡', '对比度过强', '高光压缩不足'],
            'root_causes': [
                'Gamma曲线斜率设置过高（gamma值过小）',
                'Contrast参数设置过高',
                '暗部压缩过度导致阴影断层',
                '高光区域未适当保护',
            ],
            'severity_impact': 2,
        },
        '色彩饱和度低': {
            'primary_module': 'ccm',
            'isp_blocks': ['Saturation', 'CCM', 'Hue', 'ColorSpace'],
            'causes': ['饱和度参数过低', '色彩空间转换损失', '色域压缩过度'],
            'root_causes': [
                'Saturation参数设置过低',
                'CCM色彩校正矩阵中饱和度系数不足',
                '从宽色域到sRGB映射时色域压缩过度',
                '白平衡调整后未重新饱和化',
            ],
            'severity_impact': 1,
        },
        '色彩饱和度过高': {
            'primary_module': 'ccm',
            'isp_blocks': ['Saturation', 'CCM', 'GamutMapping'],
            'causes': ['饱和度过高', '色彩空间转换溢出', '艺术化处理过重'],
            'root_causes': [
                'Saturation参数设置过高',
                'CCM矩阵系数导致色彩通道溢出',
                '色域映射（Gamut Mapping）参数不当导致色彩剪切',
            ],
            'severity_impact': 2,
        },
        '色域溢出': {
            'primary_module': 'ccm',
            'isp_blocks': ['GamutMapping', 'CCM', 'ColorSpace', 'Saturation'],
            'causes': ['色彩空间转换错误', '饱和度过高', '色彩矩阵配置错误'],
            'root_causes': [
                '色彩空间转换（WideGamut→sRGB）时色域映射参数不当',
                'CCM矩阵超出色域边界',
                '饱和度设置过高导致RGB通道超出[0,1]范围',
                'Adobe RGB vs sRGB混淆导致色彩超出目标色域',
            ],
            'severity_impact': 2,
        },
        'Gamma异常': {
            'primary_module': 'gamma',
            'isp_blocks': ['Gamma', 'ToneMapping', 'Contrast'],
            'causes': ['Gamma曲线偏离标准', '色彩管理配置错误'],
            'root_causes': [
                'Gamma值未设置为标准2.2（或sRGB定义的1/2.2≈0.45）',
                '色彩管理流程中Gamma配置不一致',
                'Tone Mapping曲线偏离摄影曲线',
            ],
            'severity_impact': 2,
        },
        '振铃效应': {
            'primary_module': 'sharpening',
            'isp_blocks': ['Sharpening', 'SharpeningKernel', 'RingingSuppression'],
            'causes': ['锐化过度', '锐化核选择不当', '频域处理不当'],
            'root_causes': [
                '锐化模块强度设置过高',
                '锐化算法核选择不当（如高半径Unsharp Mask）',
                'Ringing Suppression参数未启用或设置过低',
                '高频增强过度导致边缘震荡',
            ],
            'severity_impact': 2,
        },
        '过锐化': {
            'primary_module': 'sharpening',
            'isp_blocks': ['Sharpening', 'EdgeSharpness', 'TextureSharpness'],
            'causes': ['锐化参数过高', '多尺度锐化叠加'],
            'root_causes': [
                'Sharpening强度参数设置过高',
                '多尺度锐化（Texture+Edge）重复叠加',
                '锐化先于降噪导致噪声被锐化放大',
            ],
            'severity_impact': 2,
        },
        '边缘伪影': {
            'primary_module': 'sharpening',
            'isp_blocks': ['Sharpening', 'LDC', 'Demosaic', 'LensCorrection'],
            'causes': ['锐化在边缘处异常', '镜头畸变未校正', 'Demosaic边缘误差'],
            'root_causes': [
                '镜头桶形/枕形畸变未充分校正',
                'Demosaic算法在斜边缘处色彩分离',
                '锐化算法在高频边缘处产生异常响应',
                '镜头色差（Chromatic Aberration）未校正',
            ],
            'severity_impact': 2,
        },
        '运动模糊': {
            'primary_module': 'ae',
            'isp_blocks': ['ShutterSpeed', 'EIS', 'OIS', 'TemporalNR'],
            'causes': ['快门速度过慢', 'EIS/OIS未启用', '被摄物体运动'],
            'root_causes': [
                'AE算法曝光时间过长（暗光环境未提升ISO）',
                '电子防抖（EIS）未启用或强度不足',
                '光学防抖（OIS）参数配置不当',
                'Temporal NR对运动物体造成拖影',
            ],
            'severity_impact': 3,
        },
        '镜头畸变': {
            'primary_module': 'lens',
            'isp_blocks': ['LDC', 'GeometricCalibration', 'PerspectiveCorrection'],
            'causes': ['镜头固有畸变', '未做几何校正'],
            'root_causes': [
                'LDC（Lens Distortion Correction）参数未配置',
                '镜头畸变模型（Brown模型）参数不匹配',
                '广角镜头桶形畸变未充分校正',
                '透视校正参数不当',
            ],
            'severity_impact': 1,
        },
        '紫边': {
            'primary_module': 'lens',
            'isp_blocks': ['CACorrection', 'PurpleFringeRemoval', 'Demosaic'],
            'causes': ['镜头色差', '传感器光谱响应', '大光比场景'],
            'root_causes': [
                'CA（Chromatic Aberration）校正参数未启用',
                '紫边抑制（Purple Fringe Removal）参数不足',
                '大光比场景下色差更明显',
                'Demosaic算法在色差区域产生紫边',
            ],
            'severity_impact': 1,
        },
        '暗角': {
            'primary_module': 'lens',
            'isp_blocks': ['VignetteCorrection', 'LensShadingCorrection', 'LSC'],
            'causes': ['镜头暗角', 'LSC参数未配置', '光圈收缩'],
            'root_causes': [
                'Vignette Correction（LSC）参数未配置或强度不足',
                '光圈过小导致边缘光衰减更明显',
                '广角镜头边缘亮度衰减未充分补偿',
            ],
            'severity_impact': 1,
        },
        '闪烁': {
            'primary_module': 'ae',
            'isp_blocks': ['AEFlickerAvoid', 'AntiFlicker', 'ExpoureControl'],
            'causes': ['光源工频干扰', 'AE与光源频率不同步'],
            'root_causes': [
                '工频光源（50/60Hz）导致亮度周期性变化',
                'AE算法曝光时间与光源频率未同步',
                'Anti-Flicker（抗闪烁）功能未启用',
                '室内人工光源场景未正确识别',
            ],
            'severity_impact': 2,
        },
        '果冻效应': {
            'primary_module': 'sensor',
            'isp_blocks': ['RollingShutterCorrection', 'GlobalShutter'],
            'causes': ['卷帘快门传感器', '高速运动场景', '无校正算法'],
            'root_causes': [
                '使用卷帘快门（Rolling Shutter）传感器',
                '高速运动场景下逐行曝光时间差过大',
                'Rolling Shutter Correction（RSC）未启用',
                '果冻效应校正算法参数不当',
            ],
            'severity_impact': 2,
        },
        'ghost鬼影': {
            'primary_module': 'hdr',
            'isp_blocks': ['MHDR', 'MotionCompensation', 'GhostRemoval'],
            'causes': ['多帧HDR合成运动伪影', '运动检测与补偿不足'],
            'root_causes': [
                '多帧HDR合成时运动物体未正确检测和补偿',
                'Ghost Removal参数不足或未启用',
                '帧间运动较大导致鬼影',
                'Ghost阈值设置不当',
            ],
            'severity_impact': 2,
        },
    }

    def map_symptom_to_module(self, symptoms: List[str]) -> List[Dict]:
        """
        症状 → ISP模块映射

        输入症状列表，返回每个症状对应的根因分析结果，
        包括可能涉及的ISP模块、根因和推荐操作。

        Args:
            symptoms: 症状描述列表，如 ["噪声过多", "色彩偏蓝", "边缘模糊"]

        Returns:
            List[Dict]: 每个症状的结构化诊断报告
        """
        results = []
        for symptom in symptoms:
            # 精确匹配
            if symptom in self.SYMPTOM_MODULE_MAP:
                entry = self.SYMPTOM_MODULE_MAP[symptom]
                results.append({
                    'symptom': symptom,
                    'primary_module': entry['primary_module'],
                    'isp_blocks': entry['isp_blocks'],
                    'causes': entry['causes'],
                    'root_causes': entry['root_causes'],
                    'severity_impact': entry['severity_impact'],
                    'confidence': 1.0,
                    'match_type': 'exact',
                })
            else:
                # 模糊匹配：检查关键词
                matched = self._fuzzy_match_symptom(symptom)
                if matched:
                    results.append(matched)
                else:
                    # 完全无法匹配
                    results.append({
                        'symptom': symptom,
                        'primary_module': 'unknown',
                        'isp_blocks': [],
                        'causes': [],
                        'root_causes': ['无法确定根因，需人工分析'],
                        'severity_impact': 0,
                        'confidence': 0.0,
                        'match_type': 'none',
                        'suggestion': '请提供更多症状描述或检查ISP pipeline各模块配置',
                    })
        return results

    def _fuzzy_match_symptom(self, symptom: str) -> Optional[Dict]:
        """模糊匹配症状到已知模板"""
        symptom_lower = symptom.lower()

        # 关键词匹配表
        keyword_map = {
            'noise': '噪声过多',
            '噪点': '噪声过多',
            'grain': '噪声过多',
            'blue': '色彩偏蓝',
            '偏蓝': '色彩偏蓝',
            '蓝移': '色彩偏蓝',
            'yellow': '色彩偏黄',
            '偏黄': '色彩偏黄',
            'red': '色彩偏红',
            '偏红': '色彩偏红',
            'blur': '边缘模糊',
            '模糊': '边缘模糊',
            'detail': '细节丢失',
            '细节': '细节丢失',
            'moire': '摩尔纹',
            '摩尔': '摩尔纹',
            'false_color': '伪彩色',
            '伪彩': '伪彩色',
            'zipper': '拉链效应',
            '拉链': '拉链效应',
            'over_exposed': '过曝',
            '过曝': '过曝',
            'overexposed': '过曝',
            'under_exposed': '欠曝',
            '欠曝': '欠曝',
            'underexposed': '欠曝',
            'dynamic': '动态范围不足',
            'dr不足': '动态范围不足',
            'dynamic_range': '动态范围不足',
            'contrast_low': '对比度低',
            '对比度低': '对比度低',
            'contrast_high': '对比度过高',
            '对比度高': '对比度过高',
            'saturation_low': '色彩饱和度低',
            'sat_low': '色彩饱和度低',
            'saturation_high': '色彩饱和度过高',
            'sat_high': '色彩饱和度过高',
            'gamma': 'Gamma异常',
            'ringing': '振铃效应',
            '振铃': '振铃效应',
            'oversharp': '过锐化',
            'artifact': '边缘伪影',
            '伪影': '边缘伪影',
            'motion_blur': '运动模糊',
            '运动模糊': '运动模糊',
            'distortion': '镜头畸变',
            '畸变': '镜头畸变',
            'purple': '紫边',
            'vignette': '暗角',
            '暗角': '暗角',
            'flicker': '闪烁',
            '闪烁': '闪烁',
            'jello': '果冻效应',
            '果冻': '果冻效应',
            'ghost': 'ghost鬼影',
            '鬼影': 'ghost鬼影',
        }

        for keyword, mapped_symptom in keyword_map.items():
            if keyword in symptom_lower or symptom_lower in keyword:
                entry = self.SYMPTOM_MODULE_MAP[mapped_symptom]
                return {
                    'symptom': symptom,
                    'primary_module': entry['primary_module'],
                    'isp_blocks': entry['isp_blocks'],
                    'causes': entry['causes'],
                    'root_causes': entry['root_causes'],
                    'severity_impact': entry['severity_impact'],
                    'confidence': 0.85,
                    'match_type': 'fuzzy',
                    'matched_template': mapped_symptom,
                    'suggestion': f'模糊匹配到"{mapped_symptom}"，请确认是否正确',
                }

        return None

    def diagnose_multiple_symptoms(
        self,
        image: np.ndarray,
        symptoms: List[str]
    ) -> Dict[str, Any]:
        """
        多症状联合诊断

        综合分析图像和多个症状，进行优先级排序，
        区分主要矛盾和次要矛盾，生成结构化诊断报告。

        Args:
            image: RGB图像，shape (H, W, 3)
            symptoms: 症状描述列表

        Returns:
            Dict: 结构化诊断报告
        """
        # Step 1: 症状 → 模块映射
        symptom_results = self.map_symptom_to_module(symptoms)

        # Step 2: 调用已有的Phase 4诊断函数进行客观验证
        img = image.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0

        # 客观诊断结果
        bayer_diag = self.diagnose_bayer_issues(image)
        denoise_diag = self.suggest_denoise_params(image)
        sharp_diag = self.diagnose_sharpening_artifacts(image)
        colorspace_diag = self.diagnose_colorspace_issues(image)

        # Step 3: 综合评估
        primary_issues = []  # 主要矛盾
        secondary_issues = []  # 次要矛盾

        for sr in symptom_results:
            sev = sr['severity_impact']

            # 客观验证：根据模块查找客观诊断分数
            obj_severity = self._get_objective_severity(
                sr['primary_module'],
                bayer_diag, denoise_diag, sharp_diag, colorspace_diag
            )

            # 综合评分 = 症状严重程度权重 * 主观严重度 + 客观严重度
            if sr['primary_module'] == 'unknown':
                combined_severity = sev
            else:
                combined_severity = (sev * 0.4 + obj_severity * 0.6)

            issue_entry = {
                'symptom': sr['symptom'],
                'primary_module': sr['primary_module'],
                'isp_blocks': sr['isp_blocks'],
                'root_causes': sr['root_causes'],
                'subjective_severity': sev,
                'objective_severity': round(obj_severity, 2),
                'combined_severity': round(combined_severity, 2),
                'confidence': sr.get('confidence', 1.0),
                'causes': sr['causes'],
            }

            if combined_severity >= 2.5:
                primary_issues.append(issue_entry)
            else:
                secondary_issues.append(issue_entry)

        # Step 4: 优先级排序
        primary_sorted = self.prioritize_issues(primary_issues)
        secondary_sorted = self.prioritize_issues(secondary_issues)

        # Step 5: 生成根因分析报告
        root_cause_analysis = self._analyze_root_causes(
            primary_sorted + secondary_sorted
        )

        # Step 6: 聚合ISP模块问题
        module_summary = self._aggregate_module_issues(
            primary_sorted + secondary_sorted
        )

        return {
            'input_symptoms': symptoms,
            'symptom_mapping': symptom_results,
            'primary_issues': primary_sorted,
            'secondary_issues': secondary_sorted,
            'root_cause_analysis': root_cause_analysis,
            'module_summary': module_summary,
            'objective_diagnostics': {
                'bayer_demosaic': bayer_diag['summary'],
                'noise_level': denoise_diag['noise_level'],
                'sharpening_artifacts': sharp_diag['summary'],
                'colorspace': colorspace_diag['summary'],
            },
            'summary': {
                'total_symptoms': len(symptoms),
                'primary_issue_count': len(primary_sorted),
                'secondary_issue_count': len(secondary_sorted),
                'most_likely_root_module': primary_sorted[0]['primary_module'] if primary_sorted else 'unknown',
                'overall_severity': max(
                    [p['combined_severity'] for p in primary_sorted]
                ) if primary_sorted else 0,
                'recommendations': self._generate_diagnosis_recommendations(
                    primary_sorted, module_summary
                ),
            },
        }

    def _get_objective_severity(
        self,
        primary_module: str,
        bayer_diag: Dict,
        denoise_diag: Dict,
        sharp_diag: Dict,
        colorspace_diag: Dict
    ) -> float:
        """根据ISP模块获取客观诊断严重度"""
        module_to_objective = {
            'denoise': denoise_diag['noise_level']['severity'],
            'demosaic': bayer_diag['summary']['overall_severity'],
            'sharpening': sharp_diag['summary']['overall_severity'],
            'ccm': colorspace_diag['summary']['overall_severity'],
            'gamma': colorspace_diag.get('gamma_issues', {}).get('severity', 0),
            'ae': denoise_diag['noise_level']['severity'],  # 借用噪声评估
            'hdr': denoise_diag['noise_level']['severity'],
            'lens': colorspace_diag.get('gamut_overflow', {}).get('severity', 0),
            'sensor': denoise_diag['noise_level']['severity'],
        }
        return float(module_to_objective.get(primary_module, 0))

    def _analyze_root_causes(self, issues: List[Dict]) -> Dict[str, Any]:
        """分析根因之间的关联性"""
        # 统计根因出现频次
        root_cause_freq = {}
        for issue in issues:
            for cause in issue.get('root_causes', []):
                # 归一化根因描述
                cause_key = cause.lower().strip()
                if cause_key not in root_cause_freq:
                    root_cause_freq[cause_key] = {
                        'cause': cause,
                        'count': 0,
                        'symptoms': [],
                        'modules': [],
                    }
                root_cause_freq[cause_key]['count'] += issue['confidence']
                root_cause_freq[cause_key]['symptoms'].append(issue['symptom'])
                if issue['primary_module'] not in root_cause_freq[cause_key]['modules']:
                    root_cause_freq[cause_key]['modules'].append(issue['primary_module'])

        # 按频次排序
        sorted_causes = sorted(
            root_cause_freq.values(),
            key=lambda x: x['count'],
            reverse=True
        )

        # 区分共享根因（影响多个症状）和单一根因
        shared_causes = [c for c in sorted_causes if len(c['symptoms']) > 1]
        unique_causes = [c for c in sorted_causes if len(c['symptoms']) == 1]

        return {
            'shared_causes': shared_causes,
            'unique_causes': unique_causes,
            'most_likely_shared_cause': shared_causes[0] if shared_causes else None,
        }

    def _aggregate_module_issues(self, issues: List[Dict]) -> Dict[str, Any]:
        """聚合按ISP模块分类的问题"""
        module_issues = {}
        for issue in issues:
            module = issue['primary_module']
            if module not in module_issues:
                module_issues[module] = {
                    'module': module,
                    'symptoms': [],
                    'total_severity': 0.0,
                    'isp_blocks': set(),
                    'root_causes': [],
                }
            module_issues[module]['symptoms'].append(issue['symptom'])
            module_issues[module]['total_severity'] += issue['combined_severity']
            module_issues[module]['isp_blocks'].update(issue.get('isp_blocks', []))
            module_issues[module]['root_causes'].extend(issue.get('root_causes', []))

        # 排序
        sorted_modules = sorted(
            module_issues.values(),
            key=lambda x: x['total_severity'],
            reverse=True
        )

        # 转换set为list
        for m in sorted_modules:
            m['isp_blocks'] = list(m['isp_blocks'])
            m['total_severity'] = round(m['total_severity'], 2)

        return {
            'modules': sorted_modules,
            'priority_module': sorted_modules[0]['module'] if sorted_modules else None,
        }

    def prioritize_issues(self, issues: List[Dict]) -> List[Dict]:
        """
        问题优先级排序

        按严重程度和用户影响对问题进行排序：
        1. 综合评分（combined_severity）
        2. 用户影响度（user_impact_score）
        3. ISP模块优先级（module_priority_weight）

        Args:
            issues: 问题列表，每个问题应包含severity相关字段

        Returns:
            List[Dict]: 排序后的问题列表
        """
        # ISP模块优先级权重（模块在图像质量中的重要性）
        MODULE_PRIORITY_WEIGHT = {
            'ae': 10,          # 自动曝光：最直接影响用户感知的模块
            'ccm': 9,          # 色彩校正：直接影响色彩还原
            'sharpening': 8,   # 锐化：直接影响清晰度感知
            'gamma': 8,        # Gamma/对比度：直接影响视觉效果
            'denoise': 7,      # 降噪：夜景/暗光下重要性凸显
            'hdr': 7,          # HDR：直接影响大光比场景质量
            'demosaic': 6,     # Demosaic：基础图像重建，但问题不一定肉眼可见
            'lens': 5,         # 镜头校正：通常不易察觉
            'sensor': 4,      # 传感器处理：RAW域，通常被后续模块掩盖
        }

        # 用户影响权重
        USER_IMPACT_WEIGHT = {
            3: 3.0,   # 严重：明显影响观感，需立即处理
            2: 2.0,   # 中等：可察觉但不严重
            1: 1.0,   # 轻微：需仔细观察才能发现
            0: 0.0,   # 无影响
        }

        scored_issues = []
        for issue in issues:
            # 获取综合严重度
            combined_sev = issue.get('combined_severity',
                                     issue.get('severity_impact', 0))

            # 模块优先级权重
            module = issue.get('primary_module', 'unknown')
            module_weight = MODULE_PRIORITY_WEIGHT.get(module, 3)

            # 用户影响权重（基于症状严重度）
            subj_sev = issue.get('subjective_severity', 0)
            user_impact = USER_IMPACT_WEIGHT.get(subj_sev, 1.0)

            # 最终优先级评分 = 综合严重度 * 模块权重 * 用户影响
            # 归一化到[0, 10]
            priority_score = (
                combined_sev * 1.0 +
                module_weight * 0.3 +
                user_impact * 0.5
            )
            priority_score = round(min(priority_score, 10), 2)

            # 添加排序字段
            enriched = dict(issue)
            enriched['priority_score'] = priority_score
            enriched['module_priority_weight'] = module_weight
            enriched['user_impact'] = user_impact

            scored_issues.append(enriched)

        # 按优先级分数降序排序
        scored_issues.sort(key=lambda x: x['priority_score'], reverse=True)

        # 添加rank字段
        for i, issue in enumerate(scored_issues):
            issue['rank'] = i + 1

        return scored_issues

    def _generate_diagnosis_recommendations(
        self,
        primary_issues: List[Dict],
        module_summary: Dict[str, Any]
    ) -> List[str]:
        """生成诊断建议"""
        recommendations = []

        if not primary_issues:
            recommendations.append('未检测到明显问题，建议进行常规ISP参数维护')
            return recommendations

        # 首先处理共享根因
        shared = module_summary.get('modules', [])
        if shared:
            top_module = shared[0]
            module_name = top_module['module']
            recommendations.append(
                f'⚠️ 最优先处理模块: {module_name}，'
                f'涉及症状: {", ".join(top_module["symptoms"])}'
            )

        # 针对每个主要问题给出建议
        for issue in primary_issues[:3]:  # 最多3条
            symptom = issue['symptom']
            module = issue['primary_module']
            top_cause = issue['root_causes'][0] if issue['root_causes'] else ''

            rec = f'【{symptom}】→ 优先检查 {module} 模块'
            if top_cause:
                rec += f'，根因: {top_cause}'
            recommendations.append(rec)

        # 模块级通用建议
        modules_involved = [m['module'] for m in shared]
        if 'denoise' in modules_involved:
            recommendations.append('💡 降噪相关：建议检查Spatial NR/Temporal NR参数配置')
        if 'sharpening' in modules_involved:
            recommendations.append('💡 锐化相关：建议使用边缘自适应锐化算法')
        if 'ccm' in modules_involved:
            recommendations.append('💡 色彩相关：建议重新校准CCM矩阵或检查AWB参数')
        if 'ae' in modules_involved:
            recommendations.append('💡 曝光相关：建议调整AE目标亮度或启用HDR模式')
        if 'demosaic' in modules_involved:
            recommendations.append('💡 Demosaic相关：建议使用边缘自适应插值算法(如AHAT)')
        if 'gamma' in modules_involved:
            recommendations.append('💡 Gamma相关：建议检查Gamma曲线是否符合sRGB标准(2.2)')

        return recommendations


def create_tuning_knowledge() -> ISPTuningKnowledge:
    """创建知识库实例"""
    return ISPTuningKnowledge()
