#!/usr/bin/env python3
"""
AI图像质量评分
基于深度学习的客观质量评估 (BRISQUE/NIQE风格)
"""
import cv2
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path
from ..utils.logger import setup_logger

logger = setup_logger('isp-agent.ai_quality')


class AIQualityScorer:
    """
    AI图像质量评分器
    
    实现简化版无参考图像质量评估 (Zero-Shot Quality Assessment)
    - 清晰度评分
    - 噪声评分
    - 伪影评分
    - 色彩评分
    - 综合MOS预测
    """
    
    def __init__(self):
        pass
    
    def score(self, image_path: str) -> Dict[str, Any]:
        """
        图像质量AI评分
        
        Args:
            image_path: 图像路径
        
        Returns:
            Dict: 评分结果
                - mos_predicted: 预测MOS分数 (1-5)
                - sharpness_score: 清晰度 (0-100)
                - noise_score: 噪声 (0-100)
                - artifact_score: 伪影 (0-100)
                - color_score: 色彩 (0-100)
                - overall: 综合评分 (0-100)
                - grade: 等级 (A/B/C/D)
        """
        logger.info(f"AI质量评分: {image_path}")
        
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        # 各维度评分
        sharpness = self._assess_sharpness(img)
        noise = self._assess_noise(img)
        artifact = self._assess_artifacts(img)
        color = self._assess_color(img)
        
        # 权重
        weights = {
            'sharpness': 0.35,
            'noise': 0.25,
            'artifact': 0.20,
            'color': 0.20,
        }
        
        overall = (
            sharpness * weights['sharpness'] +
            noise * weights['noise'] +
            artifact * weights['artifact'] +
            color * weights['color']
        )
        
        # 转换为MOS格式 (1-5)
        mos = 1 + 4 * (overall / 100)
        
        # 等级
        if overall >= 90:
            grade = 'A'
        elif overall >= 75:
            grade = 'B'
        elif overall >= 60:
            grade = 'C'
        else:
            grade = 'D'
        
        return {
            'mos_predicted': round(mos, 2),
            'mos_description': self._mos_to_description(mos),
            'sharpness_score': round(sharpness, 1),
            'noise_score': round(noise, 1),
            'artifact_score': round(artifact, 1),
            'color_score': round(color, 1),
            'overall': round(overall, 1),
            'grade': grade,
            'details': {
                'sharpness': self._get_dimension_comment('sharpness', sharpness),
                'noise': self._get_dimension_comment('noise', noise),
                'artifact': self._get_dimension_comment('artifact', artifact),
                'color': self._get_dimension_comment('color', color),
            }
        }
    
    def _assess_sharpness(self, img) -> float:
        """评估清晰度"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        
        # Laplacian方差 (高频内容)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        lap_var = laplacian.var()
        
        # 边缘强度 (Sobel)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_strength = np.sqrt(sobelx**2 + sobely**2).mean()
        
        # 归一化评分
        score = 0
        if lap_var > 0:
            score += min(100, lap_var / 5)  # Laplacian方差
        if edge_strength > 0:
            score += min(100, edge_strength / 3)  # 边缘强度
        
        return min(100, score / 1.5)
    
    def _assess_noise(self, img) -> float:
        """评估噪声"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        
        # 局部方差法
        kernel_size = 7
        mean = cv2.blur(gray.astype(np.float32), (kernel_size, kernel_size))
        variance = cv2.blur((gray.astype(np.float32) - mean) ** 2, (kernel_size, kernel_size))
        
        # 中值噪声估计
        noise_estimate = float(np.sqrt(np.median(variance)))
        
        # 噪声越大分数越低
        if noise_estimate < 2:
            return 100.0
        elif noise_estimate < 10:
            return 100 - (noise_estimate - 2) * 5
        else:
            return max(0, 50 - (noise_estimate - 10) * 2)
    
    def _assess_artifacts(self, img) -> float:
        """评估伪影 (块效应、振铃等)"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        
        # 块效应检测 (DCT简化版)
        h, w = gray.shape
        block_size = 8
        
        # 计算块边界差异
        block_diff = 0
        count = 0
        
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block = gray[i:i+block_size, j:j+block_size]
                right_block = gray[i:i+block_size, j+block_size:j+2*block_size]
                bottom_block = gray[i+block_size:i+2*block_size, j:j+block_size]
                
                if j + block_size < w:
                    block_diff += np.mean(np.abs(block - right_block))
                    count += 1
                if i + block_size < h:
                    block_diff += np.mean(np.abs(block - bottom_block))
                    count += 1
        
        if count > 0:
            block_diff /= count
        
        # 伪影越少分数越高
        if block_diff < 2:
            return 100.0
        elif block_diff < 10:
            return 100 - (block_diff - 2) * 5
        else:
            return max(0, 60 - (block_diff - 10) * 2)
    
    def _assess_color(self, img) -> float:
        """评估色彩质量"""
        # 检查色彩平衡
        b, g, r = cv2.split(img)
        
        means = [float(b.mean()), float(g.mean()), float(r.mean())]
        stds = [float(b.std()), float(g.std()), float(r.std())]
        
        # 色彩一致性
        mean_consistency = 1 - np.std(means) / (np.mean(means) + 1e-6)
        
        # 色彩丰富度
        colorfulness = self._colorfulness(img)
        
        # 评分
        score = mean_consistency * 50 + min(50, colorfulness * 2)
        
        return min(100, max(0, score))
    
    def _colorfulness(self, img) -> float:
        """计算色彩丰富度"""
        if len(img.shape) == 3:
            b, g, r = cv2.split(img)
            
            rg = r - g
            yb = 0.5 * (r + g) - b
            
            std_rg = np.std(rg)
            std_yb = np.std(yb)
            mean_rg = np.mean(rg)
            mean_yb = np.mean(yb)
            
            std = np.sqrt(std_rg**2 + std_yb**2)
            mean = np.sqrt(mean_rg**2 + mean_yb**2)
            
            return std + 0.3 * mean
        return 0
    
    def _mos_to_description(self, mos: float) -> str:
        """MOS分数描述"""
        if mos >= 4.5:
            return "优秀 - 视觉体验极佳"
        elif mos >= 4.0:
            return "良好 - 轻微瑕疵但不影响"
        elif mos >= 3.5:
            return "一般 - 有明显瑕疵"
        elif mos >= 3.0:
            return "较差 - 影响观看"
        else:
            return "差 - 严重质量问题"
    
    def _get_dimension_comment(self, dimension: str, score: float) -> str:
        """各维度评语"""
        comments = {
            'sharpness': {
                'high': '清晰度优秀，边缘锐利',
                'medium': '清晰度一般，细节尚可',
                'low': '清晰度不足，画面模糊',
            },
            'noise': {
                'high': '噪声控制优秀',
                'medium': '噪声可接受',
                'low': '噪声明显，影响画质',
            },
            'artifact': {
                'high': '无明显伪影',
                'medium': '有轻微伪影',
                'low': '伪影明显，压缩过度',
            },
            'color': {
                'high': '色彩还原准确',
                'medium': '色彩表现尚可',
                'low': '色彩失真或单调',
            }
        }
        
        if score >= 80:
            return comments[dimension]['high']
        elif score >= 60:
            return comments[dimension]['medium']
        else:
            return comments[dimension]['low']
    
    def batch_score(self, image_paths: list) -> list:
        """批量评分"""
        results = []
        for path in image_paths:
            try:
                result = self.score(path)
                results.append({
                    'path': path,
                    'score': result
                })
            except Exception as e:
                logger.warning(f"评分失败 {path}: {e}")
                results.append({
                    'path': path,
                    'error': str(e)
                })
        return results


def score_image_quality(image_path: str) -> Dict[str, Any]:
    """便捷函数"""
    scorer = AIQualityScorer()
    return scorer.score(image_path)
