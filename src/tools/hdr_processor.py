#!/usr/bin/env python3
"""
HDR/多帧合成处理器
支持多帧HDR合成、多帧降噪
"""
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import numpy as np
import cv2
from ..utils.logger import setup_logger

logger = setup_logger('isp-agent.hdr')


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
        # 读取所有图像
        images = []
        for path in image_paths:
            img = cv2.imread(path)
            if img is None:
                raise ValueError(f"无法读取图像: {path}")
            images.append(img)
        
        # 转换为浮点
        imgs = [np.float32(img) for img in images]
        
        # Mertens融合
        merge_mertens = cv2.createMergeMertens()
        merged = merge_mertens.process(imgs)
        
        # 转换为8位并保存
        merged = cv2.normalize(merged, None, 0, 255, cv2.NORM_MINMAX)
        merged = np.uint8(merged)
        
        cv2.imwrite(output_path, merged)
        logger.info(f"HDR合成完成: {output_path}")
    
    def _merge_by_mtb(self, image_paths: List[str], output_path: str) -> None:
        """基于MTB(中值阈值分箱)合成"""
        # 读取基准图像
        base = cv2.imread(image_paths[0])
        if base is None:
            raise ValueError(f"无法读取基准图像: {image_paths[0]}")
        
        gray_base = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
        
        # 计算MTB
        _, mtb = cv2.threshold(gray_base, 127, 255, cv2.THRESH_BINARY)
        
        # 对齐和合成（简化版）
        result = base
        for path in image_paths[1:]:
            img = cv2.imread(path)
            if img is not None:
                # 简单混合
                result = cv2.addWeighted(result, 0.7, img, 0.3, 0)
        
        cv2.imwrite(output_path, result)
    
    def _merge_and_render(self, image_paths: List[str], output_path: str) -> None:
        """HDR渲染合成"""
        # 读取并转换为32位浮点
        images = []
        exposures = []
        
        for path in image_paths:
            img = cv2.imread(path)
            if img is not None:
                img = np.float32(img) / 255.0
                images.append(img)
                exposures.append(1.0)  # 简化：假设相同曝光
        
        if len(images) < 2:
            raise ValueError("需要至少2张图像")
        
        # 曝光融合
        merge_mertens = cv2.createMergeMertens()
        hdr = merge_mertens.process(images)
        
        # Tone mapping (Reinhard)
        tonemapper = cv2.createTonemapReinhard(1.5, 0.8, 0, 0)
        ldr = tonemapper.process(hdr)
        
        # 转换为8位
        ldr = ldr * 255
        ldr = np.clip(ldr, 0, 255).astype(np.uint8)
        
        cv2.imwrite(output_path, ldr)
        logger.info(f"HDR渲染完成: {output_path}")
    
    def multi_frame_denoise(
        self,
        image_paths: List[str],
        output_path: str
    ) -> Dict[str, Any]:
        """
        多帧降噪
        
        Args:
            image_paths: 同一场景的多帧图像路径列表
            output_path: 输出路径
        
        Returns:
            Dict: 处理结果
        """
        logger.info(f"开始多帧降噪: {len(image_paths)} 张图像")
        
        if len(image_paths) < 2:
            return {'success': False, 'error': '需要至少2张图像进行多帧降噪'}
        
        try:
            # 读取所有图像
            images = []
            for path in image_paths:
                img = cv2.imread(path)
                if img is None:
                    raise ValueError(f"无法读取图像: {path}")
                images.append(img)
            
            # 中值降噪（对静态场景效果好）
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
        """
        图像对齐（用于手持HDR拍摄）
        
        Args:
            reference_path: 参考图像路径
            image_paths: 需要对齐的图像路径列表
            output_dir: 输出目录
        
        Returns:
            List[str]: 对齐后的图像路径列表
        """
        logger.info("开始图像对齐")
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 读取参考图像
        ref = cv2.imread(reference_path)
        if ref is None:
            raise ValueError(f"无法读取参考图像: {reference_path}")
        
        ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
        
        # SIFT特征点匹配
        sift = cv2.SIFT_create()
        kp_ref, des_ref = sift.detectAndCompute(ref_gray, None)
        
        output_paths = []
        
        for i, path in enumerate(image_paths):
            img = cv2.imread(path)
            if img is None:
                continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kp, des = sift.detectAndCompute(gray, None)
            
            # 特征匹配
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des, des_ref, k=2)
            
            # 应用比率测试
            good = []
            for m_n in matches:
                if len(m_n) == 2:
                    if m_n[0].distance < 0.75 * m_n[1].distance:
                        good.append(m_n[0])
            
            if len(good) > 10:
                # 计算单应性矩阵
                src_pts = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_ref[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                
                # 透视变换
                aligned = cv2.warpPerspective(img, M, (ref.shape[1], ref.shape[0]))
                
                output_path = os.path.join(output_dir, f"aligned_{i}.jpg")
                cv2.imwrite(output_path, aligned)
                output_paths.append(output_path)
            else:
                # 特征点不足，复制原图
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
