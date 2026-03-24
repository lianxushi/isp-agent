#!/usr/bin/env python3
"""
RAW格式处理模块
支持DNG/CR2/NEF/ARW等RAW格式
"""
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
from ..utils.logger import setup_logger

logger = setup_logger('isp-agent.raw')


class RawProcessor:
    """
    RAW格式处理器
    
    支持格式:
    - DNG (Adobe Digital Negative)
    - CR2 (Canon)
    - NEF (Nikon)
    - ARW (Sony)
    - RAF (Fujifilm)
    - RW2 (Panasonic)
    """
    
    SUPPORTED_FORMATS = ['.dng', '.cr2', '.nef', '.arw', '.raf', '.rw2', '.orf']
    
    def __init__(self):
        self._check_rawpy()
    
    def _check_rawpy(self):
        """检查rawpy是否可用"""
        try:
            import rawpy
            self.rawpy = rawpy
        except ImportError:
            logger.warning("rawpy未安装，RAW处理功能受限")
            self.rawpy = None
    
    def get_info(self, raw_path: str) -> Dict[str, Any]:
        """
        获取RAW文件信息
        
        Args:
            raw_path: RAW文件路径
        
        Returns:
            Dict: RAW文件信息
        """
        logger.info(f"读取RAW信息: {raw_path}")
        
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {raw_path}")
        
        info = {
            'file_name': path.name,
            'file_size': path.stat().st_size,
            'format': path.suffix.upper(),
        }
        
        if self.rawpy:
            try:
                with self.rawpy.imread(raw_path) as raw:
                    info.update({
                        'width': raw.raw_dimensions[0],
                        'height': raw.raw_dimensions[1],
                        'pattern': str(raw.color_pattern),  # Bayer pattern
                        'white_level': raw.white_level,
                        'black_level': raw.black_level_per_channel,
                        'iso': raw.iso_speed,
                        'exposure_time': raw.exposure_time,
                        'firmware': raw.firmware,
                        'timestamp': raw.timestamp if hasattr(raw, 'timestamp') else None,
                    })
                    
                    # 色彩信息
                    if hasattr(raw, 'color_desc'):
                        info['color_desc'] = raw.color_desc
                    
                    # 镜头信息
                    if hasattr(raw, 'lens'):
                        info['lens'] = str(raw.lens)
                    
                    logger.info(f"RAW信息读取成功: {info['width']}x{info['height']}")
                    
            except Exception as e:
                logger.error(f"读取RAW信息失败: {e}")
                info['error'] = str(e)
        else:
            info['warning'] = "rawpy未安装，无法读取详细信息"
        
        return info
    
    def process(
        self,
        raw_path: str,
        output_path: str,
        settings: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        处理RAW文件
        
        Args:
            raw_path: RAW文件路径
            output_path: 输出路径
            settings: 处理设置
        
        Returns:
            Dict: 处理结果
        """
        logger.info(f"处理RAW文件: {raw_path}")
        
        settings = settings or {}
        
        if not self.rawpy:
            return {'success': False, 'error': 'rawpy未安装'}
        
        try:
            with self.rawpy.imread(raw_path) as raw:
                # 处理参数
                demosaic = settings.get('demosaic', True)
                use_camera_wb = settings.get('use_camera_wb', True)
                use_camera_nr = settings.get('use_camera_nr', False)
                no_auto_bright = settings.get('no_auto_bright', False)
                output_bps = settings.get('output_bps', 8)
                
                # Post processing设置
                pp = self.rawpy.PostProcessor()
                
                if demosaic:
                    pass  # 默认开启
                
                # 应用设置
                if 'brightness' in settings:
                    pp.brightness(settings['brightness'])
                
                if 'gamma' in settings:
                    pp.gamma(settings['gamma'])
                
                if 'no_auto_bright' in settings:
                    pp.no_auto_bright()
                
                # 执行处理
                rgb = raw.postprocess(
                    use_camera_wb=use_camera_wb,
                    use_camera_nr=use_camera_nr,
                    no_auto_bright=no_auto_bright,
                    output_bps=output_bps,
                )
                
                # 保存
                from PIL import Image
                img = Image.fromarray(rgb)
                img.save(output_path)
                
                logger.info(f"RAW处理完成: {output_path}")
                
                return {
                    'success': True,
                    'output': output_path,
                    'width': rgb.shape[1],
                    'height': rgb.shape[0],
                }
                
        except Exception as e:
            logger.error(f"RAW处理失败: {e}")
            return {'success': False, 'error': str(e)}
    
    def extract_raw_data(self, raw_path: str) -> Optional[Dict[str, Any]]:
        """
        提取RAW原始数据用于分析
        
        Args:
            raw_path: RAW文件路径
        
        Returns:
            Dict: 原始数据信息
        """
        if not self.rawpy:
            return None
        
        try:
            with self.rawpy.imread(raw_path) as raw:
                # 获取原始数据
                raw_data = raw.raw_image_visible
                
                # 统计信息
                stats = {
                    'min': int(raw_data.min()),
                    'max': int(raw_data.max()),
                    'mean': float(raw_data.mean()),
                    'std': float(raw_data.std()),
                }
                
                # Bayer pattern分析
                pattern = str(raw.color_pattern)
                
                return {
                    'dimensions': raw.raw_dimensions,
                    'pattern': pattern,
                    'stats': stats,
                    'black_level': raw.black_level_per_channel,
                    'white_level': raw.white_level,
                }
                
        except Exception as e:
            logger.error(f"提取RAW数据失败: {e}")
            return None
    
    def to_tiff(self, raw_path: str, output_path: str) -> Dict[str, Any]:
        """转换为TIFF格式"""
        return self.process(raw_path, output_path, {'output_bps': 16})
    
    def to_jpeg(self, raw_path: str, output_path: str, quality: int = 95) -> Dict[str, Any]:
        """转换为JPEG格式"""
        return self.process(
            raw_path, 
            output_path, 
            {'output_bps': 8}
        )


def process_raw(raw_path: str, output_path: str, **kwargs) -> Dict[str, Any]:
    """便捷函数"""
    processor = RawProcessor()
    return processor.process(raw_path, output_path, kwargs)


def get_raw_info(raw_path: str) -> Dict[str, Any]:
    """便捷函数"""
    processor = RawProcessor()
    return processor.get_info(raw_path)
