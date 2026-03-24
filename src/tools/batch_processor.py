#!/usr/bin/env python3
"""
批量图像处理模块
支持批量分析、处理、导出
"""
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json

from .image_analyzer import ImageAnalyzer
from .automotive_analyzer import AutomotiveQualityAnalyzer
from .ai_quality_scorer import AIQualityScorer


@dataclass
class BatchResult:
    """批量处理结果"""
    total: int
    success: int
    failed: int
    results: List[Dict[str, Any]]


class BatchProcessor:
    """
    批量图像处理器
    
    功能:
    - 批量分析图像质量
    - 批量处理(降噪/锐化)
    - 批量导出报告
    """
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.analyzer = ImageAnalyzer()
        self.auto_analyzer = AutomotiveQualityAnalyzer()
        self.quality_scorer = AIQualityScorer()
    
    def analyze_batch(
        self,
        image_paths: List[str],
        analysis_type: str = 'full'
    ) -> BatchResult:
        """
        批量分析图像
        
        Args:
            image_paths: 图像路径列表
            analysis_type: 分析类型 ('full', 'automotive', 'quality')
        
        Returns:
            BatchResult: 批量结果
        """
        results = []
        success = 0
        failed = 0
        
        def process_single(path: str) -> Dict[str, Any]:
            try:
                if analysis_type == 'full':
                    result = self.analyzer.analyze(path)
                    return {
                        'path': path,
                        'success': True,
                        'data': result.to_dict()
                    }
                elif analysis_type == 'automotive':
                    result = self.auto_analyzer.quick_check(path)
                    return {
                        'path': path,
                        'success': True,
                        'data': result
                    }
                elif analysis_type == 'quality':
                    result = self.quality_scorer.score(path)
                    return {
                        'path': path,
                        'success': True,
                        'data': result
                    }
            except Exception as e:
                return {
                    'path': path,
                    'success': False,
                    'error': str(e)
                }
        
        # 多线程处理
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(process_single, p): p for p in image_paths}
            
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                if result['success']:
                    success += 1
                else:
                    failed += 1
        
        return BatchResult(
            total=len(image_paths),
            success=success,
            failed=failed,
            results=results
        )
    
    def scan_directory(
        self,
        directory: str,
        extensions: List[str] = None
    ) -> List[str]:
        """扫描目录获取图像文件"""
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.dng']
        
        directory = Path(directory)
        image_paths = []
        
        for ext in extensions:
            image_paths.extend(directory.glob(f'*{ext}'))
            image_paths.extend(directory.glob(f'*{ext.upper()}'))
        
        return [str(p) for p in image_paths]
    
    def generate_report(
        self,
        batch_result: BatchResult,
        output_path: str,
        format: str = 'json'
    ) -> str:
        """生成批量处理报告"""
        report = {
            'summary': {
                'total': batch_result.total,
                'success': batch_result.success,
                'failed': batch_result.failed,
                'success_rate': f"{batch_result.success/batch_result.total*100:.1f}%" if batch_result.total > 0 else "0%"
            },
            'results': batch_result.results
        }
        
        if format == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
        
        elif format == 'csv':
            import csv
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['文件名', '状态', '综合评分', '夜景', 'HDR', '模糊'])
                
                for r in batch_result.results:
                    if r['success']:
                        data = r.get('data', {})
                        writer.writerow([
                            Path(r['path']).name,
                            '成功',
                            data.get('overall_score', data.get('overall', 'N/A')),
                            data.get('night_vision_score', 'N/A'),
                            data.get('hdr_score', 'N/A'),
                            data.get('motion_blur_score', 'N/A')
                        ])
                    else:
                        writer.writerow([Path(r['path']).name, '失败', r.get('error', '未知错误')])
        
        return output_path


def batch_analyze(
    directory: str,
    analysis_type: str = 'automotive',
    output: str = None
) -> BatchResult:
    """便捷函数"""
    processor = BatchProcessor()
    image_paths = processor.scan_directory(directory)
    
    if not image_paths:
        return BatchResult(0, 0, 0, [])
    
    result = processor.analyze_batch(image_paths, analysis_type)
    
    if output:
        processor.generate_report(result, output, 'json')
    
    return result
