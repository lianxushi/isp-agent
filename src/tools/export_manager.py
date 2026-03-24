#!/usr/bin/env python3
"""
图像导出和报告生成模块
支持多种格式的导出和报告生成
"""
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import asdict

from .image_analyzer import AnalysisResult
from .automotive_analyzer import AutomotiveQualityResult
from .ai_quality_scorer import AIQualityScorer


class ExportManager:
    """
    导出管理器
    
    支持格式:
    - JSON: 结构化数据
    - HTML: 可视化报告
    - Markdown: 文档格式
    - CSV: 表格数据
    """
    
    def __init__(self, template_dir: Optional[str] = None):
        self.template_dir = template_dir
    
    def export_json(
        self,
        data: Dict,
        output_path: str,
        indent: int = 2
    ) -> str:
        """导出JSON"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
        return output_path
    
    def export_markdown(
        self,
        analysis_result: AnalysisResult,
        output_path: str
    ) -> str:
        """导出Markdown报告"""
        md = []
        md.append(f"# 图像质量分析报告")
        md.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        md.append(f"\n## 基本信息")
        md.append(f"| 项目 | 值 |")
        md.append(f"|------|-----|")
        md.append(f"| 文件名 | {analysis_result.file_name} |")
        md.append(f"| 分辨率 | {analysis_result.width} x {analysis_result.height} |")
        md.append(f"| 格式 | {analysis_result.format} |")
        md.append(f"| 大小 | {analysis_result.size_kb:.1f} KB |")
        
        if analysis_result.dynamic_range:
            dr = analysis_result.dynamic_range
            md.append(f"\n## 动态范围")
            md.append(f"- 范围: {dr['min']} - {dr['max']}")
            md.append(f"- 有效范围: {dr['useful_range']}")
        
        if analysis_result.noise_level:
            md.append(f"\n## 噪声分析")
            md.append(f"- 噪声水平: {analysis_result.noise_level:.2f}")
        
        if analysis_result.color_analysis:
            ca = analysis_result.color_analysis
            md.append(f"\n## 色彩分析")
            md.append(f"- 白平衡: {ca.get('white_balance', 'N/A')}")
            md.append(f"- 饱和度: {ca.get('saturation', 'N/A')}")
        
        if analysis_result.exif:
            md.append(f"\n## EXIF信息")
            for k, v in analysis_result.exif.items():
                md.append(f"- {k}: {v}")
        
        content = '\n'.join(md)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return output_path
    
    def export_html(
        self,
        data: Dict,
        output_path: str,
        title: str = "ISP分析报告"
    ) -> str:
        """导出HTML报告"""
        html = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
               max-width: 800px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
        .card {{ background: white; border-radius: 12px; padding: 20px; margin: 20px 0; 
                 box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
        .metric {{ display: flex; justify-content: space-between; padding: 10px 0; 
                   border-bottom: 1px solid #eee; }}
        .label {{ color: #666; }}
        .value {{ font-weight: bold; color: #333; }}
        .score {{ font-size: 24px; color: #2196F3; }}
        .good {{ color: #4CAF50; }}
        .warn {{ color: #FF9800; }}
        .bad {{ color: #F44336; }}
    </style>
</head>
<body>
    <h1>📷 {title}</h1>
    <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
'''
        
        # 基本信息
        if 'file_name' in data:
            html += '''
    <div class="card">
        <h2>基本信息</h2>
'''
            for key in ['file_name', 'width', 'height', 'format', 'size_kb']:
                if key in data:
                    val = data[key]
                    if key == 'size_kb':
                        val = f"{val:.1f} KB"
                    html += f'        <div class="metric"><span class="label">{key}</span><span class="value">{val}</span></div>\n'
            html += '    </div>\n'
        
        # 质量评分
        if 'overall_score' in data or 'overall' in data:
            score = data.get('overall_score', data.get('overall', 0))
            html += f'''
    <div class="card">
        <h2>质量评分</h2>
        <div class="score">{score:.1f}</div>
    </div>
'''
        
        html += '''
</body>
</html>'''
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        return output_path
    
    def export_csv(
        self,
        results: List[Dict],
        output_path: str
    ) -> str:
        """导出CSV"""
        if not results:
            return output_path
        
        import csv
        
        # 获取所有字段
        fields = set()
        for r in results:
            fields.update(r.keys())
        
        fields = sorted(fields)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(results)
        
        return output_path
    
    def create_report(
        self,
        analysis_result,
        output_dir: str,
        formats: List[str] = None
    ) -> Dict[str, str]:
        """
        创建多格式报告
        
        Args:
            analysis_result: 分析结果
            output_dir: 输出目录
            formats: 导出格式列表
        
        Returns:
            Dict: 格式到路径的映射
        """
        if formats is None:
            formats = ['json', 'html']
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 转换为字典
        if hasattr(analysis_result, 'to_dict'):
            data = analysis_result.to_dict()
        else:
            data = analysis_result
        
        # 文件名
        base_name = Path(data.get('file_name', 'report')).stem
        
        outputs = {}
        
        for fmt in formats:
            output_path = os.path.join(output_dir, f"{base_name}.{fmt}")
            
            try:
                if fmt == 'json':
                    outputs[fmt] = self.export_json(data, output_path)
                elif fmt == 'html':
                    outputs[fmt] = self.export_html(data, output_path)
                elif fmt == 'markdown':
                    if hasattr(analysis_result, 'to_dict'):
                        outputs[fmt] = self.export_markdown(analysis_result, output_path)
            except Exception as e:
                print(f"导出{fmt}失败: {e}")
        
        return outputs


def create_export_manager() -> ExportManager:
    """创建导出管理器"""
    return ExportManager()
