#!/usr/bin/env python3
"""
HTML Report Generator
=====================

Generates interactive HTML reports for ISP version comparison results.
Supports visualizations: BRISQUE/HDR/ΔE multi-dimensional comparison tables,
matplotlib/plotly charts.

Author: ISP Team
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict

try:
    from ..utils.logger import setup_logger
except ImportError:
    import logging
    def setup_logger(name):
        logging.basicConfig(level=20, format='%(message)s')
        return logging.getLogger(name)

logger = setup_logger('isp-agent.html_reporter')

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not available, charts will be limited")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


@dataclass
class ComparisonData:
    """对比数据结构"""
    report_id: str = ""
    timestamp: str = ""
    version_a: str = ""
    version_b: str = ""
    overall_status: str = ""
    processing_time_ms: float = 0.0
    summary: str = ""
    recommendations: List[str] = None
    # 对比指标
    metrics: Optional[Dict[str, Any]] = None
    # HDR分析
    hdr_analysis: Optional[Dict[str, Any]] = None
    # ISO16505评估
    iso16505: Optional[Dict[str, Any]] = None
    # BRISQUE评分
    brisque_scores: Optional[Dict[str, float]] = None


def _safe(val, default=None):
    """安全获取字典值"""
    if val is None:
        return default
    return val


def _build_header(title: str, timestamp: str) -> str:
    """生成HTML头部"""
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        :root {{
            --primary: #1a365d;
            --secondary: #2d3748;
            --accent: #3182ce;
            --success: #38a169;
            --warning: #d69e2e;
            --danger: #e53e3e;
            --bg: #f7fafc;
            --card-bg: #ffffff;
            --border: #e2e8f0;
            --text: #2d3748;
            --text-muted: #718096;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               background: var(--bg); color: var(--text); line-height: 1.6; }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
        .header {{ background: linear-gradient(135deg, var(--primary), #2c5282);
                   color: white; padding: 30px; border-radius: 12px; margin-bottom: 24px;
                   box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
        .header h1 {{ font-size: 1.8em; margin-bottom: 8px; }}
        .header .meta {{ opacity: 0.85; font-size: 0.9em; }}
        .card {{ background: var(--card-bg); border-radius: 10px; padding: 20px;
                 margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.06);
                 border: 1px solid var(--border); }}
        .card h2 {{ font-size: 1.2em; color: var(--secondary); margin-bottom: 16px;
                    border-bottom: 2px solid var(--accent); padding-bottom: 8px; }}
        .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
        .grid-3 {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; }}
        .grid-4 {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; }}
        @media (max-width: 900px) {{ .grid-2, .grid-3, .grid-4 {{ grid-template-columns: 1fr; }} }}
        .stat-box {{ background: var(--bg); border-radius: 8px; padding: 16px; text-align: center; }}
        .stat-value {{ font-size: 2em; font-weight: bold; color: var(--accent); }}
        .stat-label {{ font-size: 0.85em; color: var(--text-muted); margin-top: 4px; }}
        .badge {{ display: inline-block; padding: 3px 10px; border-radius: 12px;
                  font-size: 0.8em; font-weight: 600; }}
        .badge-pass {{ background: #c6f6d5; color: #22543d; }}
        .badge-fail {{ background: #fed7d7; color: #742a2a; }}
        .badge-warn {{ background: #fefcbf; color: #744210; }}
        .badge-info {{ background: #bee3f8; color: #2a4365; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
        th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid var(--border); }}
        th {{ background: var(--secondary); color: white; font-weight: 600; font-size: 0.9em; }}
        tr:hover {{ background: #f8fafc; }}
        .metric-row {{ display: flex; justify-content: space-between; align-items: center; padding: 8px 0; }}
        .metric-name {{ font-weight: 500; }}
        .metric-values {{ display: flex; gap: 24px; }}
        .metric-val {{ text-align: center; min-width: 60px; }}
        .metric-val .label {{ font-size: 0.75em; color: var(--text-muted); }}
        .metric-val .value {{ font-size: 1.1em; font-weight: 600; }}
        .progress-bar {{ background: #e2e8f0; border-radius: 6px; height: 8px; overflow: hidden; margin-top: 4px; }}
        .progress-fill {{ height: 100%; border-radius: 6px; transition: width 0.5s; }}
        .img-comparison {{ display: flex; gap: 16px; flex-wrap: wrap; justify-content: center; }}
        .img-panel {{ flex: 1; min-width: 280px; }}
        .img-panel img {{ width: 100%; border-radius: 8px; border: 1px solid var(--border); }}
        .img-panel .caption {{ text-align: center; margin-top: 6px; font-size: 0.85em; color: var(--text-muted); }}
        .chart-container {{ margin: 16px 0; text-align: center; }}
        .chart-container img {{ max-width: 100%; border-radius: 8px; }}
        .recommendation {{ background: #ebf8ff; border-left: 4px solid var(--accent);
                           padding: 10px 14px; margin-bottom: 8px; border-radius: 0 6px 6px 0; }}
        .score-ring {{ display: inline-block; width: 80px; height: 80px; position: relative; }}
        .score-ring svg {{ transform: rotate(-90deg); }}
        .score-ring text {{ transform: rotate(90deg); transform-origin: center; }}
        .footer {{ text-align: center; color: var(--text-muted); font-size: 0.8em;
                   padding: 20px; margin-top: 30px; }}
        .tab-nav {{ display: flex; gap: 4px; margin-bottom: 16px; flex-wrap: wrap; }}
        .tab-btn {{ padding: 8px 16px; border: none; background: var(--bg); cursor: pointer;
                    border-radius: 6px 6px 0 0; font-size: 0.9em; color: var(--text-muted);
                    transition: all 0.2s; }}
        .tab-btn.active {{ background: var(--accent); color: white; }}
        .tab-content {{ display: none; }}
        .tab-content.active {{ display: block; }}
    </style>
</head>
<body>
<div class="container">
"""


def _build_footer() -> str:
    return f"""
    <div class="footer">
        <p>Generated by ISP-Agent &bull; {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
</div>
</body>
</html>
"""


def _status_badge(status: str) -> str:
    """状态徽章"""
    mapping = {
        'a_improved': ('A更好', 'badge-pass'),
        'b_improved': ('B更好', 'badge-pass'),
        'similar': ('相近', 'badge-info'),
        'needs_attention': ('需关注', 'badge-warn'),
        'error': ('错误', 'badge-fail'),
    }
    label, cls = mapping.get(status, (status, 'badge-info'))
    return f'<span class="badge {cls}">{label}</span>'


def _score_color(score: float) -> str:
    """根据分数返回颜色"""
    if score >= 80:
        return '#38a169'
    elif score >= 60:
        return '#3182ce'
    elif score >= 40:
        return '#d69e2e'
    else:
        return '#e53e3e'


def _render_metric_table(metrics: Dict[str, Any], title: str = "指标对比") -> str:
    """渲染指标对比表格"""
    if not metrics:
        return ""

    rows = []
    for key, val in metrics.items():
        if isinstance(val, dict):
            a_val = val.get('a_value', val.get('a', '-'))
            b_val = val.get('b_value', val.get('b', '-'))
            delta = val.get('delta', val.get('d', '-'))
            better = val.get('better', '-')
            name = val.get('name', key)
        else:
            a_val = val
            b_val = '-'
            delta = '-'
            better = '-'
            name = key

        rows.append(f"""
        <tr>
            <td>{name}</td>
            <td>{a_val}</td>
            <td>{b_val}</td>
            <td>{delta}</td>
            <td>{better}</td>
        </tr>
        """)

    return f"""
    <div class="card">
        <h2>{title}</h2>
        <table>
            <thead>
                <tr><th>指标</th><th>版本A</th><th>版本B</th><th>差异</th><th>更优</th></tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
    </div>
    """


def _render_briskque_chart(brisque_scores: Dict[str, float]) -> str:
    """渲染BRISQUE评分雷达/柱状图"""
    if not brisque_scores or not MATPLOTLIB_AVAILABLE:
        return ""

    try:
        labels = list(brisque_scores.keys())
        scores = list(brisque_scores.values())

        fig, ax = plt.subplots(figsize=(8, 4))
        colors = [_score_color(s) for s in scores]
        bars = ax.bar(labels, scores, color=colors, edgecolor='white', linewidth=1)
        ax.set_ylim(0, 100)
        ax.set_ylabel('Score')
        ax.set_title('BRISQUE Quality Scores')
        ax.axhline(y=80, color='#38a169', linestyle='--', alpha=0.7, label='Good (80)')
        ax.axhline(y=60, color='#d69e2e', linestyle='--', alpha=0.7, label='Acceptable (60)')

        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f'{score:.1f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        chart_path = '/tmp/isp_brisque_chart.png'
        plt.savefig(chart_path, dpi=120, bbox_inches='tight')
        plt.close()

        return f'<div class="chart-container"><img src="{chart_path}" alt="BRISQUE Scores"></div>'
    except Exception as e:
        logger.warning(f"BRISQUE图表渲染失败: {e}")
        return ""


def _render_hdr_analysis(hdr: Dict[str, Any]) -> str:
    """渲染HDR分析结果"""
    if not hdr:
        return ""

    dr = hdr.get('dynamic_range', {})
    stops = dr.get('stops', 'N/A')
    min_v = dr.get('min', 'N/A')
    max_v = dr.get('max', 'N/A')

    hs = hdr.get('exposure_analysis', {})
    over_pct = hs.get('over_exposed_percent', 0)
    under_pct = hs.get('under_exposed_percent', 0)
    assessment = hs.get('exposure_assessment', '')

    lc = hdr.get('local_contrast', {})
    contrast_5 = lc.get('5x5_window', lc.get('5x5', 'N/A'))
    contrast_15 = lc.get('15x15_window', lc.get('15x15', 'N/A'))

    return f"""
    <div class="card">
        <h2>HDR 动态范围分析</h2>
        <div class="grid-3">
            <div class="stat-box">
                <div class="stat-value">{stops}</div>
                <div class="stat-label">动态范围 (stops)</div>
                <div style="margin-top:6px;font-size:0.8em;color:var(--text-muted)">
                    范围 [{min_v:.2f}, {max_v:.2f}]
                </div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{over_pct:.1f}%</div>
                <div class="stat-label">过曝区域</div>
                <div style="margin-top:6px;font-size:0.8em;color:var(--text-muted)">
                    欠曝: {under_pct:.1f}%
                </div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{contrast_5:.1f}</div>
                <div class="stat-label">局部对比度 (5x5)</div>
                <div style="margin-top:6px;font-size:0.8em;color:var(--text-muted)">
                    15x15: {contrast_15:.1f}
                </div>
            </div>
        </div>
        <div style="margin-top:12px;">
            <span class="badge badge-info">{assessment}</span>
        </div>
    </div>
    """


def _render_iso16505_section(iso: Dict[str, Any]) -> str:
    """渲染ISO16505评估结果"""
    if not iso:
        return ""

    overall = iso.get('overall_score', 0)
    compliant = iso.get('compliant', False)
    color_acc = iso.get('color_accuracy', {})
    sharpness = iso.get('sharpness', {})
    noise = iso.get('noise', {})

    compliant_badge = 'badge-pass' if compliant else 'badge-fail'
    compliant_text = '合规' if compliant else '不合规'

    # 预计算颜色值
    overall_color = _score_color(overall)
    color_score = color_acc.get('score', 0)
    color_score_val = color_acc.get('score', 'N/A')
    color_score_color = _score_color(color_score)
    color_de = color_acc.get('mean_de', 'N/A')

    sharpness_score = sharpness.get('acuity_score', 0)
    sharpness_score_val = sharpness.get('acuity_score', 'N/A')
    sharpness_score_color = _score_color(sharpness_score)
    sharpness_mtf = sharpness.get('mtf_nyquist', 'N/A')

    noise_score = noise.get('noise_score', 0)
    noise_score_val = noise.get('noise_score', 'N/A')
    noise_score_color = _score_color(noise_score)
    noise_snr = noise.get('snr_db', 'N/A')

    return """
    <div class="card">
        <h2>ISO 16505 合规性评估</h2>
        <div class="grid-4">
            <div class="stat-box">
                <div class="stat-value" style="color:""" + overall_color + """">""" + str(overall) + """</div>
                <div class="stat-label">综合评分</div>
                <div style="margin-top:4px">
                    <span class="badge """ + compliant_badge + """">""" + compliant_text + """</span>
                </div>
            </div>
            <div class="stat-box">
                <div class="stat-value" style="color:""" + color_score_color + """">
                    """ + str(color_score_val) + """
                </div>
                <div class="stat-label">色彩准确性 (DE)</div>
                <div style="margin-top:4px;font-size:0.8em;color:var(--text-muted)">
                    DE=""" + str(color_de) + """
                </div>
            </div>
            <div class="stat-box">
                <div class="stat-value" style="color:""" + sharpness_score_color + """">
                    """ + str(sharpness_score_val) + """
                </div>
                <div class="stat-label">锐度 MTF</div>
                <div style="margin-top:4px;font-size:0.8em;color:var(--text-muted)">
                    @""" + str(sharpness_mtf) + """
                </div>
            </div>
            <div class="stat-box">
                <div class="stat-value" style="color:""" + noise_score_color + """">
                    """ + str(noise_score_val) + """
                </div>
                <div class="stat-label">噪声 SNR</div>
                <div style="margin-top:4px;font-size:0.8em;color:var(--text-muted)">
                    """ + str(noise_snr) + """ dB
                </div>
            </div>
        </div>
    </div>
    """


def _render_comparison_chart(metrics: Dict[str, Any]) -> str:
    """渲染BRISQUE/HDR/ΔE多维对比柱状图"""
    if not metrics or not MATPLOTLIB_AVAILABLE:
        return ""

    try:
        names = []
        a_vals = []
        b_vals = []

        for key, val in metrics.items():
            if isinstance(val, dict):
                names.append(str(val.get('name', key)))
                a_vals.append(float(val.get('a_value', val.get('a', 0))))
                b_vals.append(float(val.get('b_value', val.get('b', 0))))

        if not names:
            return ""

        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(names))
        width = 0.35

        bars_a = ax.bar(x - width/2, a_vals, width, label='Version A', color='#3182ce', alpha=0.85)
        bars_b = ax.bar(x + width/2, b_vals, width, label='Version B', color='#38a169', alpha=0.85)

        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_title('BRISQUE / HDR / Delta-E Multi-Dimensional Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha='right')
        ax.legend()
        ax.set_ylim(0, 110)
        ax.axhline(y=80, color='#38a169', linestyle='--', alpha=0.5, linewidth=0.8)
        ax.axhline(y=60, color='#d69e2e', linestyle='--', alpha=0.5, linewidth=0.8)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        chart_path = '/tmp/isp_comparison_chart.png'
        plt.savefig(chart_path, dpi=120, bbox_inches='tight')
        plt.close()

        return f'<div class="chart-container"><img src="{chart_path}" alt="Comparison Chart"></div>'
    except Exception as e:
        logger.warning(f"对比图表渲染失败: {e}")
        return ""


def _render_images(images: Dict[str, str]) -> str:
    """渲染图像对比"""
    if not images:
        return ""

    panels = []
    for label, path in images.items():
        if path and os.path.exists(path):
            panels.append(f"""
            <div class="img-panel">
                <img src="{path}" alt="{label}">
                <div class="caption">{label}</div>
            </div>
            """)

    if not panels:
        return ""

    return f"""
    <div class="card">
        <h2>视觉对比</h2>
        <div class="img-comparison">
            {''.join(panels)}
        </div>
    </div>
    """


def _render_recommendations(recs: List[str]) -> str:
    """渲染建议列表"""
    if not recs:
        return ""

    items = ''.join(f'<div class="recommendation">&#8226; {r}</div>' for r in recs)
    return f"""
    <div class="card">
        <h2>优化建议</h2>
        {items}
    </div>
    """


class HTMLReportGenerator:
    """
    ISP对比报告HTML生成器

    支持:
    - BRISQUE/HDR/ΔE 多维对比表格
    - matplotlib 可视化图表
    - 图像对比展示
    - ISO 16505 合规性面板
    """

    def __init__(self):
        self.template_header = _build_header
        self.template_footer = _build_footer

    def generate(
        self,
        data: ComparisonData,
        output_path: str,
        images: Optional[Dict[str, str]] = None,
        brisque_chart: bool = True
    ) -> str:
        """
        生成HTML报告

        Args:
            data: ComparisonData 数据对象
            output_path: 输出HTML文件路径
            images: 图像路径字典 {label: path}
            brisque_chart: 是否生成BRISQUE图表

        Returns:
            输出HTML文件路径
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        title = f"ISP Version Comparison Report"

        html = _build_header(title, data.timestamp)

        # 信息概览
        html += f"""
        <div class="header">
            <h1>{title}</h1>
            <div class="meta">
                <span>Report ID: {data.report_id}</span> &bull;
                <span>{data.timestamp}</span> &bull;
                <span>A: {data.version_a}</span> vs
                <span>B: {data.version_b}</span> &bull;
                {_status_badge(data.overall_status)} &bull;
                <span>{data.processing_time_ms:.1f}ms</span>
            </div>
        </div>
        """

        # 综合评分
        if hasattr(data, 'overall_score') and data.overall_score:
            score = data.overall_score
            html += f"""
            <div class="card">
                <h2>综合评分</h2>
                <div style="text-align:center; padding: 20px;">
                    <div style="font-size:4em; font-weight:bold; color:{_score_color(score)}">
                        {score:.1f}
                    </div>
                    <div style="color:var(--text-muted); margin-top:8px;">满分100</div>
                    <div class="progress-bar" style="margin-top:12px; height:12px;">
                        <div class="progress-fill" style="width:{score}%; background:{_score_color(score)}"></div>
                    </div>
                </div>
            </div>
            """

        # Tab导航
        html += '<div class="tab-nav">'
        tabs = []
        if data.metrics:
            tabs.append(('<button class="tab-btn active" onclick="showTab(\'metrics\')">指标对比</button>', 'metrics'))
        if data.hdr_analysis:
            tabs.append(('<button class="tab-btn" onclick="showTab(\'hdr\')">HDR分析</button>', 'hdr'))
        if data.iso16505:
            tabs.append(('<button class="tab-btn" onclick="showTab(\'iso\')">ISO16505</button>', 'iso'))
        if images:
            tabs.append(('<button class="tab-btn" onclick="showTab(\'visual\')">视觉对比</button>', 'visual'))

        for btn, _ in tabs:
            html += btn
        html += '</div>'

        # Tab内容
        if data.metrics:
            html += '<div id="tab-metrics" class="tab-content active">'
            html += _render_comparison_chart(data.metrics)
            html += _render_metric_table(data.metrics)
            html += '</div>'

        if data.hdr_analysis:
            html += '<div id="tab-hdr" class="tab-content">'
            html += _render_hdr_analysis(data.hdr_analysis)
            html += '</div>'

        if data.iso16505:
            html += '<div id="tab-iso" class="tab-content">'
            html += _render_iso16505_section(data.iso16505)
            html += '</div>'

        if images:
            html += '<div id="tab-visual" class="tab-content">'
            html += _render_images(images)
            html += '</div>'

        # 摘要和建议
        if data.summary:
            html += f"""
            <div class="card">
                <h2>摘要</h2>
                <p>{data.summary}</p>
            </div>
            """

        html += _render_recommendations(data.recommendations or [])
        html += _build_footer()

        # Tab切换脚本
        tab_script = """
        <script>
        function showTab(name) {
            document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
            document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
            document.getElementById('tab-' + name).classList.add('active');
            event.target.classList.add('active');
        }
        </script>
        """
        html = html.replace('</body>', tab_script + '</body>')

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)

        logger.info(f"HTML报告已生成: {output_path}")
        return str(output_path)

    def generate_from_dict(
        self,
        data_dict: Dict[str, Any],
        output_path: str,
        images: Optional[Dict[str, str]] = None
    ) -> str:
        """从字典生成报告"""
        data = ComparisonData(
            report_id=data_dict.get('report_id', ''),
            timestamp=data_dict.get('timestamp', ''),
            version_a=data_dict.get('version_a', ''),
            version_b=data_dict.get('version_b', ''),
            overall_status=data_dict.get('overall_status', ''),
            processing_time_ms=data_dict.get('processing_time_ms', 0.0),
            summary=data_dict.get('summary', ''),
            recommendations=data_dict.get('recommendations', []),
            metrics=data_dict.get('metrics'),
            hdr_analysis=data_dict.get('hdr_analysis'),
            iso16505=data_dict.get('iso16505'),
            brisque_scores=data_dict.get('brisque_scores'),
        )
        return self.generate(data, output_path, images)


def generate_html_report(
    data: ComparisonData,
    output_path: str,
    images: Optional[Dict[str, str]] = None
) -> str:
    """便捷函数：生成HTML报告"""
    gen = HTMLReportGenerator()
    return gen.generate(data, output_path, images)
