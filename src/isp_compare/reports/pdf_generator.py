"""
PDF Report Generator
==================

Generates PDF reports for ISP version comparison results.
Requires: pip install reportlab

Author: ISP Team
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from ..utils.logger import get_logger

logger = get_logger(__name__)

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    logger.warning("reportlab not available, PDF generation will be limited. Install with: pip install reportlab")


class PDFReportGenerator:
    """PDF report generator for ISP comparison results."""

    def __init__(self):
        if not REPORTLAB_AVAILABLE:
            logger.warning("PDF generation requires reportlab: pip install reportlab")
        self.styles = getSampleStyleSheet() if REPORTLAB_AVAILABLE else None
        if self.styles:
            self._setup_custom_styles()

    def _setup_custom_styles(self):
        self.styles.add(ParagraphStyle(name='CustomTitle', parent=self.styles['Heading1'], fontSize=24, textColor=colors.HexColor('#1a365d'), alignment=TA_CENTER, spaceAfter=30))
        self.styles.add(ParagraphStyle(name='CustomHeading', parent=self.styles['Heading2'], fontSize=16, textColor=colors.HexColor('#2d3748'), spaceBefore=20, spaceAfter=10))
        self.styles.add(ParagraphStyle(name='CustomNormal', parent=self.styles['Normal'], fontSize=11, textColor=colors.HexColor('#4a5568'), spaceAfter=6))

    def generate(self, result, output_path: str, images: Optional[Dict[str, str]] = None) -> str:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not REPORTLAB_AVAILABLE:
            return self._generate_markdown(result, str(output_path).replace(".pdf", ".md"))

        try:
            doc = SimpleDocTemplate(str(output_path), pagesize=A4, rightMargin=2*cm, leftMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
            story = []
            story.append(Paragraph("ISP Version Comparison Report", self.styles['CustomTitle']))
            story.append(Spacer(1, 20))

            info_data = [["Report ID", result.report_id], ["Timestamp", result.timestamp], ["Version A", result.version_a], ["Version B", result.version_b], ["Overall Status", self._format_status(result.overall_status)], ["Processing Time", f"{result.processing_time_ms:.1f} ms"]]
            story.append(self._create_table(info_data, [4*cm, 10*cm]))
            story.append(Spacer(1, 20))

            if images:
                story.append(Paragraph("Visual Comparison", self.styles['CustomHeading']))
                story.append(self._create_image_table(images))
                story.append(Spacer(1, 20))

            if hasattr(result, 'comparison') and result.comparison:
                story.append(Paragraph("Metrics Comparison", self.styles['CustomHeading']))
                story.append(self._create_comparison_table(result))
                story.append(Spacer(1, 20))

            story.append(Paragraph("Summary", self.styles['CustomHeading']))
            story.append(Paragraph(result.summary or "No summary available", self.styles['CustomNormal']))
            story.append(Spacer(1, 20))

            if result.recommendations:
                story.append(Paragraph("Recommendations", self.styles['CustomHeading']))
                for rec in result.recommendations:
                    story.append(Paragraph(f"• {rec}", self.styles['CustomNormal']))
                story.append(Spacer(1, 20))

            doc.build(story)
            logger.info(f"PDF report generated: {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"PDF generation failed: {e}")
            return self._generate_markdown(result, str(output_path).replace(".pdf", ".md"))

    def _create_table(self, data, col_widths):
        table = Table(data, colWidths=col_widths)
        table.setStyle(TableStyle([('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#edf2f7')), ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'), ('FONTSIZE', (0, 0), (-1, -1), 10), ('ALIGN', (0, 0), (-1, -1), 'LEFT'), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e0')), ('PADDING', (0, 0), (-1, -1), 8)]))
        return table

    def _create_image_table(self, images):
        data = []
        labels = []
        img_cells = []
        for label, path in [("Golden", images.get("golden", "")), ("Version A", images.get("version_a", "")), ("Version B", images.get("version_b", ""))]:
            labels.append(label)
            if path and os.path.exists(path):
                try:
                    img_cells.append(Image(path, width=5*cm, height=3*cm))
                except:
                    img_cells.append(Paragraph("[Image]", self.styles['CustomNormal']))
            else:
                img_cells.append(Paragraph("[N/A]", self.styles['CustomNormal']))
        data.append(labels)
        data.append(img_cells)
        table = Table(data, colWidths=[5*cm, 5*cm, 5*cm])
        table.setStyle(TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('FONTSIZE', (0, 0), (-1, 0), 10), ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold')]))
        return table

    def _create_comparison_table(self, result):
        comp = result.comparison
        data = [["Metric", "Version A", "Version B", "Delta", "Better"]]
        for name, key in [("Overall Score", "a_score"), ("Sharpness", "sharpness"), ("Noise", "noise"), ("Color", "color"), ("Traffic Light", "traffic_light")]:
            if key in comp:
                a = comp.get(f'a_{name.split()[0].lower()}', comp.get('a_score', 0))
                b = comp.get(f'b_{name.split()[0].lower()}', comp.get('b_score', 0))
                d = comp.get(f'{name.split()[0].lower()}_delta', 0)
                w = "B" if d > 0 else "A"
                data.append([name, f"{a:.1f}", f"{b:.1f}", f"{d:+.1f}", w])
        table = Table(data, colWidths=[3*cm, 3*cm, 3*cm, 2*cm, 2*cm])
        table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2d3748')), ('TEXTCOLOR', (0, 0), (-1, 0), colors.white), ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'), ('FONTSIZE', (0, 0), (-1, -1), 10), ('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e0')), ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f7fafc')])]))
        return table

    def _format_status(self, status: str) -> str:
        return {"a_improved": "A is better", "b_improved": "B is better", "similar": "Similar", "needs_attention": "Needs Attention", "error": "Error"}.get(status, status)

    def _generate_markdown(self, result, output_path: str) -> str:
        md = f"""# ISP Version Comparison Report

## Report Information
| Field | Value |
|-------|-------|
| Report ID | {result.report_id} |
| Timestamp | {result.timestamp} |
| Version A | {result.version_a} |
| Version B | {result.version_b} |
| Status | {result.overall_status} |
| Time | {result.processing_time_ms:.1f} ms |

## Summary
{result.summary or 'No summary'}

## Recommendations
"""
        for r in (result.recommendations or []):
            md += f"- {r}\n"
        md += f"\n---\n*Generated by ISP-Agent {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"
        with open(output_path, 'w') as f:
            f.write(md)
        return output_path