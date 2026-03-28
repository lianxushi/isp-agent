"""
ISP Comparator
=============

Main comparison engine for ISP version comparison.
Coordinates Comp12 parsing, CModel processing, metrics calculation, and reporting.

Author: ISP Team
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from .comp12_parser import Comp12Parser, Comp12Config
from .cmodel_wrapper import CModelISP, CModelResult
from .metrics import ImageMetrics, MetricsResult

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ComparisonConfig:
    """Comparison configuration"""
    # Comp12 settings
    comp12_width: int = 3840
    comp12_height: int = 2160
    comp12_pattern: str = "RGGB"
    
    # CModel settings
    cmodel_path: str = ""
    cmodel_threads: int = 8
    cmodel_params: Dict[str, Any] = field(default_factory=dict)
    
    # Analysis settings
    traffic_light_roi: Optional[Tuple[int, int, int, int]] = None
    enable_perception: bool = False
    
    # Output settings
    output_dir: str = "./output"
    save_intermediate: bool = False


@dataclass
class ComparisonResult:
    """Result of ISP version comparison"""
    report_id: str
    timestamp: str
    
    # Version info
    version_a: str = ""
    version_b: str = ""
    
    # Processing results
    version_a_result: Optional[CModelResult] = None
    version_b_result: Optional[CModelResult] = None
    
    # Metrics
    version_a_metrics: Optional[MetricsResult] = None
    version_b_metrics: Optional[MetricsResult] = None
    
    # Comparison
    comparison: Dict[str, Any] = field(default_factory=dict)
    
    # Summary
    overall_status: str = "unknown"
    summary: str = ""
    recommendations: List[str] = field(default_factory=list)
    
    # Performance
    processing_time_ms: float = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        d = asdict(self)
        # Handle nested objects
        if self.version_a_result:
            d["version_a_result"] = asdict(self.version_a_result)
        if self.version_b_result:
            d["version_b_result"] = asdict(self.version_b_result)
        return d


class ISPComparator:
    """
    Main ISP version comparator.
    
    Compares two ISP versions by:
    1. Processing RAW images through CModel
    2. Calculating image quality metrics
    3. Comparing results and generating report
    
    Example:
        >>> config = ComparisonConfig(
        ...     cmodel_path="/path/to/cmodel",
        ...     cmodel_threads=8
        ... )
        >>> comparator = ISPComparator(config)
        >>> result = comparator.compare_versions(
        ...     version_a_raw="isp_v1.raw",
        ...     version_b_raw="isp_v2.raw",
        ...     golden_path="golden.jpg"
        ... )
        >>> print(result.summary)
    """
    
    def __init__(self, config: ComparisonConfig):
        """
        Initialize ISP comparator.
        
        Args:
            config: Comparison configuration
        """
        self.config = config
        self.comp12_parser = Comp12Parser(Comp12Config(
            width=config.comp12_width,
            height=config.comp12_height,
            pattern=config.comp12_pattern
        ))
        
        if config.cmodel_path:
            self.cmodel = CModelISP(
                config.cmodel_path,
                num_threads=config.cmodel_threads,
                default_params=config.cmodel_params
            )
        else:
            self.cmodel = None
            logger.warning("CModel path not provided, processing disabled")
        
        self.metrics = ImageMetrics()
        
        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ISPComparator initialized: {self.output_dir}")
    
    def compare_versions(
        self,
        version_a_raw: str,
        version_b_raw: str,
        golden_path: Optional[str] = None,
        version_a_label: str = "Version A",
        version_b_label: str = "Version B"
    ) -> ComparisonResult:
        """
        Compare two ISP versions.
        
        Args:
            version_a_raw: Path to Version A RAW file
            version_b_raw: Path to Version B RAW file
            golden_path: Optional Golden reference image
            version_a_label: Label for Version A
            version_b_label: Label for Version B
            
        Returns:
            ComparisonResult: Comparison result
        """
        start_time = time.time()
        report_id = f"ISP_COMP_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Starting comparison: {version_a_label} vs {version_b_label}")
        
        result = ComparisonResult(
            report_id=report_id,
            timestamp=datetime.now().isoformat(),
            version_a=version_a_label,
            version_b=version_b_label
        )
        
        try:
            # Process Version A
            logger.info(f"Processing {version_a_label}...")
            version_a_result = self._process_raw(version_a_raw, f"{report_id}_A")
            result.version_a_result = version_a_result
            
            if not version_a_result.success:
                result.overall_status = "error"
                result.summary = f"{version_a_label} processing failed: {version_a_result.error}"
                return result
            
            # Process Version B
            logger.info(f"Processing {version_b_label}...")
            version_b_result = self._process_raw(version_b_raw, f"{report_id}_B")
            result.version_b_result = version_b_result
            
            if not version_b_result.success:
                result.overall_status = "error"
                result.summary = f"{version_b_label} processing failed: {version_b_result.error}"
                return result
            
            # Calculate metrics
            logger.info("Calculating metrics...")
            result.version_a_metrics = self.metrics.evaluate(
                version_a_result.output_path,
                reference_path=golden_path,
                traffic_light_roi=self.config.traffic_light_roi
            )
            
            result.version_b_metrics = self.metrics.evaluate(
                version_b_result.output_path,
                reference_path=golden_path,
                traffic_light_roi=self.config.traffic_light_roi
            )
            
            # Compare
            result.comparison = self._compare_metrics(
                result.version_a_metrics,
                result.version_b_metrics,
                golden_path
            )
            
            # Generate summary
            result.summary, result.recommendations = self._generate_summary(result)
            
            # Determine overall status
            if result.version_a_metrics.passed and result.version_b_metrics.passed:
                if result.comparison.get("b_better_count", 0) > result.comparison.get("a_better_count", 0):
                    result.overall_status = "b_improved"
                elif result.comparison.get("a_better_count", 0) > result.comparison.get("b_better_count", 0):
                    result.overall_status = "a_improved"
                else:
                    result.overall_status = "similar"
            else:
                result.overall_status = "needs_attention"
            
            logger.info(f"Comparison complete: {result.overall_status}")
            
        except Exception as e:
            logger.error(f"Comparison failed: {e}")
            result.overall_status = "error"
            result.summary = f"Comparison failed: {e}"
        
        result.processing_time_ms = (time.time() - start_time) * 1000
        
        return result
    
    def _process_raw(self, raw_path: str, output_prefix: str) -> CModelResult:
        """Process RAW file through CModel"""
        if not self.cmodel:
            return CModelResult(
                success=False,
                error="CModel not configured"
            )
        
        # Save Comp12 as CModel input
        raw16 = self.comp12_parser.parse(raw_path)
        temp_raw = self.output_dir / f"{output_prefix}_input.raw"
        self.comp12_parser.save_for_cmodel(raw16, str(temp_raw))
        
        # Process through CModel
        output_path = self.output_dir / f"{output_prefix}_output.jpg"
        result = self.cmodel.process(str(temp_raw), str(output_path))
        
        # Cleanup temp file
        if not self.config.save_intermediate:
            temp_raw.unlink(missing_ok=True)
        
        return result
    
    def _compare_metrics(
        self,
        metrics_a: MetricsResult,
        metrics_b: MetricsResult,
        golden_path: Optional[str]
    ) -> Dict[str, Any]:
        """Compare metrics between two versions"""
        comparison = {}
        
        # Overall score comparison
        comparison["a_score"] = metrics_a.overall_score
        comparison["b_score"] = metrics_b.overall_score
        comparison["score_delta"] = metrics_b.overall_score - metrics_a.overall_score
        
        # Count improvements
        a_better_count = 0
        b_better_count = 0
        
        # Sharpness
        comparison["sharpness"] = {
            "a": metrics_a.sharpness_score,
            "b": metrics_b.sharpness_score,
            "delta": metrics_b.sharpness_score - metrics_a.sharpness_score,
            "winner": "B" if metrics_b.sharpness_score > metrics_a.sharpness_score else "A"
        }
        if metrics_b.sharpness_score > metrics_a.sharpness_score:
            b_better_count += 1
        else:
            a_better_count += 1
        
        # Noise
        comparison["noise"] = {
            "a": metrics_a.noise_score,
            "b": metrics_b.noise_score,
            "delta": metrics_b.noise_score - metrics_a.noise_score,
            "winner": "B" if metrics_b.noise_score > metrics_a.noise_score else "A"
        }
        if metrics_b.noise_score > metrics_a.noise_score:
            b_better_count += 1
        else:
            a_better_count += 1
        
        # Color
        comparison["color"] = {
            "a": metrics_a.color_score,
            "b": metrics_b.color_score,
            "delta": metrics_b.color_score - metrics_a.color_score,
            "winner": "B" if metrics_b.color_score > metrics_a.color_score else "A"
        }
        if metrics_b.color_score > metrics_a.color_score:
            b_better_count += 1
        else:
            a_better_count += 1
        
        # Traffic light
        comparison["traffic_light"] = {
            "a": metrics_a.traffic_light_score,
            "b": metrics_b.traffic_light_score,
            "delta": metrics_b.traffic_light_score - metrics_a.traffic_light_score,
            "winner": "B" if metrics_b.traffic_light_score > metrics_a.traffic_light_score else "A",
            "red_delta_a": metrics_a.red_light_delta,
            "red_delta_b": metrics_b.red_light_delta,
            "green_delta_a": metrics_a.green_light_delta,
            "green_delta_b": metrics_b.green_light_delta
        }
        if metrics_b.traffic_light_score > metrics_a.traffic_light_score:
            b_better_count += 1
        else:
            a_better_count += 1
        
        comparison["a_better_count"] = a_better_count
        comparison["b_better_count"] = b_better_count
        
        return comparison
    
    def _generate_summary(
        self,
        result: ComparisonResult
    ) -> Tuple[str, List[str]]:
        """Generate summary and recommendations"""
        summary_parts = []
        recommendations = []
        
        # Overall comparison
        if result.comparison:
            score_a = result.comparison.get("a_score", 0)
            score_b = result.comparison.get("b_score", 0)
            delta = result.comparison.get("score_delta", 0)
            
            if abs(delta) < 2:
                summary_parts.append(f"整体质量评分接近 ({score_a:.1f} vs {score_b:.1f})")
            elif delta > 0:
                summary_parts.append(f"{result.version_b} 整体评分更高 (+{delta:.1f})")
            else:
                summary_parts.append(f"{result.version_a} 整体评分更高 (+{-delta:.1f})")
        
        # Sharpness
        sharp = result.comparison.get("sharpness", {})
        if sharp.get("winner") == "B" and sharp.get("delta", 0) > 5:
            recommendations.append(f"锐度：{result.version_b} 更优 (+{sharp.get('delta', 0):.1f})")
        elif sharp.get("winner") == "A" and sharp.get("delta", 0) > 5:
            recommendations.append(f"锐度：{result.version_a} 更优 (+{-sharp.get('delta', 0):.1f})")
        
        # Noise
        noise = result.comparison.get("noise", {})
        if noise.get("winner") == "B" and noise.get("delta", 0) > 5:
            recommendations.append(f"噪声控制：{result.version_b} 更优 (+{noise.get('delta', 0):.1f})")
        
        # Traffic light
        tl = result.comparison.get("traffic_light", {})
        if tl.get("winner") == "B" and tl.get("delta", 0) > 5:
            recommendations.append(f"交通灯还原：{result.version_b} 更优 (+{tl.get('delta', 0):.1f})")
        
        # Color deviation
        red_a = tl.get("red_delta_a", 0)
        red_b = tl.get("red_delta_b", 0)
        if red_b < red_a - 0.01:
            recommendations.append("红灯色差改善")
        elif red_b > red_a + 0.01:
            recommendations.append("⚠️ 红灯色差变差，建议检查 CCM 参数")
        
        summary = "；".join(summary_parts) if summary_parts else "对比完成"
        
        return summary, recommendations
    
    def save_result(self, result: ComparisonResult, output_path: Optional[str] = None) -> str:
        """Save comparison result to JSON"""
        if output_path is None:
            output_path = self.output_dir / f"{result.report_id}.json"
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Result saved: {output_path}")
        return str(output_path)
