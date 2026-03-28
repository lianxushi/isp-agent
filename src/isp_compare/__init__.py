"""
ISP Version Comparator
====================

A Python framework for comparing ISP (Image Signal Processing) versions.
Supports Comp12 RAW format, CModel ISP integration, and comprehensive image quality metrics.
"""

__version__ = "1.0.0"
__author__ = "ISP Team"

# Import from core module
from .core.comp12_parser import Comp12Parser, Comp12Config, Comp12ParseError
from .core.cmodel_wrapper import CModelISP, CModelResult, CModelError
from .core.metrics import ImageMetrics, MetricsResult
from .core.comparison import ISPComparator, ComparisonConfig, ComparisonResult

# Import tools
from .tools.perception_model import PerceptionModelInterface, DetectionResult

__all__ = [
    # Core
    "Comp12Parser",
    "Comp12Config",
    "Comp12ParseError",
    "CModelISP",
    "CModelResult",
    "CModelError",
    "ImageMetrics",
    "MetricsResult",
    "ISPComparator",
    "ComparisonConfig",
    "ComparisonResult",
    # Tools
    "PerceptionModelInterface",
    "DetectionResult",
]