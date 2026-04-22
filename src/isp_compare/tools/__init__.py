"""
Tools Package
============

ISP Compare tool modules including perception model interface,
traffic light evaluator, and contour evaluator.

Author: ISP Team
"""

from .perception_model import (
    PerceptionModelInterface,
    DetectionResult,
    PerceptionEvaluation
)

from .traffic_light_evaluator import (
    TrafficLightEvaluator,
    TrafficLightRegion,
    TrafficLightEvaluationResult,
    LightColor
)

from .contour_evaluator import (
    ContourEvaluator,
    SharpnessResult,
    CompletenessResult,
    AccuracyResult,
    ContourEvaluationResult
)

__all__ = [
    # Perception
    "PerceptionModelInterface",
    "DetectionResult",
    "PerceptionEvaluation",
    # Traffic Light
    "TrafficLightEvaluator",
    "TrafficLightRegion",
    "TrafficLightEvaluationResult",
    "LightColor",
    # Contour
    "ContourEvaluator",
    "SharpnessResult",
    "CompletenessResult",
    "AccuracyResult",
    "ContourEvaluationResult",
]
