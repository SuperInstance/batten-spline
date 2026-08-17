"""BattenSpline — distance-weighted interpolation for cascade routing."""

from .acclimation_fit import (
    AcclimationCurve,
    fit_acclimation,
    predict_next,
    skill_from_curve,
)
from .batten import Batten
from .router import CascadeRouter, RouteResult
from .spline import BattenSpline

__all__ = [
    "AcclimationCurve",
    "Batten",
    "BattenSpline",
    "CascadeRouter",
    "RouteResult",
    "fit_acclimation",
    "predict_next",
    "skill_from_curve",
]
__version__ = "0.1.0"
