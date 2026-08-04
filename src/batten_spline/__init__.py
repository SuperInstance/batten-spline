"""BattenSpline — distance-weighted interpolation for cascade routing."""

from .batten import Batten
from .router import CascadeRouter, RouteResult
from .spline import BattenSpline

__all__ = ["Batten", "BattenSpline", "CascadeRouter", "RouteResult"]
__version__ = "0.1.0"
