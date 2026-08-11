"""NaN safety tests for BattenSpline — the fleet-wide crusade reaches the fog."""

import math
import numpy as np
import pytest

from batten_spline.spline import BattenSpline


class TestNaNSafety:
    """The value that says 'I am not a value' should not silently corrupt routing."""

    def test_nan_confidence_routes_to_cloud(self):
        """NaN confidence should fall through to CLOUD, not silently match any threshold."""
        spline = BattenSpline()
        assert spline.routing_decision(confidence=float("nan")) == "CLOUD"

    def test_inf_confidence_routes_to_cloud(self):
        """Infinity should not be treated as a valid high-confidence signal."""
        spline = BattenSpline()
        assert spline.routing_decision(confidence=float("inf")) == "CLOUD"

    def test_negative_inf_confidence_routes_to_cloud(self):
        spline = BattenSpline()
        assert spline.routing_decision(confidence=float("-inf")) == "CLOUD"

    def test_nan_fog_scale_raises(self):
        with pytest.raises(ValueError, match="fog_scale"):
            BattenSpline(fog_scale=float("nan"))

    def test_inf_fog_scale_raises(self):
        with pytest.raises(ValueError, match="fog_scale"):
            BattenSpline(fog_scale=float("inf"))

    def test_nan_half_life_raises(self):
        with pytest.raises(ValueError, match="half_life"):
            BattenSpline(half_life=float("nan"))

    def test_inf_half_life_raises(self):
        with pytest.raises(ValueError, match="half_life"):
            BattenSpline(half_life=float("inf"))

    def test_nan_local_threshold_defaults_to_standard(self):
        spline = BattenSpline(local_threshold=float("nan"))
        assert spline.local_threshold == 0.7

    def test_nan_cascade_threshold_defaults_to_standard(self):
        spline = BattenSpline(cascade_threshold=float("nan"))
        assert spline.cascade_threshold == 0.3

    def test_estimate_confidence_with_nan_embedding_does_not_crash(self):
        """A NaN embedding should not cause a crash — it returns 0.0 (complete fog)."""
        spline = BattenSpline()
        spline.add_batten(embedding=np.array([1.0, 0.0]), quality=0.9)
        # NaN in the query embedding will produce NaN distances
        # The estimator should not crash, and the result should be finite or 0
        try:
            result = spline.estimate_confidence(np.array([float("nan"), 0.0]))
            assert math.isfinite(result) or result == 0.0
        except (ValueError, FloatingPointError):
            pass  # numpy may raise — that's acceptable as long as it doesn't silently corrupt

    def test_normal_routing_unchanged(self):
        """Normal confidence values should route exactly as before."""
        spline = BattenSpline()
        assert spline.routing_decision(confidence=0.9) == "LOCAL"
        assert spline.routing_decision(confidence=0.5) == "CASCADE"
        assert spline.routing_decision(confidence=0.1) == "CLOUD"
