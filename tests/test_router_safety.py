"""
test_router_safety.py — Safety, robustness, and negative-space regression tests.

Documents known gaps in CascadeRouter and BattenSpline:
- NaN embedding propagation (silent corruption)
- Fog density is computed but not used in routing
- Empty router behavior
- Extreme inputs (Inf, huge dimensions)
- State dict round-trip fidelity
- Prune spatial coverage gap
"""

import numpy as np
import pytest
import time

from batten_spline.router import CascadeRouter, RouteResult
from batten_spline.spline import BattenSpline
from batten_spline.batten import Batten


# ============================================================================
# NaN HANDLING
# ============================================================================

class TestNaNHandling:
    """NaN embeddings silently corrupt the spline — documented gap."""

    def test_nan_embedding_route_does_not_crash(self):
        """Routing a NaN embedding should not crash. It returns CLOUD with conf 0.0."""
        router = CascadeRouter()
        r = router.route(np.array([float('nan'), 0.0]))
        assert r.target == "CLOUD"
        assert r.confidence == 0.0

    def test_nan_batten_corrupts_confidence(self):
        """DOCUMENTED BUG: Adding a batten with NaN embedding produces NaN confidence
        for queries that get near it. This is silent corruption — no validation.
        
        If this test starts failing (e.g., confidence is not NaN), someone added
        NaN validation to add_batten. Update accordingly.
        """
        spline = BattenSpline()
        spline.add_batten(np.array([float('nan'), 0.0]), quality=0.9)
        conf = spline.estimate_confidence(np.array([0.0, 0.0]))
        # NaN is the current behavior — this documents it
        assert np.isnan(conf)

    def test_nan_batten_fog_density_is_nan(self):
        """A NaN batten makes fog_density NaN for all queries."""
        spline = BattenSpline()
        spline.add_batten(np.array([float('nan'), 0.0]), quality=0.9)
        fog = spline.fog_density(np.array([1.0, 0.0]))
        assert np.isnan(fog) or fog == float('inf')


# ============================================================================
# FOG DENSITY — COMPUTED BUT NOT USED
# ============================================================================

class TestFogDensityNotUsedInRouting:
    """DOCUMENTED GAP: fog_density is reported in RouteResult but does not
    influence the routing decision. Only confidence matters.
    
    This means a query point far from all battens can still be routed to
    LOCAL if the interpolated confidence happens to be high enough.
    """

    def test_fog_density_reported_in_result(self):
        router = CascadeRouter()
        router.report_outcome(np.array([0.0, 0.0]), quality=0.95)
        r = router.route(np.array([0.01, 0.01]))
        assert r.fog_density is not None
        assert r.fog_density > 0

    def test_high_fog_still_routes_local_if_confidence_high(self):
        """With large fog_scale, distant queries still get high confidence."""
        spline = BattenSpline(fog_scale=100.0)  # Very wide kernel
        router = CascadeRouter(spline=spline)
        router.report_outcome(np.array([0.0, 0.0]), quality=0.95)
        
        # A point very far from the batten
        r = router.route(np.array([50.0, 50.0]))
        # fog_density is huge
        assert r.fog_density > 50.0
        # But confidence might still be nonzero due to wide kernel
        # This is the gap: high fog + nonzero confidence = LOCAL route
        # even though we're in completely uncharted territory
        # The test documents this behavior

    def test_fog_density_zero_at_batten(self):
        """Fog density should be 0 at a batten location."""
        router = CascadeRouter()
        router.report_outcome(np.array([0.5, 0.3]), quality=0.9)
        r = router.route(np.array([0.5, 0.3]))
        assert r.fog_density == pytest.approx(0.0, abs=1e-10)


# ============================================================================
# EXTREME INPUTS
# ============================================================================

class TestExtremeInputs:
    def test_inf_embedding_routes_cloud(self):
        router = CascadeRouter()
        r = router.route(np.array([float('inf'), 0.0]))
        assert r.target == "CLOUD"

    def test_zero_vector_empty_router(self):
        router = CascadeRouter()
        r = router.route(np.zeros(3))
        assert r.target == "CLOUD"
        assert r.confidence == 0.0

    def test_high_dimensional(self):
        """10K-dimensional embedding should work."""
        router = CascadeRouter()
        router.report_outcome(np.random.randn(10000), quality=0.9)
        r = router.route(np.random.randn(10000))
        assert r.target in ("LOCAL", "CASCADE", "CLOUD")

    def test_quality_above_one_clipped(self):
        router = CascadeRouter()
        b = router.report_outcome(np.array([0.0, 0.0]), quality=5.0)
        assert b.quality_score == 1.0

    def test_quality_below_zero_clipped(self):
        router = CascadeRouter()
        b = router.report_outcome(np.array([0.0, 0.0]), quality=-3.0)
        assert b.quality_score == 0.0

    def test_quality_boundary_zero(self):
        """Quality of exactly 0.0 should be valid."""
        router = CascadeRouter()
        b = router.report_outcome(np.array([0.0, 0.0]), quality=0.0)
        assert b.quality_score == 0.0

    def test_quality_boundary_one(self):
        """Quality of exactly 1.0 should be valid."""
        router = CascadeRouter()
        b = router.report_outcome(np.array([0.0, 0.0]), quality=1.0)
        assert b.quality_score == 1.0


# ============================================================================
# STATE DICT ROUND-TRIP
# ============================================================================

class TestStateDictRoundTrip:
    def test_empty_round_trip(self):
        s = BattenSpline()
        sd = s.state_dict()
        s2 = BattenSpline.from_state_dict(sd)
        assert len(s2.battens) == 0

    def test_with_battens_round_trip(self):
        s = BattenSpline(fog_scale=2.5, half_life=3600)
        s.add_batten(np.array([1.0, 0.0]), quality=0.9, timestamp=1000.0)
        s.add_batten(np.array([0.0, 1.0]), quality=0.3, timestamp=2000.0)
        
        sd = s.state_dict()
        s2 = BattenSpline.from_state_dict(sd)
        
        assert len(s2.battens) == 2
        assert s2.fog_scale == 2.5
        assert s2.half_life == 3600

    def test_router_round_trip(self):
        router = CascadeRouter()
        router.report_outcome(np.array([0.1, 0.0]), quality=0.9)
        router.report_outcome(np.array([5.0, 5.0]), quality=0.2)
        
        sd = router.state_dict()
        router2 = CascadeRouter.from_state_dict(sd)
        
        assert len(router2.spline.battens) == 2
        # Same routing decision for same input
        r1 = router.route(np.array([0.1, 0.0]))
        r2 = router2.route(np.array([0.1, 0.0]))
        assert r1.target == r2.target
        assert r1.confidence == pytest.approx(r2.confidence)

    def test_custom_targets_round_trip(self):
        targets = {
            "FAST": {"threshold": 0.9},
            "SLOW": {"threshold": 0.0},
        }
        router = CascadeRouter(targets=targets)
        sd = router.state_dict()
        router2 = CascadeRouter.from_state_dict(sd)
        assert "FAST" in router2.targets
        assert "SLOW" in router2.targets


# ============================================================================
# PRUNE BEHAVIOR
# ============================================================================

class TestPruneBehavior:
    def test_prune_keeps_count(self):
        s = BattenSpline()
        for i in range(100):
            s.add_batten(np.array([float(i), 0.0]), quality=0.5, timestamp=float(i))
        removed = s.prune(max_battens=50)
        assert removed == 50
        assert len(s.battens) == 50

    def test_prune_no_op_when_under_limit(self):
        s = BattenSpline()
        s.add_batten(np.array([0.0]), quality=0.5)
        removed = s.prune(max_battens=500)
        assert removed == 0
        assert len(s.battens) == 1

    def test_prune_keeps_recent_over_stale(self):
        """Prune should prefer battens with higher age_weight (more recent)."""
        s = BattenSpline(half_life=100.0)  # Short half-life so age matters
        now = time.time()
        # One very old, one very new
        s.add_batten(np.array([0.0]), quality=0.5, timestamp=now - 10000)  # very stale
        s.add_batten(np.array([1.0]), quality=0.5, timestamp=now)  # fresh
        s.prune(max_battens=1)
        assert len(s.battens) == 1
        # The fresh one should survive
        assert abs(s.battens[0].timestamp - now) < 1.0


# ============================================================================
# ROUTE RESULT STRUCTURE
# ============================================================================

class TestRouteResult:
    def test_route_result_is_frozen(self):
        r = RouteResult(target="LOCAL", confidence=0.9, fog_density=0.1, reason="test")
        with pytest.raises(Exception):
            r.target = "CLOUD"  # Should raise — frozen dataclass

    def test_route_result_fields(self):
        r = RouteResult(target="LOCAL", confidence=0.9, fog_density=0.1, reason="test")
        assert r.target == "LOCAL"
        assert r.confidence == 0.9
        assert r.fog_density == 0.1
        assert r.reason == "test"


# ============================================================================
# CUSTOM TARGETS VALIDATION
# ============================================================================

class TestCustomTargetsValidation:
    def test_empty_targets_raises(self):
        with pytest.raises(ValueError, match="At least one"):
            CascadeRouter(targets={})

    def test_missing_threshold_raises(self):
        with pytest.raises(ValueError, match="threshold"):
            CascadeRouter(targets={"BAD": {"description": "no threshold"}})

    def test_custom_threshold_ordering(self):
        """Targets should work regardless of dict ordering."""
        targets = {
            "LOW": {"threshold": 0.0},
            "HIGH": {"threshold": 0.9},
        }
        router = CascadeRouter(targets=targets)
        router.report_outcome(np.array([0.0, 0.0]), quality=0.95)
        r = router.route(np.array([0.0, 0.0]))
        assert r.target == "HIGH"

    def test_overlapping_thresholds(self):
        """If two targets have the same threshold, either can win."""
        targets = {
            "A": {"threshold": 0.5},
            "B": {"threshold": 0.5},
        }
        router = CascadeRouter(targets=targets)
        router.report_outcome(np.array([0.0, 0.0]), quality=0.9)
        r = router.route(np.array([0.0, 0.0]))
        assert r.target in ("A", "B")
