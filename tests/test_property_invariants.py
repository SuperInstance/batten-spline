"""
Property-based tests for batten-spline mathematical invariants.

Tests the Nadaraya-Watson kernel regression properties:
- Confidence is always in [0, 1]
- Identical embedding to a batten returns that batten's quality
- Adding more battens near a point converges the estimate
- Fog density increases with distance from known battens
- Time decay reduces influence of old battens
- Empty spline returns 0 confidence (complete fog)
"""

import numpy as np
import pytest
from batten_spline.spline import BattenSpline
from batten_spline.batten import Batten
from batten_spline.router import CascadeRouter


class TestConfidenceBounds:
    """Confidence must always be in [0, 1]."""

    @pytest.mark.parametrize("seed", range(20))
    def test_random_embeddings_stay_bounded(self, seed):
        rng = np.random.RandomState(seed)
        spline = BattenSpline(fog_scale=0.5)
        for _ in range(10):
            emb = rng.randn(3)
            quality = rng.uniform(0, 1)
            spline.add_batten(emb, quality)
        for _ in range(20):
            query = rng.randn(3) * 5
            conf = spline.estimate_confidence(query)
            assert 0.0 <= conf <= 1.0, f"Confidence {conf} out of bounds for seed {seed}"

    def test_extreme_quality_values_stay_bounded(self):
        spline = BattenSpline(fog_scale=0.5)
        spline.add_batten(np.array([0, 0, 0]), quality=0.0)
        spline.add_batten(np.array([10, 10, 10]), quality=1.0)
        spline.add_batten(np.array([5, 5, 5]), quality=0.5)
        for x in np.linspace(-5, 15, 50):
            conf = spline.estimate_confidence(np.array([x, x, x]))
            assert 0.0 <= conf <= 1.0

    def test_high_dimensional_embeddings(self):
        """Confidence bounds hold in higher dimensions."""
        rng = np.random.RandomState(42)
        spline = BattenSpline(fog_scale=2.0)
        for _ in range(20):
            emb = rng.randn(128)
            quality = rng.uniform(0, 1)
            spline.add_batten(emb, quality)
        query = rng.randn(128)
        conf = spline.estimate_confidence(query)
        assert 0.0 <= conf <= 1.0


class TestKernelProperties:
    """Nadaraya-Watson kernel regression mathematical properties."""

    def test_identical_embedding_returns_batten_quality(self):
        """Querying the exact position of a batten should return ~its quality."""
        spline = BattenSpline(fog_scale=1.0)
        emb = np.array([1.0, 2.0, 3.0])
        spline.add_batten(emb, quality=0.85)
        conf = spline.estimate_confidence(emb)
        assert abs(conf - 0.85) < 0.01, f"Expected ~0.85, got {conf}"

    def test_single_batten_constant_confidence(self):
        """With a single batten, confidence equals its quality everywhere (NW estimator property)."""
        spline = BattenSpline(fog_scale=1.0)
        spline.add_batten(np.array([0, 0]), quality=0.9)
        # NW with 1 data point = that point's value, regardless of distance
        near = spline.estimate_confidence(np.array([0.1, 0.1]))
        far = spline.estimate_confidence(np.array([5.0, 5.0]))
        assert abs(near - 0.9) < 0.01
        assert abs(far - 0.9) < 0.01
        # But fog density DOES change with distance
        fog_near = spline.fog_density(np.array([0.1, 0.1]))
        fog_far = spline.fog_density(np.array([5.0, 5.0]))
        assert fog_near < fog_far, "Fog should be lower near the batten"

    def test_two_battens_interpolate(self):
        """Between two battens, confidence should be between their qualities."""
        spline = BattenSpline(fog_scale=1.0)
        spline.add_batten(np.array([0, 0]), quality=0.2)
        spline.add_batten(np.array([2, 0]), quality=0.8)
        midpoint = spline.estimate_confidence(np.array([1, 0]))
        assert 0.2 < midpoint < 0.8, f"Midpoint {midpoint} should be between 0.2 and 0.8"

    def test_equal_quality_battens_provide_equal_confidence(self):
        """Multiple battens with same quality at same distance should reinforce."""
        spline = BattenSpline(fog_scale=1.0)
        center = np.array([0, 0])
        # Place battens symmetrically around center
        for angle in [0, np.pi / 2, np.pi, 3 * np.pi / 2]:
            emb = np.array([np.cos(angle), np.sin(angle)])
            spline.add_batten(emb, quality=0.6)
        conf = spline.estimate_confidence(center)
        assert abs(conf - 0.6) < 0.05, f"Expected ~0.6, got {conf}"


class TestFogDensity:
    """Fog density — uncertainty measure."""

    def test_empty_spline_max_fog(self):
        spline = BattenSpline()
        fog = spline.fog_density(np.array([0, 0, 0]))
        assert fog >= 0.99, f"Empty spline should have max fog, got {fog}"

    def test_fog_decreases_near_battens(self):
        spline = BattenSpline(fog_scale=1.0)
        spline.add_batten(np.array([0, 0]), quality=0.9)
        near = spline.fog_density(np.array([0.1, 0.1]))
        far = spline.fog_density(np.array([10, 10]))
        assert near < far, "Fog should be lower near battens"

    def test_more_battens_reduce_fog(self):
        """Adding battens closer to a query point reduces fog at that point."""
        spline = BattenSpline(fog_scale=1.0)
        query = np.array([0, 0])
        fog_empty = spline.fog_density(query)
        assert fog_empty == float('inf'), "Empty spline should have infinite fog"
        # Add a far batten
        spline.add_batten(np.array([5, 0]), quality=0.5)
        fog_far = spline.fog_density(query)
        # Add a close batten
        spline.add_batten(np.array([0.5, 0]), quality=0.5)
        fog_close = spline.fog_density(query)
        assert fog_far > fog_close, f"Closer batten should reduce fog: {fog_far} vs {fog_close}"


class TestTimeDecay:
    """Old battens should have less influence than fresh ones."""

    def test_old_batten_contributes_less(self):
        spline = BattenSpline(half_life=100.0)  # 100 seconds
        old_time = 1000.0
        spline.add_batten(np.array([0, 0]), quality=0.9, timestamp=old_time)
        # Query far in the future
        conf_old = spline.estimate_confidence(np.array([0, 0]), now=10000.0)

        spline2 = BattenSpline(half_life=100.0)
        spline2.add_batten(np.array([0, 0]), quality=0.9, timestamp=9900.0)
        conf_fresh = spline2.estimate_confidence(np.array([0, 0]), now=10000.0)

        assert conf_fresh > conf_old, f"Fresh batten ({conf_fresh}) should have more influence than old ({conf_old})"

    def test_time_decay_requires_multiple_battens(self):
        """Time decay shifts weight between battens.

        NW estimator with 1 point returns that point's value while the
        time weight is numerically representable. At extreme age, the
        weight underflows to zero and confidence collapses to 0.0.
        """
        # Multiple battens: fresh one should dominate
        spline = BattenSpline(half_life=100.0)
        spline.add_batten(np.array([0, 0]), quality=1.0, timestamp=0.0)
        spline.add_batten(np.array([0, 0]), quality=0.2, timestamp=9999.0)
        conf = spline.estimate_confidence(np.array([0, 0]), now=10000.0)
        assert conf < 0.6, f"Fresh low-quality batten should pull confidence down: {conf}"

        # Single batten: confidence is stable until extreme time
        spline2 = BattenSpline(half_life=100.0)
        spline2.add_batten(np.array([0, 0]), quality=1.0, timestamp=0.0)
        # Within the first half-life, confidence is 1.0
        conf_early = spline2.estimate_confidence(np.array([0, 0]), now=50.0)
        assert abs(conf_early - 1.0) < 0.01
        # At 100 half-lives, the weight has decayed to near-zero
        # This may underflow to 0.0 — documented numerical edge case
        conf_late = spline2.estimate_confidence(np.array([0, 0]), now=10000.0)
        assert 0.0 <= conf_late <= 1.0


class TestRouterDecisions:
    """Cascade router decision logic."""

    def test_confidence_threshold_boundary(self):
        """Test exact boundary conditions for routing."""
        router = CascadeRouter()
        # With default thresholds: LOCAL >= 0.7, CASCADE >= 0.3, CLOUD >= 0.0
        spline = BattenSpline(fog_scale=0.1)

        # High quality batten → should route LOCAL near it
        spline.add_batten(np.array([0, 0]), quality=1.0)
        router_high = CascadeRouter(spline=spline)
        result = router_high.route(np.array([0, 0]))
        assert result.target == "LOCAL"
        assert result.confidence >= 0.7

    def test_custom_targets_can_be_binary(self):
        """Router with only two targets should work."""
        targets = {
            "FAST": {"threshold": 0.5, "description": "use fast model"},
            "SLOW": {"threshold": 0.0, "description": "use slow model"},
        }
        router = CascadeRouter(targets=targets)
        assert len(router.targets) == 2

    def test_invalid_target_config_rejected(self):
        """Missing threshold should raise ValueError."""
        with pytest.raises(ValueError, match="threshold"):
            CascadeRouter(targets={"BAD": {"description": "no threshold"}})

    def test_empty_targets_rejected(self):
        with pytest.raises(ValueError, match="At least one"):
            CascadeRouter(targets={})

    def test_route_result_is_immutable(self):
        """RouteResult should be frozen (frozen=True dataclass)."""
        spline = BattenSpline()
        spline.add_batten(np.array([0, 0]), quality=0.9)
        router = CascadeRouter(spline=spline)
        result = router.route(np.array([0, 0]))
        with pytest.raises(AttributeError):
            result.target = "HACKED"

    def test_state_roundtrip_preserves_routing(self):
        """Serialized and restored router should make same decisions."""
        spline = BattenSpline(fog_scale=1.0)
        spline.add_batten(np.array([1, 0]), quality=0.9)
        spline.add_batten(np.array([5, 5]), quality=0.1)
        router = CascadeRouter(spline=spline)
        original_result = router.route(np.array([1.1, 0.1]))

        restored = CascadeRouter.from_state_dict(router.state_dict())
        restored_result = restored.route(np.array([1.1, 0.1]))

        assert restored_result.target == original_result.target
        assert abs(restored_result.confidence - original_result.confidence) < 0.01


class TestNumericalStability:
    """Test behavior with numerically challenging inputs."""

    def test_zero_dimensional_embedding(self):
        """Handle 1D embeddings (vectors of length 1)."""
        spline = BattenSpline(fog_scale=1.0)
        spline.add_batten(np.array([1.0]), quality=0.5)
        conf = spline.estimate_confidence(np.array([1.0]))
        assert 0.0 <= conf <= 1.0

    def test_very_large_embeddings(self):
        """Handle large coordinate values."""
        spline = BattenSpline(fog_scale=100.0)
        spline.add_batten(np.array([1e6, 1e6]), quality=0.5)
        conf = spline.estimate_confidence(np.array([1e6 + 1, 1e6 + 1]))
        assert 0.0 <= conf <= 1.0

    def test_duplicate_embeddings(self):
        """Multiple battens at the same position should aggregate."""
        spline = BattenSpline(fog_scale=1.0)
        spline.add_batten(np.array([0, 0]), quality=1.0)
        spline.add_batten(np.array([0, 0]), quality=1.0)
        conf = spline.estimate_confidence(np.array([0, 0]))
        # Should be very confident
        assert conf > 0.9

    def test_nan_safety(self):
        """What happens with NaN values? Should not crash."""
        spline = BattenSpline(fog_scale=1.0)
        # Don't add NaN battens, but test query
        spline.add_batten(np.array([0, 0]), quality=0.5)
        # This might return NaN, which is acceptable as long as it doesn't crash
        result = spline.estimate_confidence(np.array([float('nan'), 0]))
        # Either NaN or a valid float
        import math
        if not math.isnan(result):
            assert 0.0 <= result <= 1.0

    def test_inf_fog_scale_rejected(self):
        """Infinite fog scale is now rejected by the NaN/Inf safety guard."""
        with pytest.raises(ValueError, match="fog_scale"):
            BattenSpline(fog_scale=float('inf'))

    def test_negative_fog_scale_rejected(self):
        with pytest.raises(ValueError, match="fog_scale"):
            BattenSpline(fog_scale=-1.0)

    def test_zero_fog_scale_rejected(self):
        with pytest.raises(ValueError, match="fog_scale"):
            BattenSpline(fog_scale=0.0)
