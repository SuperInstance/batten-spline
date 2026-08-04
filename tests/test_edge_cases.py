"""Edge-case tests for batten-spline: empty/None inputs, extreme values, concurrency, serialization."""
import json
import threading
import time

import numpy as np
import pytest

from batten_spline.batten import Batten
from batten_spline.spline import BattenSpline
from batten_spline.router import CascadeRouter


# ── Empty / None inputs ────────────────────────────────────────


class TestEmptyAndNoneInputs:
    def test_batten_with_empty_embedding(self):
        b = Batten(prompt_embedding=np.array([]))
        assert b.distance(np.array([])) == 0.0

    def test_batten_with_zero_embedding(self):
        b = Batten(prompt_embedding=np.zeros(0))
        assert b.prompt_embedding.size == 0

    def test_spline_empty_battens_fog_density(self):
        spline = BattenSpline()
        assert spline.fog_density(np.zeros(4)) == float("inf")

    def test_spline_empty_battens_estimate(self):
        spline = BattenSpline()
        assert spline.estimate_confidence(np.zeros(4)) == 0.0

    def test_spline_empty_battens_routing(self):
        spline = BattenSpline()
        assert spline.routing_decision(confidence=0.0) == "CLOUD"

    def test_spline_add_batten_none_metadata(self):
        spline = BattenSpline()
        b = spline.add_batten(np.zeros(3), quality=0.5, metadata=None)
        assert b.metadata == {}

    def test_router_empty_spline_routes_to_cloud(self):
        router = CascadeRouter()
        r = router.route(np.array([100.0, 100.0]))
        assert r.target == "CLOUD"
        assert r.confidence == 0.0

    def test_prune_empty_spline(self):
        spline = BattenSpline()
        assert spline.prune(max_battens=10) == 0


# ── Extreme values ─────────────────────────────────────────────


class TestExtremeValues:
    def test_batten_very_large_embedding(self):
        large = np.full(100, 1e10)
        b = Batten(prompt_embedding=large)
        assert b.distance(np.zeros(100)) == pytest.approx(np.sqrt(100 * 1e20), rel=1e-6)

    def test_batten_quality_extreme_high(self):
        b = Batten(prompt_embedding=np.zeros(2), quality_score=1e100)
        assert b.quality_score == 1.0

    def test_batten_quality_extreme_low(self):
        b = Batten(prompt_embedding=np.zeros(2), quality_score=-1e100)
        assert b.quality_score == 0.0

    def test_batten_half_life_very_small(self):
        b = Batten(prompt_embedding=np.zeros(2), half_life=1e-15)
        assert b.half_life > 0

    def test_spline_extreme_fog_scale(self):
        spline = BattenSpline(fog_scale=1e10)
        spline.add_batten(np.zeros(2), quality=0.9)
        # With huge fog_scale, even distant points have high confidence
        conf = spline.estimate_confidence(np.array([100.0, 0.0]))
        assert conf > 0.5

    def test_spline_fog_scale_zero_raises(self):
        with pytest.raises(ValueError):
            BattenSpline(fog_scale=0.0)

    def test_spline_negative_fog_scale_raises(self):
        with pytest.raises(ValueError):
            BattenSpline(fog_scale=-1.0)

    def test_batten_extreme_age(self):
        b = Batten(prompt_embedding=np.zeros(2), timestamp=time.time() - 1e15)
        weight = b.age_weight()
        assert 0.0 <= weight < 1e-10

    def test_spline_many_battens(self):
        spline = BattenSpline()
        for i in range(1000):
            spline.add_batten(np.array([float(i)]), quality=float(i) / 1000.0)
        assert len(spline.battens) == 1000
        conf = spline.estimate_confidence(np.array([500.0]))
        assert 0.0 <= conf <= 1.0

    def test_batten_high_dimensional_embedding(self):
        dim = 500
        b = Batten(prompt_embedding=np.random.default_rng(0).standard_normal(dim))
        other = np.random.default_rng(1).standard_normal(dim)
        d = b.distance(other)
        assert d > 0.0

    def test_routing_decision_extreme_confidence(self):
        spline = BattenSpline(local_threshold=0.7, cascade_threshold=0.3)
        assert spline.routing_decision(confidence=1e100) == "LOCAL"
        assert spline.routing_decision(confidence=-1e100) == "CLOUD"


# ── Concurrent operations ──────────────────────────────────────


class TestConcurrency:
    def test_concurrent_add_battens(self):
        spline = BattenSpline()

        def add_batch(start: int) -> None:
            for i in range(start, start + 100):
                spline.add_batten(np.array([float(i)]), quality=0.5)

        threads = [threading.Thread(target=add_batch, args=(i * 100,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(spline.battens) == 400

    def test_concurrent_estimate_confidence(self):
        spline = BattenSpline()
        for i in range(50):
            spline.add_batten(np.array([float(i)]), quality=float(i) / 50.0)

        results: list[float] = []
        results_lock = threading.Lock()

        def estimate() -> None:
            c = spline.estimate_confidence(np.array([25.0]))
            with results_lock:
                results.append(c)

        threads = [threading.Thread(target=estimate) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 8
        assert all(0.0 <= r <= 1.0 for r in results)


# ── Serialization round-trips ──────────────────────────────────


class TestSerialization:
    def test_full_router_state_roundtrip(self):
        router = CascadeRouter()
        router.report_outcome(np.array([1.0, 2.0]), quality=0.9, metadata={"src": "test"})
        router.report_outcome(np.array([5.0, 6.0]), quality=0.1, metadata={"src": "test2"})

        state = router.state_dict()
        json_str = json.dumps(state)
        restored = CascadeRouter.from_state_dict(json.loads(json_str))

        r1 = router.route(np.array([1.5, 2.5]))
        r2 = restored.route(np.array([1.5, 2.5]))
        assert r1.target == r2.target
        assert r1.confidence == pytest.approx(r2.confidence, rel=1e-10)

    def test_state_dict_with_empty_battens(self):
        spline = BattenSpline()
        state = spline.state_dict()
        assert state["battens"] == []
        json_str = json.dumps(state)
        restored = BattenSpline.from_state_dict(json.loads(json_str))
        assert len(restored.battens) == 0

    def test_state_dict_preserves_thresholds(self):
        spline = BattenSpline(
            fog_scale=3.14,
            half_life=99,
            local_threshold=0.85,
            cascade_threshold=0.42,
        )
        state = spline.state_dict()
        restored = BattenSpline.from_state_dict(json.loads(json.dumps(state)))
        assert restored.fog_scale == pytest.approx(3.14)
        assert restored.half_life == pytest.approx(99.0)
        assert restored.local_threshold == pytest.approx(0.85)
        assert restored.cascade_threshold == pytest.approx(0.42)

    def test_state_dict_with_metadata_nested(self):
        spline = BattenSpline()
        meta = {"a": {"b": {"c": [1, 2, 3]}}, "x": None}
        spline.add_batten(np.zeros(2), quality=0.5, metadata=meta)
        state = spline.state_dict()
        json_str = json.dumps(state)
        loaded = json.loads(json_str)
        assert loaded["battens"][0]["metadata"]["a"]["b"]["c"] == [1, 2, 3]
        assert loaded["battens"][0]["metadata"]["x"] is None
