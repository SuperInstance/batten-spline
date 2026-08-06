"""
Additional tests for batten-spline CascadeRouter and BattenSpline.

Focuses on:
  - Router target ordering logic
  - Router with non-default targets
  - RouteResult dataclass properties
  - state_dict / from_state_dict with learned battens
  - report_outcome adds to spline
  - Spline routing_decision method
  - Spline learn method
  - Spline prune with max_battens
"""

import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from batten_spline.batten import Batten
from batten_spline.spline import BattenSpline
from batten_spline.router import CascadeRouter, RouteResult


class TestRouteResult(unittest.TestCase):
    """RouteResult dataclass."""

    def test_is_frozen(self):
        r = RouteResult(target="LOCAL", confidence=0.9, fog_density=0.1, reason="high confidence")
        with self.assertRaises(Exception):
            r.target = "CLOUD"  # type: ignore

    def test_fields_accessible(self):
        r = RouteResult(target="LOCAL", confidence=0.9, fog_density=0.1, reason="high confidence")
        self.assertEqual(r.target, "LOCAL")
        self.assertEqual(r.confidence, 0.9)
        self.assertEqual(r.fog_density, 0.1)
        self.assertEqual(r.reason, "high confidence")


class TestRouterTargetOrdering(unittest.TestCase):
    """Router correctly orders targets by threshold."""

    def test_picks_highest_threshold_satisfied(self):
        router = CascadeRouter()
        emb = np.array([0.5, 0.5])
        # Learn a high-quality batten so confidence is high
        for _ in range(5):
            router.report_outcome(emb, quality=1.0)

        result = router.route(emb)
        self.assertEqual(result.target, "LOCAL")
        self.assertGreater(result.confidence, 0.5)

    def test_picks_cloud_when_no_battens(self):
        router = CascadeRouter()
        emb = np.array([0.5, 0.5])
        result = router.route(emb)
        self.assertEqual(result.target, "CLOUD")

    def test_custom_two_targets(self):
        targets = {
            "MODEL_A": {"threshold": 0.5, "description": "use model A"},
            "MODEL_B": {"threshold": 0.0, "description": "use model B"},
        }
        router = CascadeRouter(targets=targets)
        emb = np.array([1.0, 0.0])
        for _ in range(5):
            router.report_outcome(emb, quality=1.0)
        result = router.route(emb)
        self.assertIn(result.target, ["MODEL_A", "MODEL_B"])

    def test_single_target(self):
        targets = {"ONLY": {"threshold": 0.0}}
        router = CascadeRouter(targets=targets)
        emb = np.array([0.5, 0.5])
        result = router.route(emb)
        self.assertEqual(result.target, "ONLY")


class TestRouterReportOutcome(unittest.TestCase):
    """report_outcome adds battens to the spline."""

    def test_report_outcome_returns_batten(self):
        router = CascadeRouter()
        emb = np.array([0.5, 0.5])
        batten = router.report_outcome(emb, quality=0.8)
        self.assertIsInstance(batten, Batten)

    def test_report_outcome_improves_confidence(self):
        router = CascadeRouter()
        emb = np.array([0.5, 0.5])
        before = router.route(emb).confidence
        for _ in range(5):
            router.report_outcome(emb, quality=1.0)
        after = router.route(emb).confidence
        self.assertGreater(after, before)

    def test_report_outcome_with_metadata(self):
        router = CascadeRouter()
        emb = np.array([0.5, 0.5])
        batten = router.report_outcome(emb, quality=0.8, metadata={"source": "test"})
        self.assertEqual(batten.metadata["source"], "test")


class TestRouterStateDict(unittest.TestCase):
    """State serialization with learned data."""

    def test_roundtrip_preserves_targets(self):
        targets = {"A": {"threshold": 0.5}, "B": {"threshold": 0.0}}
        router = CascadeRouter(targets=targets)
        state = router.state_dict()
        restored = CascadeRouter.from_state_dict(state)
        self.assertEqual(set(restored.targets.keys()), {"A", "B"})

    def test_roundtrip_preserves_battens(self):
        router = CascadeRouter()
        emb = np.array([0.5, 0.5])
        for i in range(5):
            router.report_outcome(emb, quality=0.8 + i * 0.02)
        state = router.state_dict()
        restored = CascadeRouter.from_state_dict(state)
        self.assertEqual(len(restored.spline.battens), len(router.spline.battens))

    def test_roundtrip_preserves_routing_behavior(self):
        router = CascadeRouter()
        emb = np.array([0.5, 0.5])
        for _ in range(5):
            router.report_outcome(emb, quality=1.0)
        original_result = router.route(emb)
        restored = CascadeRouter.from_state_dict(router.state_dict())
        restored_result = restored.route(emb)
        self.assertAlmostEqual(original_result.confidence, restored_result.confidence, places=3)


class TestSplineRoutingDecision(unittest.TestCase):
    """BattenSpline.routing_decision() method."""

    def test_returns_string_target(self):
        spline = BattenSpline()
        emb = np.array([0.5, 0.5])
        spline.learn(emb, quality=1.0)
        target = spline.routing_decision(new_embedding=emb)
        self.assertIsInstance(target, str)
        self.assertIn(target, ["LOCAL", "CASCADE", "CLOUD"])

    def test_empty_spline_routes_to_cloud(self):
        spline = BattenSpline()
        emb = np.array([0.5, 0.5])
        target = spline.routing_decision(new_embedding=emb)
        self.assertEqual(target, "CLOUD")

    def test_high_confidence_routes_local(self):
        spline = BattenSpline()
        emb = np.array([0.5, 0.5])
        for _ in range(5):
            spline.learn(emb, quality=1.0)
        target = spline.routing_decision(new_embedding=emb)
        self.assertEqual(target, "LOCAL")

    def test_routing_decision_with_explicit_confidence(self):
        spline = BattenSpline()
        target = spline.routing_decision(confidence=0.9)
        self.assertEqual(target, "LOCAL")


class TestSplineLearn(unittest.TestCase):
    """BattenSpline.learn() method."""

    def test_learn_adds_batten(self):
        spline = BattenSpline()
        emb = np.array([0.5, 0.5])
        initial_count = len(spline.battens)
        spline.learn(emb, quality=0.8)
        self.assertEqual(len(spline.battens), initial_count + 1)

    def test_learn_with_metadata(self):
        spline = BattenSpline()
        emb = np.array([0.5, 0.5])
        batten = spline.learn(emb, quality=0.8, metadata={"source": "test"})
        self.assertEqual(batten.metadata["source"], "test")

    def test_learn_requires_quality(self):
        spline = BattenSpline()
        emb = np.array([0.5, 0.5])
        # quality is a required argument
        with self.assertRaises(TypeError):
            spline.learn(emb)


class TestSplinePrune(unittest.TestCase):
    """BattenSpline.prune() with max_battens."""

    def test_prune_empty_does_nothing(self):
        spline = BattenSpline()
        spline.prune(max_battens=10)
        self.assertEqual(len(spline.battens), 0)

    def test_prune_under_limit_no_change(self):
        spline = BattenSpline()
        for i in range(5):
            spline.learn(np.array([float(i), 0.0]), quality=0.8)
        spline.prune(max_battens=10)
        self.assertEqual(len(spline.battens), 5)

    def test_prune_over_limit_removes_oldest(self):
        spline = BattenSpline()
        for i in range(10):
            spline.learn(np.array([float(i), 0.0]), quality=0.8)
        spline.prune(max_battens=5)
        self.assertLessEqual(len(spline.battens), 5)


if __name__ == "__main__":
    unittest.main()
