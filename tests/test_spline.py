import json
import time

import numpy as np
import pytest

from batten_spline.spline import BattenSpline


def test_empty_confidence_is_zero():
    spline = BattenSpline()
    assert spline.estimate_confidence(np.zeros(4)) == 0.0


def test_single_batten_returns_its_quality():
    spline = BattenSpline()
    spline.add_batten(np.zeros(4), quality=0.9)
    assert spline.estimate_confidence(np.zeros(4)) == pytest.approx(0.9)


def test_weighted_interpolation():
    spline = BattenSpline(fog_scale=1.0)
    spline.add_batten(np.array([0.0, 0.0]), quality=1.0)
    spline.add_batten(np.array([10.0, 0.0]), quality=0.0)
    # Exactly at the first batten.
    assert spline.estimate_confidence(np.array([0.0, 0.0])) == pytest.approx(1.0)
    # Near the first batten, confidence should stay high.
    conf = spline.estimate_confidence(np.array([0.5, 0.0]))
    assert conf > 0.75


def test_fog_density_inf_when_empty():
    spline = BattenSpline()
    assert spline.fog_density(np.zeros(3)) == float("inf")


def test_fog_density_nearest():
    spline = BattenSpline()
    spline.add_batten(np.array([3.0, 4.0]), quality=0.5)
    spline.add_batten(np.array([0.0, 0.0]), quality=0.5)
    assert spline.fog_density(np.array([1.0, 0.0])) == pytest.approx(1.0)


def test_routing_decision_thresholds():
    spline = BattenSpline(local_threshold=0.7, cascade_threshold=0.3)
    assert spline.routing_decision(confidence=0.8) == "LOCAL"
    assert spline.routing_decision(confidence=0.5) == "CASCADE"
    assert spline.routing_decision(confidence=0.1) == "CLOUD"


def test_routing_decision_requires_argument():
    spline = BattenSpline()
    with pytest.raises(ValueError):
        spline.routing_decision()


def test_learn_adds_batten():
    spline = BattenSpline()
    spline.learn(np.zeros(3), quality=0.6)
    assert len(spline.battens) == 1


def test_prune_keeps_most_recent():
    spline = BattenSpline()
    now = time.time()
    spline.add_batten(np.zeros(2), quality=0.1, timestamp=now - 1000)
    spline.add_batten(np.zeros(2), quality=0.9, timestamp=now)
    removed = spline.prune(max_battens=1)
    assert removed == 1
    assert len(spline.battens) == 1
    assert spline.battens[0].quality_score == pytest.approx(0.9)


def test_state_roundtrip():
    spline = BattenSpline(fog_scale=2.5, half_life=3600)
    spline.add_batten(np.array([1.0, 2.0]), quality=0.75)
    spline.add_batten(np.array([3.0, 4.0]), quality=0.25, metadata={"tag": "foo"})

    state = spline.state_dict()
    restored = BattenSpline.from_state_dict(state)

    assert restored.fog_scale == pytest.approx(2.5)
    assert restored.half_life == pytest.approx(3600.0)
    assert len(restored.battens) == 2
    assert restored.battens[1].metadata["tag"] == "foo"
    assert restored.estimate_confidence(np.array([1.0, 2.0])) == pytest.approx(
        spline.estimate_confidence(np.array([1.0, 2.0]))
    )


def test_state_json_serializable():
    spline = BattenSpline()
    spline.add_batten(np.array([1.0, 2.0]), quality=0.5)
    state = spline.state_dict()
    text = json.dumps(state)
    loaded = json.loads(text)
    assert loaded["battens"][0]["quality_score"] == 0.5
