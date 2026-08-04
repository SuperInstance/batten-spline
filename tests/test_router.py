import numpy as np
import pytest

from batten_spline.router import CascadeRouter
from batten_spline.spline import BattenSpline


def test_default_route_local():
    router = CascadeRouter()
    router.report_outcome(np.zeros(4), quality=0.9)
    r = router.route(np.zeros(4))
    assert r.target == "LOCAL"
    assert r.confidence > 0.7


def test_default_route_cloud():
    router = CascadeRouter()
    router.report_outcome(np.zeros(4), quality=0.1)
    r = router.route(np.array([100.0, 0.0, 0.0, 0.0]))
    assert r.target == "CLOUD"


def test_default_route_cascade():
    router = CascadeRouter(spline=BattenSpline(fog_scale=1.0))
    # One high-quality region and one low-quality region.
    router.report_outcome(np.zeros(4), quality=1.0)
    router.report_outcome(np.full(4, 6.0), quality=0.0)
    # Exactly midway, the estimate interpolates to ~0.5 -> CASCADE.
    r = router.route(np.full(4, 3.0))
    assert r.target == "CASCADE"


def test_custom_targets():
    targets = {
        "BIG": {"threshold": 0.9},
        "SMALL": {"threshold": 0.0, "description": "fallback"},
    }
    router = CascadeRouter(targets=targets)
    router.report_outcome(np.zeros(2), quality=0.95)
    r = router.route(np.zeros(2))
    assert r.target == "BIG"


def test_target_requires_threshold():
    with pytest.raises(ValueError):
        CascadeRouter(targets={"BAD": {}})


def test_router_state_roundtrip():
    router = CascadeRouter()
    router.report_outcome(np.array([1.0, 2.0]), quality=0.8)
    state = router.state_dict()
    restored = CascadeRouter.from_state_dict(state)
    r = restored.route(np.array([1.0, 2.0]))
    assert r.target == "LOCAL"
    assert r.confidence > 0.7
