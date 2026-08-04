import time

import numpy as np
import pytest

from batten_spline.batten import Batten


def test_distance():
    b = Batten(prompt_embedding=np.array([1.0, 0.0]))
    assert b.distance(np.array([0.0, 0.0])) == pytest.approx(1.0)


def test_age_weight_now():
    now = time.time()
    b = Batten(prompt_embedding=np.zeros(2), timestamp=now)
    assert b.age_weight(now) == pytest.approx(1.0)


def test_age_weight_decay():
    now = time.time()
    b = Batten(prompt_embedding=np.zeros(2), timestamp=now - 86400.0, half_life=86400.0)
    assert b.age_weight(now) == pytest.approx(0.5)


def test_quality_clipping():
    b = Batten(prompt_embedding=np.zeros(2), quality_score=1.5)
    assert b.quality_score == 1.0
    b2 = Batten(prompt_embedding=np.zeros(2), quality_score=-0.5)
    assert b2.quality_score == 0.0


def test_invalid_half_life():
    with pytest.raises(ValueError):
        Batten(prompt_embedding=np.zeros(2), half_life=0.0)
