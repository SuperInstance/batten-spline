"""Acclimation-fit tests — the batten through an agent's warming.

The elephant's acclimation curve (exponential relaxation, rate = modulation
skill) meets the batten spline: observed states become battens, the fairing
curve is the agent's empirical warming, and the fit reads the skill back.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from batten_spline.acclimation_fit import (
    AcclimationCurve,
    _elephant_module,
    fit_acclimation,
    predict_next,
    skill_from_curve,
)
from batten_spline.spline import BattenSpline


def _exp_states(rate, t, room=0.0, start=1.0, noise=0.0, seed=0):
    """Exact exponential relaxation, optionally with gaussian noise."""
    t = np.asarray(t, dtype=float)
    s = room + (start - room) * np.exp(-rate * t)
    if noise:
        s = s + np.random.default_rng(seed).normal(0.0, noise, size=s.shape)
    return s


class TestWarmingAgent:
    def test_positive_rate_high_confidence(self):
        t = np.linspace(0.0, 6.0, 9)
        states = _exp_states(0.4, t, room=0.2, start=1.0, noise=0.004)
        res = fit_acclimation(t, states, 0.2)
        assert res["rate"] == pytest.approx(0.4, abs=0.06)
        assert res["rate"] > 0.0
        assert res["confidence"] > 0.7
        assert res["r_squared"] > 0.99
        assert math.isfinite(res["half_life"]) and res["half_life"] > 0.0

    def test_initial_slope_is_warming_and_bounded_by_rate(self):
        """The spline's own initial slope is a smoothed under-estimate of
        the true rate (the fitted exponential is the primary estimate)."""
        t = np.linspace(0.0, 5.0, 7)
        states = _exp_states(0.3, t, room=0.1, start=0.9)
        res = fit_acclimation(t, states, 0.1)
        assert res["initial_slope"] > 0.0
        assert res["initial_slope"] <= res["rate"] * 1.01
        assert res["initial_slope"] == pytest.approx(0.3, abs=0.25)

    def test_vector_states_fit_along_the_axis(self):
        t = np.linspace(0.0, 5.0, 8)
        room = np.array([0.4, 0.6, 0.5])
        start = np.array([0.0, 0.1, 0.2])
        delta = start - room
        g = np.exp(-0.35 * t)
        states = room + np.outer(g, delta)
        res = fit_acclimation(t, states, room)
        assert res["rate"] == pytest.approx(0.35, abs=0.05)
        assert res["confidence"] > 0.7
        assert res["n_observations"] == 8
        # extrapolation follows the same relaxation
        dt = 1.0
        expected = room + delta * np.exp(-0.35 * (t[-1] + dt))
        assert np.allclose(predict_next(res["curve"], dt), expected, atol=0.02)

    def test_curve_is_a_batten_spline(self):
        t = np.linspace(0.0, 4.0, 6)
        states = _exp_states(0.5, t, room=0.0, start=1.0)
        res = fit_acclimation(t, states, 0.0)
        assert isinstance(res["curve"], AcclimationCurve)
        assert isinstance(res["curve"].spline, BattenSpline)
        assert len(res["curve"].spline.battens) == 6

    def test_at_tracks_observed_states_inside_span(self):
        """Inside the observed span the fairing curve hugs the truth (it is
        a smoother, not an interpolator, so a loose tolerance)."""
        t = np.linspace(0.0, 5.0, 11)
        states = _exp_states(0.4, t, room=0.2, start=1.0)
        res = fit_acclimation(t, states, 0.2)
        for tt, obs in zip(t[1:-1], states[1:-1]):
            assert res["curve"].at(tt) == pytest.approx(obs, abs=0.08)

    def test_at_never_leaves_the_room_start_segment(self):
        """Even for far-past or far-future times the curve clamps between
        the room and the start — and never overflows the exponential."""
        t = np.array([0.0, 1.0, 2.0, 3.0])
        states = _exp_states(0.5, t, room=0.3, start=1.0)
        res = fit_acclimation(t, states, 0.3)
        assert res["curve"].at(-1000.0) == pytest.approx(1.0, abs=1e-6)
        assert res["curve"].at(1e6) == pytest.approx(0.3, abs=1e-6)
        assert math.isfinite(float(res["curve"].at(-1000.0)))


class TestSlowAgent:
    def test_slow_agent_low_rate(self):
        t = np.linspace(0.0, 20.0, 11)
        states = _exp_states(0.02, t, room=0.0, start=1.0)
        res = fit_acclimation(t, states, 0.0)
        assert 0.0 < res["rate"] < 0.05
        assert res["half_life"] > 10.0  # ln2 / 0.02 ≈ 34.7
        assert math.isfinite(res["half_life"])

    def test_two_points_suffice(self):
        """Two observations determine an exponential — the elephant's own
        rate_from uses exactly the endpoints.  But confidence is capped:
        two points prove a line, not an exponential shape."""
        t = np.array([0.0, 5.0])
        states = _exp_states(0.25, t, room=0.0, start=1.0)
        res = fit_acclimation(t, states, 0.0)
        assert res["rate"] == pytest.approx(0.25, abs=1e-6)
        assert res["confidence"] <= 0.3

    def test_drifting_agent_is_not_warming(self):
        """An agent drifting away from the room (gap ratio > 1) is not
        warming.  The clamp mirrors the elephant exactly, so the rate reads
        0 — not negative — and the half-life is infinite."""
        t = np.array([0.0, 1.0, 2.0, 3.0])
        states = np.array([0.70, 0.72, 0.71, 0.73])  # room at 0, drifting up
        res = fit_acclimation(t, states, 0.0)
        assert res["rate"] == 0.0
        assert res["warming"] is False
        assert res["half_life"] == math.inf

    def test_curve_carries_confidence_and_residual(self):
        t = np.linspace(0.0, 4.0, 6)
        states = _exp_states(0.5, t, room=0.0, start=1.0)
        res = fit_acclimation(t, states, 0.0)
        assert res["curve"].confidence == res["confidence"]
        assert res["curve"].residual == res["residual"]

    def test_weighted_fit_reduces_log_space_bias(self):
        """The log-fit is inverse-variance weighted, so small-gap points
        (where log noise explodes) do not drag the rate low."""
        t = np.linspace(0.0, 6.0, 12)
        states = _exp_states(0.4, t, room=0.0, start=1.0, noise=0.02)
        res = fit_acclimation(t, states, 0.0)
        assert res["rate"] == pytest.approx(0.4, abs=0.08)


class TestResidual:
    def test_residual_small_for_clean_exponential(self):
        t = np.linspace(0.0, 4.0, 10)
        states = _exp_states(0.5, t, room=0.0, start=1.0)
        res = fit_acclimation(t, states, 0.0)
        # model residual: the exponential is the true generating curve
        assert res["residual"] < 1e-6
        # spline residual: the NW fairing curve is a *smoother*, not an
        # interpolator — it hugs the truth rather than nailing each point
        assert res["spline_residual"] < 0.05
        assert res["spline_residual"] > 0.0

    def test_residual_grows_with_noise(self):
        t = np.linspace(0.0, 4.0, 10)
        clean = _exp_states(0.5, t, room=0.0, start=1.0)
        noisy = _exp_states(0.5, t, room=0.0, start=1.0, noise=0.08)
        r_clean = fit_acclimation(t, clean, 0.0)
        r_noisy = fit_acclimation(t, noisy, 0.0)
        assert r_noisy["residual"] > r_clean["residual"]
        assert r_noisy["confidence"] < r_clean["confidence"]


class TestPredictNext:
    def test_extrapolates_sensibly(self):
        rate, room, start = 0.3, 0.2, 1.0
        t = np.linspace(0.0, 5.0, 6)
        states = _exp_states(rate, t, room=room, start=start)
        res = fit_acclimation(t, states, room)
        dt = 3.0
        expected = room + (start - room) * math.exp(-rate * (t[-1] + dt))
        assert predict_next(res["curve"], dt) == pytest.approx(expected, abs=0.02)

    def test_zero_dt_is_the_last_state(self):
        t = np.array([0.0, 1.0, 2.0, 3.0])
        states = _exp_states(0.4, t, room=0.0, start=1.0)
        res = fit_acclimation(t, states, 0.0)
        assert predict_next(res["curve"], 0.0) == pytest.approx(states[-1], abs=0.01)

    def test_converges_to_room(self):
        t = np.linspace(0.0, 4.0, 6)
        states = _exp_states(0.5, t, room=0.3, start=1.0)
        res = fit_acclimation(t, states, 0.3)
        assert predict_next(res["curve"], 100.0) == pytest.approx(0.3, abs=1e-6)


class TestSkillFromCurve:
    def test_matches_curve_rate(self):
        t = np.linspace(0.0, 5.0, 8)
        states = _exp_states(0.25, t, room=0.1, start=0.8)
        res = fit_acclimation(t, states, 0.1)
        assert skill_from_curve(res["curve"], 0.1) == pytest.approx(res["rate"], abs=1e-9)

    def test_different_room_reference_changes_the_skill(self):
        """The reference room is part of the read: same trajectory, a
        different room yields a different derived skill.  A room the
        trajectory crosses reads as fast overshoot — the elephant's own
        clamp semantics (large but finite rate)."""
        t = np.linspace(0.0, 5.0, 8)
        states = _exp_states(0.25, t, room=0.1, start=0.8)
        res = fit_acclimation(t, states, 0.1)
        skill_same = skill_from_curve(res["curve"], 0.1)
        assert skill_same == pytest.approx(res["rate"], abs=1e-9)
        skill_warm = skill_from_curve(res["curve"], 0.15)
        assert math.isfinite(skill_warm) and skill_warm > 0.0
        assert skill_warm != skill_same
        skill_crossed = skill_from_curve(res["curve"], 0.6)  # states fall below 0.6
        assert math.isfinite(skill_crossed)
        assert skill_crossed > skill_warm

    def test_vector_skill_from_curve(self):
        t = np.linspace(0.0, 5.0, 8)
        room = np.array([0.4, 0.6, 0.5])
        start = np.array([0.0, 0.1, 0.2])
        delta = start - room
        states = room + np.outer(np.exp(-0.35 * t), delta)
        res = fit_acclimation(t, states, room)
        assert skill_from_curve(res["curve"], room) == pytest.approx(res["rate"], abs=1e-9)

    def test_room_shape_mismatch_raises(self):
        t = np.linspace(0.0, 4.0, 6)
        states = np.stack([_exp_states(0.5, t, room=0.0, start=1.0),
                           _exp_states(0.4, t, room=0.0, start=1.0)], axis=1)
        res = fit_acclimation(t, states, np.zeros(2))
        with pytest.raises(ValueError, match="shape"):
            skill_from_curve(res["curve"], np.zeros(3))


class TestAlreadyAtRoom:
    def test_agent_at_room_zero_rate_perfect_confidence(self):
        t = np.array([0.0, 1.0, 2.0])
        states = np.full(3, 0.4)
        res = fit_acclimation(t, states, 0.4)
        assert res["rate"] == 0.0
        assert res["confidence"] == 1.0
        assert res["half_life"] == math.inf
        assert res["curve"].at(5.0) == pytest.approx(0.4, abs=1e-9)
        assert res["residual"] == 0.0


class TestElephantBridge:
    def test_bridge_contract(self):
        """Available → agreement is high on clean exponential data;
        unavailable → the fit is still pure and complete."""
        t = np.linspace(0.0, 6.0, 7)
        states = _exp_states(0.3, t, room=0.0, start=1.0)
        res = fit_acclimation(t, states, 0.0)
        bridge = res["elephant"]
        assert "available" in bridge
        if bridge["available"]:
            assert bridge["analytic_rate"] == pytest.approx(0.3, abs=0.05)
            assert bridge["agreement"] > 0.9
        else:
            assert res["rate"] == pytest.approx(0.3, abs=0.05)

    def test_pure_mode_when_elephant_missing(self, monkeypatch):
        monkeypatch.setattr("batten_spline.acclimation_fit._elephant_module",
                            lambda: None)
        t = np.linspace(0.0, 5.0, 6)
        states = _exp_states(0.4, t, room=0.0, start=1.0)
        res = fit_acclimation(t, states, 0.0)
        assert res["elephant"] == {"available": False}
        assert res["rate"] == pytest.approx(0.4, abs=0.05)
        assert res["confidence"] > 0.7


class TestOvershoot:
    def test_overshoot_reads_as_large_finite_rate(self):
        """An agent that passes the room yields a large *finite* rate and a
        finite half-life — the elephant's own clamp semantics, never inf."""
        t = np.array([0.0, 1.0, 2.0, 3.0])
        # start 1.0, room 0.0; the agent crashes through the room to -0.2
        states = np.array([1.0, 0.4, 0.05, -0.2])
        res = fit_acclimation(t, states, 0.0)
        assert math.isfinite(res["rate"]) and res["rate"] > 0.0
        assert math.isfinite(res["half_life"])
        # the curve clamps at the room: it never dips below it
        assert float(res["curve"].at(3.5)) >= -1e-9
        assert float(res["curve"].at(100.0)) >= -1e-9


class TestNaNSafety:
    """A non-value must not silently corrupt a fit — it raises cleanly."""

    def test_nan_times_raise(self):
        with pytest.raises(ValueError, match="finite"):
            fit_acclimation(np.array([0.0, np.nan, 2.0]),
                            np.array([1.0, 0.5, 0.3]), 0.0)

    def test_inf_states_raise(self):
        with pytest.raises(ValueError, match="finite"):
            fit_acclimation(np.array([0.0, 1.0, 2.0]),
                            np.array([1.0, np.inf, 0.3]), 0.0)

    def test_nan_room_raises(self):
        with pytest.raises(ValueError, match="finite"):
            fit_acclimation(np.array([0.0, 1.0]),
                            np.array([1.0, 0.5]), float("nan"))

    def test_nan_in_vector_states_raise(self):
        with pytest.raises(ValueError, match="finite"):
            fit_acclimation(np.array([0.0, 1.0]),
                            np.array([[1.0, 0.0], [np.nan, 0.5]]),
                            np.array([0.0, 0.0]))

    def test_non_monotonic_times_raise(self):
        with pytest.raises(ValueError, match="increasing"):
            fit_acclimation(np.array([0.0, 2.0, 1.0]),
                            np.array([1.0, 0.5, 0.3]), 0.0)

    def test_duplicate_times_raise(self):
        with pytest.raises(ValueError, match="increasing"):
            fit_acclimation(np.array([0.0, 1.0, 1.0]),
                            np.array([1.0, 0.5, 0.3]), 0.0)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length"):
            fit_acclimation(np.array([0.0, 1.0, 2.0]),
                            np.array([1.0, 0.5]), 0.0)

    def test_too_few_points_raise(self):
        with pytest.raises(ValueError, match="2"):
            fit_acclimation(np.array([0.0]), np.array([1.0]), 0.0)

    def test_bad_fog_scale_raises(self):
        t = np.array([0.0, 1.0, 2.0])
        states = _exp_states(0.4, t, room=0.0, start=1.0)
        with pytest.raises(ValueError, match="fog_scale"):
            fit_acclimation(t, states, 0.0, fog_scale=float("nan"))
        with pytest.raises(ValueError, match="half_life"):
            fit_acclimation(t, states, 0.0, half_life=-1.0)

    def test_predict_next_bad_dt_raises(self):
        t = np.array([0.0, 1.0, 2.0])
        res = fit_acclimation(t, _exp_states(0.4, t, room=0.0, start=1.0), 0.0)
        with pytest.raises(ValueError, match="finite"):
            predict_next(res["curve"], float("nan"))
        with pytest.raises(ValueError, match="non-negative"):
            predict_next(res["curve"], -1.0)

    def test_curve_at_bad_time_raises(self):
        t = np.array([0.0, 1.0, 2.0])
        res = fit_acclimation(t, _exp_states(0.4, t, room=0.0, start=1.0), 0.0)
        with pytest.raises(ValueError, match="finite"):
            res["curve"].at(float("inf"))
