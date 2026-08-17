"""The batten through an agent's warming — empirical acclimation curves.

Cross-pollination of two fleet ideas:

- **batten-spline**: verified outcomes are *battens* (anchor posts); between
  them is fog-of-war; confidence is interpolated with a Gaussian kernel and
  exponential age decay.  The shipwright's line: *"You don't make the shape.
  You let it out."*  The batten lets the fairing curve out through the points.
- **elephant's acclimation**: a newcomer warms to a room — quickly or slowly
  depending on how experienced, talented, and trained they are at modulating
  their vibe toward the room.  That modulation skill IS the rate of an
  exponential relaxation ``a(t) = room + (agent - room) * e^(-rate*t)``.

Here the batten spline is bent through an agent's *observed states* in a room,
and the shape that comes out is the agent's **empirical acclimation curve**.
From it we read:

- ``rate`` — the modulation skill (the fitted exponential rate; the spline's
  initial slope is reported alongside as a diagnostic),
- ``confidence`` — how true the shape is (the spline's own interpolation
  machinery crossed with the exponential goodness-of-fit),
- ``half_life`` — the spline's half-life concept applied to warming: the time
  to close half the gap to the room,
- ``predict_next`` — where the agent will be, governed by the skill.

If the elephant lives at ``/home/eileen/projects/elephant`` and imports
cleanly, the fit also compares its rate against
``elephant.field.acclimation_rate_from`` (the analytic endpoint inversion) and
reports the agreement.  Otherwise the fit is pure batten-spline.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .spline import BattenSpline

#: The elephant clamps its gap ratio to this floor so an agent that has
#: already overshot the room still yields a large *finite* rate.  We use the
#: same floor in the log-fit so a fully-warmed agent never poisons the rate
#: with log(0) or inf.
_LOG_FLOOR = 1e-9

#: Where the elephant lives.  Overridable with the ELEPHANT_ROOT env var so
#: the bridge is not welded to one workstation.  Best-effort: if the elephant
#: is not importable, the fit is pure batten-spline.
_ELEPHANT_ROOT = os.environ.get("ELEPHANT_ROOT", "/home/eileen/projects/elephant")

#: A start already this close to the room means "nothing to acclimate".
_ROOM_EPS = 1e-9


@dataclass
class AcclimationCurve:
    """The empirical fairing curve: a batten spline through observed states.

    ``at(t)`` returns the fitted agent state at time ``t``:

    - inside the observed span the **batten spline** is the curve — the
      fairing shape the battens let out through the points,
    - outside the observed span the **fitted exponential** takes over on
      both ends — extrapolation is governed by the modulation skill, not by
      local smoothing (and never leaves the room↔start segment).
    """

    spline: BattenSpline
    times: np.ndarray
    states: np.ndarray
    gaps: np.ndarray
    start: np.ndarray
    room: np.ndarray
    delta: np.ndarray
    rate: float
    t0: float
    t_last: float
    log_intercept: float = 0.0
    mode: str = "scalar"
    confidence: float = 0.0
    residual: float = 0.0

    def at(self, t: float) -> np.ndarray:
        """The fairing curve's state at time ``t`` (scalar or vector)."""
        t = float(t)
        if not math.isfinite(t):
            raise ValueError("curve time must be finite")
        if self.t0 <= t <= self.t_last:
            return _spline_state(self.spline, self.room, self.delta, t)
        return curve_exponential(
            self.room, self.delta, self.rate, self.log_intercept, self.t0, t
        )

    def __repr__(self) -> str:
        return (
            f"AcclimationCurve({self.mode}, n={len(self.times)}, "
            f"rate={self.rate:.4g}, t=[{self.t0:.4g}, {self.t_last:.4g}])"
        )


def _estimate(spline: BattenSpline, t: float) -> float:
    """Batten-spline confidence estimate at time ``t``.

    The ``now`` argument is essential: the spline's age weights are relative
    to a wall-clock ``now``, and our observation times are arbitrary (often
    small offsets from an epoch), so every evaluation must pin ``now`` to the
    queried time or all battens age to zero.
    """
    return float(spline.estimate_confidence(np.array([t], dtype=float), now=t))


def _exp_frac(log_intercept: float, rate: float, t0: float, t: float) -> float:
    """The fitted exponential fraction at time ``t``, clamped to [0, 1].

    The curve lives between the room and the start: an agent can overshoot,
    but the shape clamps at the room and at home.  The exponent itself is
    clamped so far-past evaluations (``t << t0`` with a large rate) can
    never overflow ``math.exp``.
    """
    exponent = min(700.0, max(-745.0, log_intercept - rate * (t - t0)))
    frac = math.exp(exponent)
    return min(1.0, max(0.0, frac))


def curve_exponential(
    room: np.ndarray, delta: np.ndarray, rate: float,
    log_intercept: float, t0: float, t: float,
) -> np.ndarray:
    """The fitted exponential state at time ``t`` (the model curve)."""
    return room + delta * _exp_frac(log_intercept, rate, t0, t)


def _spline_state(
    spline: BattenSpline, room: np.ndarray, delta: np.ndarray, t: float
) -> np.ndarray:
    """The batten-spline fairing state at time ``t`` (NW, clamped)."""
    frac = min(1.0, max(0.0, _estimate(spline, t)))
    return room + delta * frac


def _validate_inputs(
    times: Any, states: Any, room_state: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, int]:
    """Validate and normalize (times, states, room, mode, n).

    Raises ``ValueError`` with a clear message on empty / non-finite /
    non-monotonic / mismatched input — the repo's NaN/Inf safety culture:
    a non-value must never silently corrupt a fit.
    """
    t = np.asarray(times, dtype=float).reshape(-1)
    s = np.asarray(states, dtype=float)
    n = t.size
    if n < 2:
        raise ValueError("at least 2 observations are required to fit acclimation")
    if not np.all(np.isfinite(t)):
        raise ValueError("times must be finite")
    if np.any(np.diff(t) <= 0.0):
        raise ValueError("times must be strictly increasing")
    if s.shape[0] != n:
        raise ValueError(f"states and times must have the same length ({s.shape[0]} != {n})")
    if not np.all(np.isfinite(s)):
        raise ValueError("states must be finite")

    if s.ndim == 1:
        mode = "scalar"
        room = np.asarray(room_state, dtype=float)
        if room.size != 1:
            raise ValueError("scalar states need a scalar room_state")
        room = float(room)
    elif s.ndim == 2:
        mode = "vector"
        room = np.asarray(room_state, dtype=float).reshape(-1)
        if room.size != s.shape[1]:
            raise ValueError(
                f"room_state must have shape ({s.shape[1]},) to match vector states"
            )
    else:
        raise ValueError("states must be 1-D (scalar series) or 2-D (vector series)")
    if not np.all(np.isfinite(room)):
        raise ValueError("room_state must be finite")
    return t, s, room, mode, n


def _gap_series(
    states: np.ndarray, room: np.ndarray, mode: str
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Project the trajectory onto the agent→room axis.

    Returns ``(raw_gaps, axis, gap0, delta)`` where ``raw_gaps`` is the
    remaining distance from the room along the acclimation axis (state
    units), ``axis`` is the unit direction of acclimation, ``gap0`` the
    initial gap, and ``delta = start - room`` (the full vector/scalar the
    curve relaxes along).
    """
    start = states[0]
    delta = start - room
    if mode == "vector":
        gap0 = float(np.linalg.norm(delta))
        axis = delta / gap0 if gap0 >= _ROOM_EPS else np.zeros_like(delta)
        raw_gaps = (states - room) @ axis
    else:
        gap0 = float(abs(delta))
        axis = np.array([1.0])
        raw_gaps = states - room
    return raw_gaps, axis, gap0, delta


def _fit_rate(
    times: np.ndarray, raw_gaps: np.ndarray, gap0: float
) -> tuple[float, float, np.ndarray]:
    """Weighted log-fit: ``g(t) = exp(c - rate * (t - t0))``.

    Gaps are clamped to ``[floor, 1]`` exactly like the elephant clamps its
    ratio in ``acclimation_rate_from``, so an agent that has already
    overshot the room yields a large *finite* rate instead of inf.

    The regression is inverse-variance weighted (weights = g²): plain least
    squares on ``log(g)`` implicitly over-weights the small-gap points where
    log noise is largest (variance of ``log g`` scales as ``σ_g²/g²``),
    which biases the rate low.  Weighted least squares removes that bias.
    """
    t0 = float(times[0])
    x = times - t0
    clamped = np.minimum(1.0, np.maximum(_LOG_FLOOR, raw_gaps / gap0))
    y = np.log(clamped)
    w = clamped * clamped  # inverse-variance weights
    wsum = float(w.sum())
    if wsum <= 0.0:
        return 0.0, 0.0, clamped
    xw = float(np.dot(w, x)) / wsum
    yw = float(np.dot(w, y)) / wsum
    denom = float(np.dot(w, (x - xw) ** 2))
    if denom <= 1e-18:
        rate = 0.0
    else:
        rate = -float(np.dot(w, (x - xw) * (y - yw))) / denom
    c = yw + rate * xw  # y = c - rate*x
    return rate, c, clamped


def _spline_initial_slope(
    spline: BattenSpline, times: np.ndarray, fog_scale: float
) -> float:
    """Analytic derivative of the batten spline at ``t0``, as a rate.

    The Nadaraya-Watson estimator ``f(t) = Σ w_i g_i / Σ w_i`` has
    derivative ``f' = Σ w_i' (g_i - f) / Σ w_i``.  NOTE: this formula is
    specific to the Gaussian kernel + exponential age decay of
    :class:`BattenSpline`; if the kernel ever changes, it must change here
    too.  At ``t0`` the age factor is flat (every batten lies at or after
    it), so only the Gaussian part contributes: ``w_i' = w_i * (t_i - t0)/σ²``.

    The reported value is ``-f'(t0) / f(t0)`` — the exponential rate the
    spline's own initial slope implies.  A smoother (wider kernel) flattens
    this below the true rate; the fitted exponential ``rate`` is the
    primary estimate.
    """
    t0 = float(times[0])
    sig2 = fog_scale * fog_scale
    ts = np.asarray([b.prompt_embedding[0] for b in spline.battens], dtype=float)
    g = np.asarray([b.quality_score for b in spline.battens], dtype=float)
    w = np.exp(-((t0 - ts) ** 2) / (2.0 * sig2))
    total = float(w.sum())
    if total <= 1e-12:
        return 0.0
    f0 = float(np.dot(w, g) / total)
    wp = w * (ts - t0) / sig2
    fp = float(np.dot(wp, g - f0) / total)
    return -fp / max(f0, 1e-12)


def _spline_fidelity(spline: BattenSpline, times: np.ndarray, gaps: np.ndarray) -> float:
    """How truly the batten lets the observed shape out (0..1).

    Leave-everything-in cross-check: at each observed time, the spline's own
    ``estimate_confidence`` machinery re-reads the shape; the mean absolute
    error against the observed gap fractions, normalized by the gap span, is
    the shape's untruth.  Confidence = 1 - untruth.
    """
    span = float(np.max(gaps) - np.min(gaps))
    if span < 1e-9:
        return 1.0  # nothing to be untrue about — the shape is flat
    errs = [abs(_estimate(spline, float(t)) - float(g)) for t, g in zip(times, gaps)]
    mean_err = float(np.mean(errs))
    return float(np.clip(1.0 - mean_err / max(span, 1e-9), 0.0, 1.0))


def _exp_fidelity(x: np.ndarray, y: np.ndarray, m: float, c: float) -> float:
    """R² of the log-fit — how exponential the observed warming is (0..1)."""
    fitted = m * x + c
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot <= 1e-12:
        return 1.0
    return float(np.clip(1.0 - ss_res / ss_tot, 0.0, 1.0))


_elephant_tried = False
_elephant_field: Optional[Any] = None


def _elephant_module() -> Optional[Any]:
    """Best-effort import of ``elephant.field`` (cached)."""
    global _elephant_tried, _elephant_field
    if _elephant_tried:
        return _elephant_field
    _elephant_tried = True
    try:
        if os.path.isdir(_ELEPHANT_ROOT) and _ELEPHANT_ROOT not in sys.path:
            sys.path.insert(0, _ELEPHANT_ROOT)
        from elephant import field as ef

        _elephant_field = ef
    except Exception:
        _elephant_field = None
    return _elephant_field


def fit_acclimation(
    times: Any,
    states: Any,
    room_state: Any,
    *,
    fog_scale: Optional[float] = None,
    half_life: Optional[float] = None,
) -> dict[str, Any]:
    """Fit the empirical acclimation curve through an agent's observed states.

    Parameters
    ----------
    times:
        Observation times, strictly increasing and finite (length ``n``).
    states:
        Observed agent states: a 1-D scalar series (e.g. felt warmth) or a
        2-D ``(n, d)`` vector series (e.g. dial-vector states).
    room_state:
        The room's ambient state — scalar for scalar states, shape ``(d,)``
        for vector states.
    fog_scale, half_life:
        Optional batten-spline kernel parameters, in time units.  Defaults:
        ``fog_scale`` = median observation gap (neighbours have a say),
        ``half_life`` = 10× the observation span (the fairing curve hears
        all the battens — the *agent's* half-life, ``ln2 / rate``, is what
        the fit reports).

    Returns
    -------
    dict with

    - ``curve`` — the :class:`AcclimationCurve` (call ``.at(t)``),
    - ``rate`` — the modulation skill (fitted exponential rate, 1/time),
    - ``initial_slope`` — the batten spline's own initial slope at ``t0``,
    - ``confidence`` — min of the spline's interpolation fidelity and the
      log-fit R²: how true the fairing shape is,
    - ``residual`` — uniform RMSE of observed states vs the fitted curve,
      in state units,
    - ``spline_residual`` — RMSE of observed states vs the batten
      interpolation, in state units,
    - ``r_squared`` — state-space R² of the exponential model,
    - ``half_life`` — ``ln2 / rate``: time to close half the gap,
    - ``fog_scale`` / ``spline_half_life`` — the kernel parameters used,
    - ``elephant`` — bridge report: ``{"available": True, "analytic_rate",
      "agreement"}`` when the elephant imports, else ``{"available": False}``.

    NaN/Inf in any input raises ``ValueError`` — a non-value must not
    silently corrupt a fit (the repo's safety culture).
    """
    t, s, room, mode, n = _validate_inputs(times, states, room_state)

    span = float(t[-1] - t[0])
    if fog_scale is None:
        fog_scale = float(np.median(np.diff(t)))
    fog_scale = float(fog_scale)
    if not math.isfinite(fog_scale) or fog_scale <= 0.0:
        raise ValueError("fog_scale must be a positive finite number")
    if half_life is None:
        half_life = 10.0 * max(span, 1e-9)
    half_life = float(half_life)
    if not math.isfinite(half_life) or half_life <= 0.0:
        raise ValueError("half_life must be a positive finite number")

    raw_gaps, axis, gap0, delta = _gap_series(s, room, mode)
    t0, t_last = float(t[0]), float(t[-1])

    # --- Already at the room: nothing to acclimate. ------------------ #
    if gap0 < _ROOM_EPS:
        spline = BattenSpline(fog_scale=fog_scale, half_life=half_life)
        for tt in (t0, t_last):
            spline.add_batten(np.array([tt], dtype=float), 0.0, timestamp=tt,
                              half_life=half_life)
        curve = AcclimationCurve(
            spline=spline, times=t, states=s, gaps=np.zeros(n),
            start=s[0], room=room, delta=np.zeros_like(delta),
            rate=0.0, t0=t0, t_last=t_last, log_intercept=0.0, mode=mode,
            confidence=1.0, residual=0.0,
        )
        return {
            "curve": curve, "rate": 0.0, "initial_slope": 0.0,
            "confidence": 1.0, "residual": 0.0, "spline_residual": 0.0,
            "r_squared": 1.0, "half_life": math.inf, "warming": False,
            "fog_scale": fog_scale, "spline_half_life": half_life,
            "n_observations": n, "start": s[0], "room": room, "delta": delta,
            "method": "already at room", "valid": True,
            "elephant": _bridge_report(0.0, s[0], s[-1], room, t_last - t0),
        }

    # --- Fit: log-fit the gap series, fair the spline through time. -- #
    rate, log_intercept, clamped = _fit_rate(t, raw_gaps, gap0)
    gaps = raw_gaps / gap0  # unclamped fractions (for fidelity/residuals)

    spline = BattenSpline(fog_scale=fog_scale, half_life=half_life)
    for tt, g in zip(t, np.minimum(1.0, np.maximum(0.0, gaps))):
        spline.add_batten(np.array([tt], dtype=float), float(g), timestamp=tt,
                          half_life=half_life)

    # --- Confidence: the spline's own fidelity × exponential shape. -- #
    x = t - t0
    y = np.log(clamped)
    spline_fid = _spline_fidelity(spline, t, gaps)
    exp_fid = _exp_fidelity(x, y, -rate, log_intercept)
    confidence = float(min(spline_fid, exp_fid))
    if n < 3:
        # Two points determine a line, not an exponential shape: a perfect
        # two-point fit proves nothing, so the confidence is capped.
        confidence = min(confidence, 0.3)

    # --- Residuals (state units). ------------------------------------ #
    fitted = np.asarray([curve_exponential(room, delta, rate, log_intercept,
                                           t0, tt) for tt in t])
    if mode == "vector":
        resid = float(np.sqrt(np.mean(np.sum((s - fitted) ** 2, axis=1))))
        spline_res = float(np.sqrt(np.mean(np.sum(
            (s - np.asarray([_spline_state(spline, room, delta, tt)
                             for tt in t])) ** 2, axis=1))))
        ss_res = float(np.sum(np.sum((s - fitted) ** 2, axis=1)))
        ss_tot = float(np.sum(np.sum((s - s.mean(axis=0)) ** 2, axis=1)))
    else:
        resid = float(np.sqrt(np.mean((s - fitted) ** 2)))
        spline_res = float(np.sqrt(np.mean(
            (s - np.asarray([float(_spline_state(spline, room, delta, tt))
                             for tt in t])) ** 2)))
        ss_res = float(np.sum((s - fitted) ** 2))
        ss_tot = float(np.sum((s - np.mean(s)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0

    curve = AcclimationCurve(
        spline=spline, times=t, states=s, gaps=gaps,
        start=s[0], room=room, delta=delta,
        rate=rate, t0=t0, t_last=t_last, log_intercept=log_intercept,
        mode=mode, confidence=confidence, residual=resid,
    )

    half_life_out = math.inf if rate <= 0.0 else math.log(2.0) / rate

    return {
        "curve": curve,
        "rate": rate,
        "initial_slope": _spline_initial_slope(spline, t, fog_scale),
        "confidence": confidence,
        "residual": resid,
        "spline_residual": spline_res,
        "r_squared": float(np.clip(r_squared, 0.0, 1.0)),
        "half_life": half_life_out,
        "warming": rate > 0.0,
        "fog_scale": fog_scale,
        "spline_half_life": half_life,
        "n_observations": n,
        "start": s[0],
        "room": room,
        "delta": delta,
        "method": "exponential log-fit + batten spline fairing",
        "valid": True,
        "elephant": _bridge_report(rate, s[0], s[-1], room, t_last - t0),
    }


def _bridge_report(
    rate: float, start: Any, last: Any, room: Any, t_span: float
) -> dict[str, Any]:
    """Compare the fitted rate with the elephant's analytic endpoint rate."""
    ef = _elephant_module()
    if ef is None or t_span <= 0.0:
        return {"available": False}
    try:
        analytic = float(ef.acclimation_rate_from(start, last, room, t_span))
        denom = max(abs(rate), abs(analytic), 1e-12)
        agreement = 1.0 - min(abs(rate - analytic) / denom, 1.0)
        return {
            "available": True,
            "analytic_rate": analytic,
            "agreement": float(agreement),
            "note": (
                "elephant.field.acclimation_rate_from inverts the analytic "
                "relaxation from the endpoints; the fit uses all observations."
            ),
        }
    except Exception:
        return {"available": False}


def predict_next(curve: AcclimationCurve, dt: float) -> np.ndarray:
    """Where the agent will be ``dt`` after the last observation.

    Extrapolation is governed by the modulation skill: the fitted
    exponential relaxation, not local smoothing.  ``dt`` must be finite and
    non-negative.
    """
    dt = float(dt)
    if not math.isfinite(dt):
        raise ValueError("dt must be finite")
    if dt < 0.0:
        raise ValueError("dt must be non-negative")
    t = curve.t_last + dt
    return curve_exponential(
        curve.room, curve.delta, curve.rate, curve.log_intercept, curve.t0, t
    )


def skill_from_curve(curve: AcclimationCurve, room: Any) -> float:
    """Re-derive the modulation skill (rate) from a fitted curve.

    Re-runs the exponential log-fit on the curve's observed states against
    the *given* room — so you can ask "what would this agent's skill be in
    a different room?"  For the curve's own room this reproduces
    ``curve.rate`` exactly.
    """
    room = np.asarray(room, dtype=float)
    if curve.mode == "scalar":
        if room.size != 1:
            raise ValueError("scalar curve needs a scalar room_state")
        room = float(room)
    else:
        room = room.reshape(-1)
        if room.size != np.asarray(curve.room).size:
            raise ValueError("room_state shape does not match the curve's states")
    raw_gaps, _axis, gap0, _delta = _gap_series(curve.states, room, curve.mode)
    if gap0 < _ROOM_EPS:
        return 0.0
    rate, _c, _clamped = _fit_rate(curve.times, raw_gaps, gap0)
    return rate
