# Acclimation fit — the batten through an agent's warming

*Cross-pollination: the elephant's acclimation curve, fairied through a
batten spline.  The shipwright's line — "You don't make the shape. You let
it out." — is the whole method: the battens are the agent's observed states,
and the fairing curve that comes out between them is the agent's character.*

---

## The batten as the shipwright's fairing curve

In a shipyard lofting floor, a shipwright does not draw a hull by decree.
They drive a handful of **battens** — thin wooden strips bent against
weighted posts — through the known points, and the wood finds the fairest
curve the points will allow.  No point is exactly nailed; each bends the
strip a little, and the shape that comes out is truer than any single
measurement.  *You don't make the shape. You let it out.*

`batten-spline` has been bending battens through prompt embeddings to read
model confidence.  This module bends a batten through **an agent's observed
states in a room** — and the shape that comes out is the agent's *empirical
acclimation curve*.

## The acclimation curve as the agent's character

The elephant (JEPA-is-the-elephant reframing) says: a newcomer warms to a
room — quickly or slowly depending on how experienced, talented, and
trained they are at modulating their vibe toward the room.  That modulation
skill is a rate, and the curve is an exponential relaxation:

```
a(t) = room + (agent − room) · e^(−rate·t)
```

The elephant's `field.py` has the *analytic* version of this curve and its
inverse (`acclimation_rate_from`): given where an agent started, where they
are now, and the room, recover the skill.  It is a two-point inversion —
one observation, one rate.

But an agent is observed *many* times.  Their warmth at t=0, t=1, t=2, …
are points on a lofting floor.  Drive battens through them and let the
shape out: that is `fit_acclimation`.  The fairing curve is the agent's
empirical warming, and from it we read:

| Quantity | Meaning |
|----------|---------|
| `rate` | the **modulation skill** — fitted exponential rate (1/time) |
| `initial_slope` | the batten spline's own slope at t₀ (smoothed; a diagnostic) |
| `confidence` | how true the shape is — the spline's fidelity × exponential R² |
| `half_life` | `ln2 / rate` — the time to close half the gap to the room |
| `residual` | RMSE of observed states vs the fitted curve (state units) |
| `curve` | the fairing curve itself: `curve.at(t)` anywhere |

## The bridge: empirical curve vs analytic relaxation

`elephant.field.acclimation_rate_from(start, obs, room, t)` inverts the
analytic curve from the endpoints.  The batten fit uses *all* observations
with an inverse-variance weighted log-fit.  When the elephant is
importable, every fit reports a bridge:

```python
res["elephant"]  # {'available': True, 'analytic_rate': 0.3599,
                 #  'agreement': 0.955, 'note': ...}
```

On clean exponential data the two rates agree closely (agreement > 0.9);
with noise the all-points fit is the more robust read.  The agreement is
the empirical curve shaking hands with the analytic relaxation — the same
shape, measured two ways.  If the elephant is missing, the fit is pure
batten-spline (`available: False`), and `ELEPHANT_ROOT` overrides where the
bridge looks.

## The math, briefly

1. **Project onto the acclimation axis.**  For vector states, the warming
   happens along `delta = start − room`; each observation is projected onto
   that axis and normalized by the initial gap, giving gap fractions
   `g_i ∈ (0, 1]` — the fraction of the original gap still left at t_i.

2. **Weighted log-fit.**  `ln g = c − rate·(t − t₀)`, fitted by
   inverse-variance weighted least squares (weights `g²`), because plain
   least squares on logs over-weights the small-gap points where log noise
   is largest.  Gaps are clamped to `[1e-9, 1]` — the *same* clamp the
   elephant uses — so an agent that has already overshot the room yields a
   large finite rate instead of `inf`.

3. **The batten spline.**  The gap fractions become battens in time space
   (embedding `[t]`, quality `g`).  The spline's own machinery —
   Gaussian kernel + exponential age decay + `estimate_confidence` — is the
   fairing curve *inside* the observed span.  Every evaluation pins
   `now=t` so the age weights stay relative to the queried time.

4. **Confidence.**  Two truths must hold: the spline must faithfully
   reproduce the observed states (interpolation fidelity, via the spline's
   own estimator) *and* the shape must look exponential (R² of the log-fit).
   `confidence = min(fidelity, R²)` — conservative by construction.  With
   only two observations, confidence is capped at 0.3: two points prove a
   line, not an exponential shape.

5. **Prediction.**  Outside the observed span, the fitted exponential takes
   over (both tails) — the future is governed by the skill, not by local
   smoothing.  `predict_next(curve, dt)` is where the agent will be `dt`
   after the last observation; the curve never leaves the room↔start
   segment.

### The elephant's pulse, one level up

The same one-math as `perception_math.py` applies to the *gap series*
itself: two readings give direction (first difference — is the agent
warming?), three give rate of change (second difference — warming faster or
slower, cascading or exhausting).  The batten fit reads the integrated
shape; the second difference reads its acceleration.  Same derivative
ladder, two instruments.

## Usage

```python
import numpy as np
from batten_spline import fit_acclimation, predict_next, skill_from_curve

room  = np.array([0.55, 0.6, 0.5, 0.45, 0.5, 0.5, 0.5])
start = np.array([0.1, 0.2, 0.15, 0.3, 0.25, 0.2, 0.2])
t = np.linspace(0.0, 10.0, 12)
states = room + np.outer(np.exp(-0.35 * t), start - room)   # true rate 0.35

res = fit_acclimation(t, states, room)
res["rate"]          # 0.344  (the modulation skill)
res["confidence"]    # 0.98   (the shape is true)
res["half_life"]     # 2.0 s  (time to close half the gap)
res["elephant"]      # {'available': True, 'analytic_rate': 0.36, 'agreement': 0.96}

predict_next(res["curve"], 2.0)          # where the agent will be
skill_from_curve(res["curve"], room)     # re-derive the skill
```

Scalar series work the same way — pass a 1-D `states` and a scalar `room`.

## Safety culture

A non-value must not silently corrupt a fit.  NaN/Inf in `times`, `states`,
or `room_state` raises a clear `ValueError`; times must be strictly
increasing; there must be at least two observations; `fog_scale` and
`half_life` must be positive and finite; `predict_next` rejects non-finite
or negative `dt`; and the exponential is exponent-clamped so even a
far-past query on a fast-warming agent can never overflow.  An agent that
drifts *away* from the room is not warming: the rate reads 0 and the
half-life `inf` — exactly what the elephant's own clamp reports.

## Files

- `src/batten_spline/acclimation_fit.py` — the module
- `tests/test_acclimation_fit.py` — 35 tests: warming/slow agents, vector
  states, residuals, prediction, skill re-derivation, overshoot, the
  elephant bridge, and the NaN/Inf crusade
