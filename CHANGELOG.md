# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Acclimation fit** — the batten through an agent's warming (cross-pollinated from the elephant)
  - `fit_acclimation(times, states, room_state)` — batten spline through observed states, exponential log-fit for the rate (the modulation skill), confidence from the spline's own fidelity × R², residual, half-life, and a best-effort bridge to `elephant.field.acclimation_rate_from`
  - `AcclimationCurve` — the fairing curve: batten spline inside the observed span, fitted exponential outside; clamps between room and start
  - `predict_next(curve, dt)` — where the agent will be, governed by the skill
  - `skill_from_curve(curve, room)` — re-derive the rate, even against a different room reference
  - Inverse-variance weighted log-fit (no OLS-on-log bias), elephant-consistent gap clamp (`[1e-9, 1]`), NaN/Inf safety raising clean `ValueError`s, confidence capped for two-point fits
- Hero image + mermaid in README; full writeup in `docs/acclimation-fit.md`

## [0.1.0] — 2026-08-04

### Added
- **Batten** — dataclass for verified anchor points in embedding space with Euclidean distance and exponential age-decay weighting
- **BattenSpline** — Nadaraya-Watson kernel regression estimator with Gaussian spatial kernel and temporal decay
  - `add_batten()` / `learn()` — register verified outcomes
  - `estimate_confidence()` — distance-and-time-weighted quality estimate
  - `fog_density()` — distance to nearest batten (exploration metric)
  - `routing_decision()` — map confidence to LOCAL/CASCADE/CLOUD
  - `prune()` — drop stalest battens to bound memory
  - `state_dict()` / `from_state_dict()` — JSON serialization
- **CascadeRouter** — deployable routing policy built on BattenSpline
  - `route()` — returns `RouteResult(target, confidence, fog_density, reason)`
  - `report_outcome()` — feedback loop for self-improvement
  - Configurable targets with custom thresholds
  - Serialization support
- **CLI** (`batten-spline`) — Click-based commands
  - `demo` — synthetic local-vs-cloud routing demo
  - `route` — route a single embedding from JSON array
  - `save-battens` — normalize battens JSON
- 258 tests covering spline estimation, batten lifecycle, router policy, CLI, serialization, and edge cases
- MIT license
