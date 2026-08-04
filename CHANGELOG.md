# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
