# Contributing to Batten-Spline

Thanks for your interest in improving batten-spline!

## Getting Started

```bash
git clone https://github.com/SuperInstance/batten-spline.git
cd batten-spline
pip install -e ".[dev]"
```

### Prerequisites

- Python 3.10+
- NumPy >= 1.24
- Click >= 8.0 (for CLI)

## Development Workflow

```bash
# Run all tests
pytest -v

# Run specific test module
pytest tests/test_spline.py -v

# Run the demo
batten-spline demo

# Route a single embedding
batten-spline route '[0.1, -0.2, 0.0, 0.4]'
```

### Code Style

- **Python 3.10+** with `from __future__ import annotations`
- **Type hints** on all function signatures
- **Dataclasses** for structured data (`Batten`, `RouteResult`)
- **NumPy** for all numerical operations
- Docstrings on all public classes and methods
- Test every edge case: empty battens, dimension mismatch, extreme values

### Architecture

```
src/batten_spline/
├── __init__.py    — public API exports
├── batten.py      — Batten dataclass: anchor point in embedding space
├── spline.py      — BattenSpline: Nadaraya-Watson kernel regression
├── router.py      — CascadeRouter: routing policy on top of spline
└── cli.py         — Click CLI: demo, route, save-battens
```

### The Math

The estimator is a **Nadaraya-Watson kernel regressor** with:
- **Gaussian (RBF) spatial kernel** — `exp(-d²/2σ²)` weighted by embedding distance
- **Exponential temporal decay** — `0.5^(Δt/τ)` weighted by age

See the [README](README.md#the-math) for the full mathematical formulation.

### Edge Cases to Test

When contributing, test these edge cases:
1. **Empty batten list** — confidence should be 0.0, fog density infinity
2. **Dimension mismatch** — NumPy will raise; document the expectation
3. **All weights → 0** — total weight < 1e-12, return 0.0
4. **Single batten at query point** — should return that batten's quality
5. **Quality clipping** — values outside [0, 1] should be clipped
6. **Negative half_life** — should raise `ValueError`

## Adding Functionality

The router is generic — it works with any named targets, not just LOCAL/CASCADE/CLOUD. To add custom routing targets:

```python
router = CascadeRouter(targets={
    "FAST_MODEL": {"threshold": 0.8, "description": "fast but limited"},
    "MED_MODEL": {"threshold": 0.5, "description": "balanced"},
    "SLOW_MODEL": {"threshold": 0.0, "description": "powerful but slow"},
})
```

## Submitting Changes

1. Feature branch: `git checkout -b feat/your-feature`
2. Run `pytest -v` and ensure all tests pass
3. Add tests for any new functionality
4. Write clear commit messages
5. Open a PR

## Reporting Bugs

Include:
- Python and NumPy versions (`python -c "import numpy; print(numpy.__version__)"`)
- Embedding dimensions
- Number of battens
- The spline parameters (fog_scale, half_life)
- Expected vs. actual output
- Whether `prune()` was called recently

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
