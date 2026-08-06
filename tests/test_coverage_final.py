"""Coverage gap tests — targeting the last uncovered lines."""

import io
import json
import sys
from unittest.mock import patch

import numpy as np
import pytest

from batten_spline.router import CascadeRouter, RouteResult
from batten_spline.spline import BattenSpline
from batten_spline.batten import Batten
from batten_spline import cli as cli_module


class TestRouterValidation:
    """router.py:63 — _validate_targets raises for missing threshold."""

    def test_missing_threshold_raises(self):
        with pytest.raises(ValueError, match="missing required 'threshold'"):
            CascadeRouter(targets={"BAD": {"description": "no threshold"}})

    def test_empty_targets_raises(self):
        with pytest.raises(ValueError, match="At least one routing target"):
            CascadeRouter(targets={})

    def test_valid_targets_no_error(self):
        router = CascadeRouter(targets={"A": {"threshold": 0.5}})
        assert router.targets["A"]["threshold"] == 0.5


class TestRouterFallbackTarget:
    """router.py:91 — fallback to lowest-threshold target.

    This line is only reachable if confidence is negative (below 0 threshold).
    Since confidence can technically be negative with certain spline configurations,
    we test the _pick_target directly.
    """

    def test_pick_target_negative_confidence_falls_back(self):
        """If all thresholds are positive and confidence is negative, use fallback."""
        router = CascadeRouter(
            targets={
                "HIGH": {"threshold": 0.9},
                "MID": {"threshold": 0.5},
                "LOW": {"threshold": 0.1},
            }
        )
        # Confidence below all thresholds
        target, reason = router._pick_target(-0.5)
        assert target == "LOW"  # falls back to lowest-threshold

    def test_pick_target_zero_confidence_with_zero_threshold(self):
        """With a zero-threshold target, confidence=0 should match it."""
        router = CascadeRouter()  # default has CLOUD at threshold 0.0
        target, reason = router._pick_target(0.0)
        # CLOUD has threshold 0.0, CASCADE has 0.3, LOCAL has 0.7
        # At confidence 0.0, the highest threshold satisfied is CLOUD (0.0)
        assert target == "CLOUD"

    def test_pick_target_returns_description(self):
        router = CascadeRouter(
            targets={"ONLY": {"threshold": 0.0, "description": "the only target"}}
        )
        target, reason = router._pick_target(0.5)
        assert reason == "the only target"

    def test_pick_target_missing_description_returns_empty(self):
        router = CascadeRouter(
            targets={"ONLY": {"threshold": 0.0}}
        )
        target, reason = router._pick_target(0.5)
        assert reason == ""


class TestSaveBattens:
    """cli.py:142-143 — save_battens pass-through helper."""

    def test_save_battens_normalizes_json(self):
        """save_battens should read JSON from battens and write to output."""
        from click.testing import CliRunner
        import tempfile, os

        battens_data = [
            {"embedding": [1.0, 2.0], "quality": 0.8, "metadata": {}},
            {"embedding": [3.0, 4.0], "quality": 0.6, "metadata": {"tag": "test"}},
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as bf:
            json.dump(battens_data, bf)
            battens_path = bf.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as of:
            output_path = of.name

        try:
            runner = CliRunner()
            result = runner.invoke(cli_module.cli, ["save-battens", output_path, "--battens", battens_path])
            assert result.exit_code == 0, f"CLI failed: {result.output}\n{result.exception}"

            with open(output_path) as f:
                data = json.load(f)
            assert isinstance(data, list)
            assert len(data) == 2
            assert data[0]["quality"] == 0.8
        finally:
            os.unlink(battens_path)
            os.unlink(output_path)

    def test_save_battens_preserves_data(self):
        """The function should be a pass-through — data preserved."""
        from click.testing import CliRunner
        import tempfile, os

        original = {"test": [1, 2, 3], "nested": {"a": "b"}}

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as bf:
            json.dump(original, bf)
            battens_path = bf.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as of:
            output_path = of.name

        try:
            runner = CliRunner()
            result = runner.invoke(cli_module.cli, ["save-battens", output_path, "--battens", battens_path])
            assert result.exit_code == 0, f"CLI failed: {result.output}"

            with open(output_path) as f:
                data = json.load(f)
            assert data == original
        finally:
            os.unlink(battens_path)
            os.unlink(output_path)

    def test_save_battens_output_is_indented(self):
        """Output should be indented JSON."""
        from click.testing import CliRunner
        import tempfile, os

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as bf:
            json.dump({"a": 1}, bf)
            battens_path = bf.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as of:
            output_path = of.name

        try:
            runner = CliRunner()
            result = runner.invoke(cli_module.cli, ["save-battens", output_path, "--battens", battens_path])
            assert result.exit_code == 0, f"CLI failed: {result.output}"

            with open(output_path) as f:
                content = f.read()
            assert "\n" in content
        finally:
            os.unlink(battens_path)
            os.unlink(output_path)


class TestRouterReportOutcomeEdgeCases:
    """Additional router tests for report_outcome."""

    def test_report_outcome_with_metadata(self):
        router = CascadeRouter()
        emb = np.array([1.0, 0.5])
        batten = router.report_outcome(emb, quality=0.9, metadata={"model": "test"})
        assert isinstance(batten, Batten)
        assert batten.metadata["model"] == "test"

    def test_report_outcome_without_metadata(self):
        router = CascadeRouter()
        emb = np.array([0.3, 0.7])
        batten = router.report_outcome(emb, quality=0.5)
        assert isinstance(batten, Batten)

    def test_state_dict_roundtrip(self):
        router = CascadeRouter()
        emb = np.array([1.0, 0.0])
        router.report_outcome(emb, quality=0.8)

        state = router.state_dict()
        restored = CascadeRouter.from_state_dict(state)

        assert restored.targets == router.targets
        # The spline state should be preserved
        result = restored.route(emb)
        assert isinstance(result, RouteResult)


class TestRouteResultProperties:
    """RouteResult dataclass behavior."""

    def test_route_result_is_frozen(self):
        result = RouteResult(target="LOCAL", confidence=0.9, fog_density=0.1, reason="test")
        with pytest.raises(AttributeError):
            result.target = "CLOUD"

    def test_route_result_fields(self):
        result = RouteResult(target="CLOUD", confidence=0.2, fog_density=0.8, reason="foggy")
        assert result.target == "CLOUD"
        assert result.confidence == 0.2
        assert result.fog_density == 0.8
        assert result.reason == "foggy"
