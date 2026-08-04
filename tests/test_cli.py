import json

import numpy as np
from click.testing import CliRunner

from batten_spline.cli import cli


def test_cli_demo():
    runner = CliRunner()
    result = runner.invoke(cli, ["demo"])
    assert result.exit_code == 0
    assert "known-local" in result.output
    assert "known-cloud" in result.output


def test_cli_route():
    runner = CliRunner()
    result = runner.invoke(cli, ["route", "[0.0, 0.0, 0.0, 0.0]"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["target"] == "CLOUD"  # no battens -> total fog
    assert data["confidence"] == 0.0


def test_cli_route_with_battens_file():
    runner = CliRunner()
    battens = {
        "battens": [
            {"embedding": [0.0, 0.0, 0.0], "quality": 0.95},
        ]
    }
    with runner.isolated_filesystem():
        with open("battens.json", "w") as f:
            json.dump(battens, f)
        result = runner.invoke(
            cli, ["route", "[0.0, 0.0, 0.0]", "--battens", "battens.json"]
        )
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["target"] == "LOCAL"
