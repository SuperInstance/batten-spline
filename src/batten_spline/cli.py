"""Command-line interface for batten-spline."""

from __future__ import annotations

import json
from typing import TextIO

import click
import numpy as np

from .router import CascadeRouter
from .spline import BattenSpline


@click.group()
@click.version_option(version="0.1.0", prog_name="batten-spline")
def cli() -> None:
    """Distance-weighted interpolation for cascade routing.

    Battens are verified anchor points in embedding space.  This tool
    estimates how confidently a cheaper/local model can handle a new prompt
    and decides whether to route it LOCAL, CASCADE, or CLOUD.
    """


@cli.command()
@click.option(
    "--dims",
    default=8,
    show_default=True,
    help="Dimensionality of the synthetic prompt embeddings.",
)
@click.option(
    "--battens",
    default=24,
    show_default=True,
    help="Number of synthetic anchor points to create.",
)
@click.option(
    "--fog-scale",
    default=1.0,
    show_default=True,
    help="Gaussian kernel width (embedding-space distance units).",
)
def demo(dims: int, battens: int, fog_scale: float) -> None:
    """Run a toy local-vs-cloud routing demo with synthetic embeddings."""
    router = CascadeRouter(spline=BattenSpline(fog_scale=fog_scale))
    rng = np.random.default_rng(42)

    # Region A: local model does well.
    for _ in range(battens):
        emb = rng.normal(loc=0.0, scale=0.4, size=dims)
        router.report_outcome(emb, quality=rng.uniform(0.8, 1.0))

    # Region B: local model fails; cloud is needed.
    for _ in range(battens // 2):
        emb = rng.normal(loc=5.0, scale=0.4, size=dims)
        router.report_outcome(emb, quality=rng.uniform(0.0, 0.25))

    probes = [
        ("known-local", rng.normal(loc=0.0, scale=0.4, size=dims)),
        ("known-cloud", rng.normal(loc=5.0, scale=0.4, size=dims)),
        ("unknown/fog", rng.normal(loc=10.0, scale=0.4, size=dims)),
    ]

    click.echo(f"{'probe':>12}  {'target':>8}  {'conf':>6}  {'fog':>6}")
    click.echo("-" * 42)
    for label, probe in probes:
        r = router.route(probe)
        click.echo(
            f"{label:>12}  {r.target:>8}  {r.confidence:>6.3f}  {r.fog_density:>6.3f}"
        )


@cli.command("route")
@click.argument("embedding", type=str)
@click.option(
    "--battens",
    type=click.File("r"),
    help="JSON file of battens produced by `batten-spline save-battens`.",
)
@click.option(
    "--fog-scale",
    default=1.0,
    show_default=True,
    type=float,
    help="Gaussian kernel width.",
)
@click.option(
    "--local",
    default=0.7,
    show_default=True,
    type=float,
    help="Minimum confidence for LOCAL routing.",
)
@click.option(
    "--cascade",
    default=0.3,
    show_default=True,
    type=float,
    help="Minimum confidence for CASCADE routing.",
)
def route_cmd(
    embedding: str,
    battens: TextIO | None,
    fog_scale: float,
    local: float,
    cascade: float,
) -> None:
    """Route a single prompt embedding supplied as a JSON array.

    Example:

        batten-spline route '[0.1, -0.2, 0.0, 0.4]'
    """
    router = _load_router(battens, fog_scale, local, cascade)
    x = np.array(json.loads(embedding), dtype=float)
    r = router.route(x)
    click.echo(
        json.dumps(
            {
                "target": r.target,
                "confidence": r.confidence,
                "fog_density": r.fog_density,
                "reason": r.reason,
            },
            indent=2,
        )
    )


@cli.command("save-battens")
@click.argument("output", type=click.File("w"))
@click.option(
    "--battens",
    type=click.File("r"),
    required=True,
    help="Optional input battens JSON to re-emit (demo utility).",
)
def save_battens(output: TextIO, battens: TextIO) -> None:
    """Pass-through helper to normalize a battens JSON file."""
    data = json.load(battens)
    json.dump(data, output, indent=2)


def _load_router(
    battens_file: TextIO | None,
    fog_scale: float,
    local: float,
    cascade: float,
) -> CascadeRouter:
    spline = BattenSpline(
        fog_scale=fog_scale,
        local_threshold=local,
        cascade_threshold=cascade,
    )
    router = CascadeRouter(spline=spline)
    if battens_file is not None:
        data = json.load(battens_file)
        for b in data.get("battens", []):
            router.report_outcome(
                embedding=np.array(b["embedding"], dtype=float),
                quality=b["quality"],
                metadata=b.get("metadata"),
            )
    return router


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
