"""High-level cascade router built on top of BattenSpline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .batten import Batten
from .spline import BattenSpline


@dataclass(frozen=True)
class RouteResult:
    """The outcome of a routing decision."""

    target: str
    confidence: float
    fog_density: float
    reason: str


class CascadeRouter:
    """Turn a BattenSpline into a deployable routing policy.

    The default targets are LOCAL / CASCADE / CLOUD, but the router is
    deliberately generic: you can supply any named targets with thresholds,
    making it useful for local-vs-cloud, model-A-vs-model-B, or any other
    embedding-based routing problem.
    """

    DEFAULT_TARGETS: dict[str, dict[str, Any]] = {
        "LOCAL": {
            "threshold": 0.7,
            "description": "local model is reliable in this neighbourhood",
        },
        "CASCADE": {
            "threshold": 0.3,
            "description": "try local model first, escalate to cloud if weak",
        },
        "CLOUD": {
            "threshold": 0.0,
            "description": "unfamiliar territory; go straight to cloud",
        },
    }

    def __init__(
        self,
        spline: BattenSpline | None = None,
        targets: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.spline = spline if spline is not None else BattenSpline()
        self.targets = (
            {k: dict(v) for k, v in targets.items()}
            if targets is not None
            else dict(self.DEFAULT_TARGETS)
        )
        self._validate_targets()

    def _validate_targets(self) -> None:
        if not self.targets:
            raise ValueError("At least one routing target must be configured")
        for name, cfg in self.targets.items():
            if "threshold" not in cfg:
                raise ValueError(f"Target {name!r} is missing required 'threshold'")

    def route(self, embedding: np.ndarray) -> RouteResult:
        """Route a single prompt embedding."""
        confidence = self.spline.estimate_confidence(embedding)
        fog = self.spline.fog_density(embedding)
        target, reason = self._pick_target(confidence)
        return RouteResult(
            target=target,
            confidence=confidence,
            fog_density=fog,
            reason=reason,
        )

    def _pick_target(self, confidence: float) -> tuple[str, str]:
        """Select the highest-threshold target that the confidence satisfies."""
        ordered = sorted(
            self.targets.items(),
            key=lambda item: item[1]["threshold"],
            reverse=True,
        )
        for name, cfg in ordered:
            if confidence >= cfg["threshold"]:
                return name, cfg.get("description", "")
        # Fallback to the lowest-threshold target.
        return ordered[-1][0], ordered[-1][1].get("description", "")

    def report_outcome(
        self,
        embedding: np.ndarray,
        quality: float,
        metadata: dict[str, Any] | None = None,
    ) -> Batten:
        """Record the actual quality of a route so the spline can learn."""
        return self.spline.learn(embedding, quality, metadata=metadata)

    def state_dict(self) -> dict[str, Any]:
        """Serialize the router to a JSON-friendly dictionary."""
        return {
            "targets": self.targets,
            "spline": self.spline.state_dict(),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "CascadeRouter":
        """Restore a router from :meth:`state_dict`."""
        spline = BattenSpline.from_state_dict(state["spline"])
        return cls(spline=spline, targets=state.get("targets"))
