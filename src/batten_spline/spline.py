"""Distance-weighted interpolation between battens (Nadaraya-Watson kernel regression)."""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from .batten import Batten


class BattenSpline:
    """Estimate local-model confidence for arbitrary prompt embeddings.

    The metaphor: verified outcomes are "battens" (support posts).  Between
    them the terrain is fog-of-war.  Confidence is interpolated from nearby
    battens using a Gaussian kernel; distant or forgotten battens fade away.
    """

    def __init__(
        self,
        fog_scale: float = 1.0,
        half_life: float = 86400.0 * 7,
        local_threshold: float = 0.7,
        cascade_threshold: float = 0.3,
    ) -> None:
        self.battens: list[Batten] = []
        self.fog_scale = float(fog_scale)
        if not math.isfinite(self.fog_scale) or self.fog_scale <= 0.0:
            raise ValueError("fog_scale must be a positive finite number")
        self.half_life = float(half_life)
        if not math.isfinite(self.half_life) or self.half_life <= 0.0:
            raise ValueError("half_life must be a positive finite number")
        self.local_threshold = float(local_threshold)
        self.cascade_threshold = float(cascade_threshold)
        if not math.isfinite(self.local_threshold):
            self.local_threshold = 0.7
        if not math.isfinite(self.cascade_threshold):
            self.cascade_threshold = 0.3

    def add_batten(
        self,
        embedding: np.ndarray,
        quality: float,
        timestamp: float | None = None,
        half_life: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Batten:
        """Add a verified anchor point to the spline."""
        batten = Batten(
            prompt_embedding=np.asarray(embedding, dtype=float),
            quality_score=float(np.clip(quality, 0.0, 1.0)),
            timestamp=timestamp if timestamp is not None else time.time(),
            half_life=half_life if half_life is not None else self.half_life,
            metadata=dict(metadata) if metadata is not None else {},
        )
        self.battens.append(batten)
        return batten

    def estimate_confidence(
        self,
        new_embedding: np.ndarray,
        now: float | None = None,
    ) -> float:
        """Return the distance-and-time weighted quality estimate.

        With no battens, confidence is 0.0 (complete fog).  The estimator is
        a Nadaraya-Watson kernel regressor with a Gaussian kernel and
        exponential age decay.
        """
        if not self.battens:
            return 0.0

        now = time.time() if now is None else float(now)
        x = np.asarray(new_embedding, dtype=float)

        # Guard: if embedding contains NaN/Inf, return 0.0 (complete fog)
        if not np.all(np.isfinite(x)):
            return 0.0

        weights: list[float] = []
        scores: list[float] = []
        two_sigma2 = 2.0 * self.fog_scale**2

        for batten in self.battens:
            dist = batten.distance(x)
            age_w = batten.age_weight(now)
            w = age_w * np.exp(-(dist**2) / two_sigma2)
            weights.append(w)
            scores.append(batten.quality_score)

        weights_arr = np.array(weights, dtype=float)
        total = weights_arr.sum()
        if total < 1e-12:
            return 0.0

        return float(np.average(scores, weights=weights_arr))

    def fog_density(self, new_embedding: np.ndarray) -> float:
        """Distance to the nearest batten.  Higher = thicker fog."""
        if not self.battens:
            return float("inf")
        x = np.asarray(new_embedding, dtype=float)
        return min(batten.distance(x) for batten in self.battens)

    def routing_decision(
        self,
        confidence: float | None = None,
        new_embedding: np.ndarray | None = None,
    ) -> str:
        """Map confidence to a default routing target.

        LOCAL    : confidence >= local_threshold
        CASCADE  : cascade_threshold <= confidence < local_threshold
        CLOUD    : confidence < cascade_threshold
        """
        if confidence is None:
            if new_embedding is None:
                raise ValueError("Provide either confidence or new_embedding")
            confidence = self.estimate_confidence(new_embedding)

        # NaN/Inf confidence defaults to CLOUD (safest fallback)
        if not (isinstance(confidence, (int, float)) and math.isfinite(confidence)):
            return "CLOUD"

        if confidence >= self.local_threshold:
            return "LOCAL"
        if confidence >= self.cascade_threshold:
            return "CASCADE"
        return "CLOUD"

    def learn(
        self,
        embedding: np.ndarray,
        quality: float,
        metadata: dict[str, Any] | None = None,
    ) -> Batten:
        """Add a new verified outcome, extending the spline's reach."""
        return self.add_batten(embedding, quality, metadata=metadata)

    def prune(self, max_battens: int = 500) -> int:
        """Drop stale battens, keeping the most influential ones."""
        if len(self.battens) <= max_battens:
            return 0
        now = time.time()
        self.battens.sort(key=lambda b: b.age_weight(now), reverse=True)
        removed = len(self.battens) - max_battens
        self.battens = self.battens[:max_battens]
        return removed

    def state_dict(self) -> dict[str, Any]:
        """Serialize the spline to a JSON-friendly dictionary."""
        return {
            "fog_scale": self.fog_scale,
            "half_life": self.half_life,
            "local_threshold": self.local_threshold,
            "cascade_threshold": self.cascade_threshold,
            "battens": [
                {
                    "prompt_embedding": b.prompt_embedding.tolist(),
                    "quality_score": b.quality_score,
                    "timestamp": b.timestamp,
                    "half_life": b.half_life,
                    "metadata": b.metadata,
                }
                for b in self.battens
            ],
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "BattenSpline":
        """Restore a spline from :meth:`state_dict`."""
        spline = cls(
            fog_scale=state.get("fog_scale", 1.0),
            half_life=state.get("half_life", 86400.0 * 7),
            local_threshold=state.get("local_threshold", 0.7),
            cascade_threshold=state.get("cascade_threshold", 0.3),
        )
        for b in state.get("battens", []):
            spline.add_batten(
                embedding=np.array(b["prompt_embedding"], dtype=float),
                quality=b["quality_score"],
                timestamp=b["timestamp"],
                half_life=b.get("half_life", spline.half_life),
                metadata=b.get("metadata", {}),
            )
        return spline
