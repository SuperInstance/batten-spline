"""Verified anchor points (battens) in embedding space."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Batten:
    """A single verified outcome in prompt-embedding space.

    A batten is a truth you have already measured: for a given prompt
    embedding, the local model produced quality ``quality_score``.  Its
    influence on future routing predictions decays exponentially with age,
    so recent feedback matters more than stale feedback.
    """

    prompt_embedding: np.ndarray
    quality_score: float = field(default=0.5)
    timestamp: float = field(default_factory=time.time)
    half_life: float = field(default=86400.0 * 7)  # one week, in seconds
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.prompt_embedding = np.asarray(self.prompt_embedding, dtype=float)
        self.quality_score = float(np.clip(self.quality_score, 0.0, 1.0))
        self.half_life = float(self.half_life)
        if self.half_life <= 0.0:
            raise ValueError("half_life must be positive")

    def age_weight(self, now: float | None = None) -> float:
        """Exponential-decay weight: 0.5^((now - timestamp) / half_life)."""
        now = time.time() if now is None else float(now)
        dt = now - self.timestamp
        if dt <= 0.0:
            return 1.0
        return 0.5 ** (dt / self.half_life)

    def distance(self, other_embedding: np.ndarray) -> float:
        """Euclidean distance from this batten to another embedding."""
        other = np.asarray(other_embedding, dtype=float)
        return float(np.linalg.norm(self.prompt_embedding - other))
