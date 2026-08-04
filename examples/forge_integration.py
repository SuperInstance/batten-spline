#!/usr/bin/env python3
"""
BattenSpline + Slackwater Forge Integration

Demonstrates batten-spline deciding whether to run a forge job locally
(via Ollama) or cascade to a cloud model. This is the real-world pattern:
the spline learns which kinds of prompts the local GPU handles well and
routes the rest to the cloud.

Decision flow:
    1. Embed the incoming prompt
    2. Ask BattenSpline for confidence
    3. LOCAL (>=0.7)  → run on Ollama directly
    4. CASCADE (>=0.3) → try Ollama, verify output, escalate if poor
    5. CLOUD (<0.3)   → send to cloud API immediately

Run:  python forge_integration.py
"""

from __future__ import annotations

import numpy as np

from batten_spline import BattenSpline, CascadeRouter


# ── Mock forge job types ─────────────────────────────────────────

class ForgeJob:
    """A simplified forge job for demonstration."""
    def __init__(self, name: str, prompt: str, model: str, priority: str = "medium"):
        self.name = name
        self.prompt = prompt
        self.model = model
        self.priority = priority

    def embed(self, dims: int = 16) -> np.ndarray:
        """Deterministic mock embedding."""
        rng = np.random.default_rng(hash(self.prompt) % (2**32))
        return rng.normal(loc=0.0, scale=0.5, size=dims)


def fake_local_generate(job: ForgeJob) -> tuple[str, float]:
    """Pretend to run the job on Ollama; return (text, quality_estimate)."""
    # In real life: call slackwater_forge.models.OllamaClient.generate()
    if "code review" in job.prompt.lower():
        quality = 0.93
        text = "[Local] Code looks clean. No issues found."
    elif "creative" in job.prompt.lower():
        quality = 0.25
        text = "[Local] The wind blew. The end. (Low quality — local model can't write.)"
    else:
        quality = 0.60
        text = "[Local] Generic output."
    return text, quality


def fake_cloud_generate(job: ForgeJob) -> str:
    """Pretend to call a cloud API."""
    return f"[Cloud] High-quality response for '{job.name}'."


# ── Integration: batten-spline routing forge jobs ────────────────

def main() -> None:
    # 1. Create and seed the router
    spline = BattenSpline(fog_scale=1.2, local_threshold=0.7, cascade_threshold=0.3)
    router = CascadeRouter(spline=spline)

    # Seed with historical data: the local model is great at code review
    code_review_prompts = [
        "Review this Lua code for bugs",
        "Check this Python script for performance",
        "Audit this Roblox Luau module",
    ]
    for p in code_review_prompts:
        job = ForgeJob("seed", p, "granite3.1-dense:2b")
        router.report_outcome(job.embed(), quality=0.92)

    # Seed: the local model is bad at creative writing
    creative_prompts = [
        "Write a atmospheric vignette about a fishing village",
        "Compose a poem about the ocean at dusk",
        "Write dialogue for two old sailors remembering a storm",
    ]
    for p in creative_prompts:
        job = ForgeJob("seed", p, "granite3.1-dense:2b")
        router.report_outcome(job.embed(), quality=0.20)

    # 2. Incoming overnight forge jobs
    incoming_jobs = [
        ForgeJob("code_audit", "Review this Lua code for bugs and best practices",
                 "granite3.1-dense:2b", priority="high"),
        ForgeJob("harbor_story", "Write a creative vignette about a foggy harbor at dawn",
                 "granite3.1-dense:2b", priority="medium"),
        ForgeJob("unknown_task", "Generate a recipe for sourdough bread",
                 "granite3.1-dense:2b", priority="low"),
    ]

    print("═" * 72)
    print("  Forge Integration — BattenSpline Routing Decisions")
    print("═" * 72)

    for job in incoming_jobs:
        emb = job.embed()
        decision = router.route(emb)

        print(f"\n📋 Job: {job.name}")
        print(f"   Prompt: {job.prompt[:60]}...")
        print(f"   Route: {decision.target} (confidence={decision.confidence:.3f}, "
              f"fog={decision.fog_density:.3f})")
        print(f"   Reason: {decision.reason}")

        # Execute based on routing decision
        if decision.target == "LOCAL":
            # High confidence — run locally, no verification needed
            text, quality = fake_local_generate(job)
            print(f"   ✅ Executed locally → {text[:60]}...")
            # Report actual quality back to the spline
            router.report_outcome(emb, quality=quality)
            print(f"   📈 Reported quality={quality:.2f} to spline")

        elif decision.target == "CASCADE":
            # Medium confidence — try local, verify, escalate if needed
            text, quality = fake_local_generate(job)
            print(f"   ⚡ Tried locally → quality={quality:.2f}")
            if quality < 0.5:
                print(f"   ⬆️  Quality too low — escalating to cloud")
                text = fake_cloud_generate(job)
                print(f"   ☁️  Executed on cloud → {text[:60]}...")
                # Report failure so spline learns
                router.report_outcome(emb, quality=quality)
            else:
                router.report_outcome(emb, quality=quality)

        else:  # CLOUD
            # Low confidence — skip local entirely
            text = fake_cloud_generate(job)
            print(f"   ☁️  Executed on cloud → {text[:60]}...")
            # We don't know the quality yet; the spline stays as-is.
            # In production, you'd report quality after reviewing output.

    # 3. Show what the spline learned
    print(f"\n{'═' * 72}")
    print(f"  Spline state: {len(spline.battens)} battens")
    print(f"  Pruned to top 500? {len(spline.battens) <= 500}")
    print(f"{'═' * 72}")


if __name__ == "__main__":
    main()
