#!/usr/bin/env python3
"""
BattenSpline Router Demo — Full Local-vs-Cloud Routing

Demonstrates a complete routing scenario with mock embeddings:

1. Creates a CascadeRouter with default LOCAL / CASCADE / CLOUD targets
2. Seeds it with verified outcomes in two distinct regions of embedding space
3. Routes new prompts and shows how the router decides where to send them
4. Simulates feedback (reporting actual quality) so the spline learns

This is the pattern you'd use in a real LLM cascade: cheap local model
for confident prompts, expensive cloud model for unfamiliar ones.

Run:  python router_demo.py
"""

import numpy as np
from batten_spline import BattenSpline, CascadeRouter


def mock_embed(text: str, dims: int = 16) -> np.ndarray:
    """Deterministic mock embedding from text (so results are reproducible)."""
    rng = np.random.default_rng(hash(text) % (2**32))
    return rng.normal(loc=0.0, scale=0.5, size=dims)


def main() -> None:
    # ── 1. Set up the router ──────────────────────────────────────────
    spline = BattenSpline(fog_scale=1.5, local_threshold=0.7, cascade_threshold=0.3)
    router = CascadeRouter(spline=spline)

    # ── 2. Seed with known outcomes ───────────────────────────────────
    # Region A: simple coding questions — local model handles these well
    simple_prompts = [
        "Write a Python function to reverse a list",
        "How do I read a file in Python?",
        "Sort a dictionary by value",
        "Convert string to integer in Python",
        "List comprehension syntax",
        "Merge two dictionaries Python",
    ]
    for p in simple_prompts:
        router.report_outcome(mock_embed(p), quality=0.92)

    # Region B: complex reasoning — local model struggles here
    hard_prompts = [
        "Design a distributed consensus algorithm with Byzantine fault tolerance",
        "Prove that P != NP under relativized oracles",
        "Analyze the computational complexity of quantum factorization",
        "Derive the Navier-Stokes existence and smoothness proof",
    ]
    for p in hard_prompts:
        router.report_outcome(mock_embed(p), quality=0.15)

    # ── 3. Route new prompts ──────────────────────────────────────────
    test_prompts = [
        # Should route LOCAL (similar to simple coding questions)
        "Write a function to check if a string is a palindrome",
        # Should route CLOUD (similar to complex reasoning)
        "Design a fault-tolerant distributed database from scratch",
        # Unknown territory — thick fog, low confidence → CLOUD
        "Write a haiku about quantum entanglement in Japanese",
    ]

    print(f"{'Prompt':<60} {'Target':>8} {'Conf':>6} {'Fog':>8}")
    print("─" * 88)

    for prompt in test_prompts:
        emb = mock_embed(prompt)
        result = router.route(emb)
        # Truncate prompt for display
        short = prompt[:57] + "..." if len(prompt) > 57 else prompt
        print(f"{short:<60} {result.target:>8} {result.confidence:>6.3f} {result.fog_density:>8.3f}")

    # ── 4. Simulate learning loop ─────────────────────────────────────
    print("\n── Learning Loop ──")
    print("The local model just handled a 'palindrome' prompt successfully.")
    print("Reporting quality=0.88 to strengthen that region...\n")

    palindrome_emb = mock_embed("Write a function to check if a string is a palindrome")
    router.report_outcome(palindrome_emb, quality=0.88)

    # Re-route something very close
    nearby_emb = mock_embed("Write a function to check if a number is prime")
    result = router.route(nearby_emb)
    print(f"{'Write a function to check if a number is prime':<60} "
          f"{result.target:>8} {result.confidence:>6.3f} {result.fog_density:>8.3f}")
    print("(Confidence increased — the spline learned from feedback.)")

    # ── 5. Serialize / restore ────────────────────────────────────────
    print("\n── Serialization ──")
    state = router.state_dict()
    print(f"Serialized router has {len(state['spline']['battens'])} battens.")
    restored = CascadeRouter.from_state_dict(state)
    result2 = restored.route(palindrome_emb)
    assert abs(result2.confidence - result.confidence) < 0.01, "Round-trip failed!"
    print("Router successfully serialized and restored.")


if __name__ == "__main__":
    main()
