"""
Semantic Distance Calculator (Δ)
================================

Computes semantic distance between text outputs using BAAI/bge-m3 embeddings
via DeepInfra. Classifies results into zones derived from batten-spline theory:

    STALE      Δ < 0.20   — redundant, near-duplicate output
    CREATIVE   0.40–0.60  — productive divergence, novel but related
    CHAOTIC    Δ > 0.80   — completely unrelated, no shared semantic ground

The Δ metric is 1 - cos_sim(a, b), so:
    Δ = 0.0 → identical semantics
    Δ = 1.0 → orthogonal (no semantic overlap)

Usage
-----
    from experiments.delta_calculator import calculate_delta, classify_zone

    delta = calculate_delta("The cat sat on the mat.", "A feline rested on the rug.")
    print(delta)  # ~0.05 (semantically close)
    print(classify_zone(delta))  # STALE

    # Batch: pairwise distance matrix for an entire corpus
    from experiments.delta_calculator import batch_delta
    texts = [open(f).read() for f in corpus_files]
    matrix = batch_delta(texts)

CLI
---
    python -m experiments.delta_calculator --a file1.txt --b file2.txt
    python -m experiments.delta_calculator --corpus ./ai-writings/
"""

from __future__ import annotations

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEEPINFRA_EMBEDDING_URL = "https://api.deepinfra.com/v1/openai/embeddings"
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024  # bge-m3 output dimensionality

# Zone boundaries (from batten-spline theory)
STALE_THRESHOLD = 0.20
CREATIVE_LO = 0.40
CREATIVE_HI = 0.60
CHAOTIC_THRESHOLD = 0.80


# ---------------------------------------------------------------------------
# Zone Classification
# ---------------------------------------------------------------------------

def classify_zone(delta: float) -> str:
    """Classify a Δ value into STALE, CREATIVE, or CHAOTIC.

    Also returns TRANSITIONAL for the gap zones (0.20–0.40 and 0.60–0.80)
    where outputs are diverging but not yet in a defined zone.

    Parameters
    ----------
    delta : float
        Semantic distance in [0.0, 1.0].

    Returns
    -------
    str
        One of STALE, TRANSITIONAL_LOW, CREATIVE, TRANSITIONAL_HIGH, CHAOTIC.
    """
    if delta < STALE_THRESHOLD:
        return "STALE"
    elif delta < CREATIVE_LO:
        return "TRANSITIONAL_LOW"
    elif delta <= CREATIVE_HI:
        return "CREATIVE"
    elif delta <= CHAOTIC_THRESHOLD:
        return "TRANSITIONAL_HIGH"
    else:
        return "CHAOTIC"


def zone_description(zone: str) -> str:
    """Human-readable description of a zone."""
    return {
        "STALE": "Redundant — near-duplicate output, no new information",
        "TRANSITIONAL_LOW": "Diverging — minor novelty, mostly overlapping",
        "CREATIVE": "Productive divergence — novel but semantically related",
        "TRANSITIONAL_HIGH": "Strongly diverging — tenuous semantic connection",
        "CHAOTIC": "Unrelated — no shared semantic ground",
    }.get(zone, "Unknown zone")


# ---------------------------------------------------------------------------
# Embedding via DeepInfra
# ---------------------------------------------------------------------------

def _get_api_key() -> str:
    """Resolve the DeepInfra API key from env or config file."""
    key = os.environ.get("DEEPINFRA_API_KEY")
    if key:
        return key

    # Try the known config location
    config_path = Path.home() / "mcp-deeinfra" / ".env"
    if config_path.exists():
        for line in config_path.read_text().splitlines():
            if line.startswith("DEEPINFRA_API_KEY="):
                return line.split("=", 1)[1].strip()

    raise RuntimeError(
        "DEEPINFRA_API_KEY not found. Set it as an env var or "
        f"create {config_path}"
    )


def embed_text(text: str, model: str = EMBEDDING_MODEL) -> np.ndarray:
    """Embed a single text using BAAI/bge-m3 via DeepInfra.

    Returns a 1024-dimensional L2-normalized embedding vector.
    """
    api_key = _get_api_key()
    resp = requests.post(
        DEEPINFRA_EMBEDDING_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "input": text,
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    embedding = np.array(data["data"][0]["embedding"], dtype=np.float32)
    # L2 normalize so cosine similarity is just dot product
    norm = np.linalg.norm(embedding)
    if norm > 1e-12:
        embedding /= norm
    return embedding


def embed_batch(
    texts: Sequence[str],
    model: str = EMBEDDING_MODEL,
    batch_size: int = 32,
) -> np.ndarray:
    """Embed multiple texts, handling batching and rate limits.

    Returns an (N, 1024) matrix of L2-normalized embeddings.
    """
    api_key = _get_api_key()
    all_embeddings: list[np.ndarray] = []

    for i in range(0, len(texts), batch_size):
        batch = list(texts[i : i + batch_size])
        resp = requests.post(
            DEEPINFRA_EMBEDDING_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "input": batch,
            },
            timeout=60,
        )
        if resp.status_code == 429:
            # Rate limited — wait and retry
            retry_after = int(resp.headers.get("Retry-After", "5"))
            time.sleep(retry_after)
            return embed_batch(texts, model, batch_size)

        resp.raise_for_status()
        data = resp.json()
        for item in sorted(data["data"], key=lambda d: d["index"]):
            emb = np.array(item["embedding"], dtype=np.float32)
            norm = np.linalg.norm(emb)
            if norm > 1e-12:
                emb /= norm
            all_embeddings.append(emb)

    return np.stack(all_embeddings)


# ---------------------------------------------------------------------------
# Core Δ Calculations
# ---------------------------------------------------------------------------

def calculate_delta(text_a: str, text_b: str) -> float:
    """Compute the semantic distance Δ between two texts.

    Δ = 1 - cos_sim(a, b)

    Returns a float in [0.0, 1.0] where:
        0.0 = identical semantics
        1.0 = completely unrelated

    Examples
    --------
    >>> delta = calculate_delta("Hello world", "Greetings, planet")
    >>> 0.0 <= delta <= 1.0
    True
    """
    emb_a = embed_text(text_a)
    emb_b = embed_text(text_b)
    cos_sim = float(np.dot(emb_a, emb_b))
    return max(0.0, min(1.0, 1.0 - cos_sim))


def delta_from_embeddings(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """Compute Δ from pre-computed embeddings (avoids redundant API calls)."""
    cos_sim = float(np.dot(emb_a, emb_b))
    return max(0.0, min(1.0, 1.0 - cos_sim))


def batch_delta(
    texts: Sequence[str],
    embeddings: np.ndarray | None = None,
) -> np.ndarray:
    """Compute the pairwise Δ distance matrix for a list of texts.

    Parameters
    ----------
    texts : Sequence[str]
        The texts to compare.
    embeddings : np.ndarray, optional
        Pre-computed embeddings (N, D). If None, will embed via API.

    Returns
    -------
    np.ndarray
        An (N, N) symmetric matrix where matrix[i, j] = Δ(texts[i], texts[j]).
        Diagonal is 0.0 (self-distance).
    """
    if embeddings is None:
        embeddings = embed_batch(texts)

    n = len(texts)
    # Cosine similarity matrix = E @ E.T (since rows are L2-normalized)
    sim_matrix = embeddings @ embeddings.T
    # Δ = 1 - similarity, clipped to [0, 1]
    delta_matrix = np.clip(1.0 - sim_matrix, 0.0, 1.0)
    # Ensure diagonal is exactly 0
    np.fill_diagonal(delta_matrix, 0.0)
    return delta_matrix


# ---------------------------------------------------------------------------
# Corpus Analysis
# ---------------------------------------------------------------------------

def analyze_corpus(
    corpus_dir: str | Path,
    output_path: str | Path | None = None,
) -> dict:
    """Analyze an entire corpus of text files and report Δ distribution.

    Scans for .txt, .md, .json files containing text.

    Returns a dict with:
        - files: list of file paths
        - delta_matrix: flattened upper-triangle distances
        - mean_delta, median_delta, std_delta
        - zone_distribution: count of each zone
        - per_file_mean: mean Δ for each file vs all others
    """
    corpus_dir = Path(corpus_dir)
    files = sorted(
        f for f in corpus_dir.rglob("*")
        if f.is_file() and f.suffix in (".txt", ".md", ".json")
    )

    if len(files) < 2:
        raise ValueError(f"Need at least 2 files in {corpus_dir}, found {len(files)}")

    texts = []
    for f in files:
        try:
            texts.append(f.read_text(encoding="utf-8")[:8000])  # Cap at 8K chars
        except Exception:
            texts.append("")

    # Embed all at once
    embeddings = embed_batch(texts)
    matrix = batch_delta(texts, embeddings)

    # Upper triangle (excluding diagonal)
    n = len(texts)
    iu = np.triu_indices(n, k=1)
    upper_deltas = matrix[iu]

    # Zone distribution
    zones = [classify_zone(d) for d in upper_deltas]
    zone_counts: dict[str, int] = {}
    for z in zones:
        zone_counts[z] = zone_counts.get(z, 0) + 1

    # Per-file mean Δ
    per_file_mean = []
    for i in range(n):
        others = [matrix[i, j] for j in range(n) if j != i]
        per_file_mean.append(float(np.mean(others)) if others else 0.0)

    result = {
        "corpus_dir": str(corpus_dir),
        "file_count": n,
        "files": [str(f.relative_to(corpus_dir)) for f in files],
        "pair_count": len(upper_deltas),
        "mean_delta": float(np.mean(upper_deltas)),
        "median_delta": float(np.median(upper_deltas)),
        "std_delta": float(np.std(upper_deltas)),
        "min_delta": float(np.min(upper_deltas)),
        "max_delta": float(np.max(upper_deltas)),
        "zone_distribution": zone_counts,
        "per_file_mean_delta": {
            str(files[i].relative_to(corpus_dir)): per_file_mean[i]
            for i in range(n)
        },
    }

    if output_path:
        output_path = Path(output_path)
        output_path.write_text(json.dumps(result, indent=2))
        # Also save the raw matrix
        np.save(output_path.with_suffix(".npy"), matrix)

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli():
    parser = argparse.ArgumentParser(
        description="Semantic Distance Calculator (Δ) — batten-spline experiments"
    )
    sub = parser.add_subparsers(dest="command")

    # Compare two texts
    cmp = sub.add_parser("compare", help="Compare two text files")
    cmp.add_argument("--a", required=True, help="First text file")
    cmp.add_argument("--b", required=True, help="Second text file")

    # Corpus analysis
    cor = sub.add_parser("corpus", help="Analyze an entire corpus")
    cor.add_argument("--dir", required=True, help="Corpus directory")
    cor.add_argument("--output", "-o", default=None, help="Output JSON path")

    # Interactive
    inter = sub.add_parser("inline", help="Compare two inline strings")
    inter.add_argument("--a", required=True, help="First text")
    inter.add_argument("--b", required=True, help="Second text")

    args = parser.parse_args()

    if args.command in ("compare", "inline"):
        if args.command == "compare":
            text_a = Path(args.a).read_text(encoding="utf-8")
            text_b = Path(args.b).read_text(encoding="utf-8")
        else:
            text_a = args.a
            text_b = args.b

        delta = calculate_delta(text_a, text_b)
        zone = classify_zone(delta)

        print(f"Δ = {delta:.4f}")
        print(f"Zone: {zone}")
        print(f"  {zone_description(zone)}")

    elif args.command == "corpus":
        result = analyze_corpus(args.dir, args.output)
        print(f"\nCorpus: {result['corpus_dir']}")
        print(f"Files: {result['file_count']}")
        print(f"Pairs: {result['pair_count']}")
        print(f"\nΔ Distribution:")
        print(f"  Mean:   {result['mean_delta']:.4f}")
        print(f"  Median: {result['median_delta']:.4f}")
        print(f"  Std:    {result['std_delta']:.4f}")
        print(f"  Range:  [{result['min_delta']:.4f}, {result['max_delta']:.4f}]")
        print(f"\nZone Distribution:")
        for zone, count in sorted(result["zone_distribution"].items()):
            pct = 100 * count / result["pair_count"]
            print(f"  {zone:20s} {count:4d} ({pct:5.1f}%)")

        if args.output:
            print(f"\nResults saved to {args.output}")

    else:
        parser.print_help()


if __name__ == "__main__":
    _cli()
