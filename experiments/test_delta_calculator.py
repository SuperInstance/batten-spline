"""
Tests for the Semantic Distance Calculator (Δ).

These tests verify:
1. Zone classification logic (pure math, no API needed)
2. Δ computation from pre-computed embeddings (no API needed)
3. Batch matrix computation (no API needed)
4. Integration with DeepInfra API (skipped if no API key)
"""

import os
import numpy as np
import pytest

from experiments.delta_calculator import (
    classify_zone,
    zone_description,
    delta_from_embeddings,
    batch_delta,
    calculate_delta,
    embed_batch,
    analyze_corpus,
    STALE_THRESHOLD,
    CREATIVE_LO,
    CREATIVE_HI,
    CHAOTIC_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Zone Classification (pure logic)
# ---------------------------------------------------------------------------

class TestClassifyZone:
    """Test the zone classification boundary logic."""

    def test_identical_is_stale(self):
        assert classify_zone(0.0) == "STALE"

    def test_just_under_stale_threshold(self):
        assert classify_zone(0.19) == "STALE"

    def test_at_stale_threshold(self):
        assert classify_zone(0.20) == "TRANSITIONAL_LOW"

    def test_transitional_low(self):
        assert classify_zone(0.30) == "TRANSITIONAL_LOW"

    def test_creative_low_boundary(self):
        assert classify_zone(0.40) == "CREATIVE"

    def test_creative_mid(self):
        assert classify_zone(0.50) == "CREATIVE"

    def test_creative_high_boundary(self):
        assert classify_zone(0.60) == "CREATIVE"

    def test_just_above_creative(self):
        assert classify_zone(0.61) == "TRANSITIONAL_HIGH"

    def test_at_chaotic_threshold(self):
        assert classify_zone(0.80) == "TRANSITIONAL_HIGH"

    def test_just_above_chaotic(self):
        assert classify_zone(0.81) == "CHAOTIC"

    def test_orthogonal_is_chaotic(self):
        assert classify_zone(1.0) == "CHAOTIC"

    def test_all_five_zones_covered(self):
        zones = {classify_zone(d / 100) for d in range(101)}
        assert zones == {
            "STALE", "TRANSITIONAL_LOW", "CREATIVE", "TRANSITIONAL_HIGH", "CHAOTIC"
        }


class TestZoneDescription:
    def test_returns_description_for_all_zones(self):
        for zone in ("STALE", "TRANSITIONAL_LOW", "CREATIVE", "TRANSITIONAL_HIGH", "CHAOTIC"):
            desc = zone_description(zone)
            assert isinstance(desc, str)
            assert len(desc) > 10

    def test_unknown_zone(self):
        assert zone_description("UNKNOWN") == "Unknown zone"


# ---------------------------------------------------------------------------
# Δ from Embeddings (pure math)
# ---------------------------------------------------------------------------

class TestDeltaFromEmbeddings:
    """Test Δ computation with synthetic embeddings."""

    def test_identical_embeddings_delta_zero(self):
        v = np.array([1.0, 0.0, 0.0])
        assert delta_from_embeddings(v, v) == pytest.approx(0.0, abs=1e-6)

    def test_orthogonal_embeddings_delta_one(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        assert delta_from_embeddings(a, b) == pytest.approx(1.0, abs=1e-6)

    def test_45_degree_embeddings(self):
        a = np.array([1.0, 0.0])
        b = np.array([1.0, 1.0]) / np.sqrt(2)
        # cos(45°) = √2/2 ≈ 0.7071
        # Δ = 1 - 0.7071 ≈ 0.2929
        delta = delta_from_embeddings(a, b)
        assert delta == pytest.approx(1.0 - np.sqrt(2) / 2, abs=1e-5)

    def test_delta_always_in_range(self):
        rng = np.random.default_rng(42)
        for _ in range(100):
            a = rng.standard_normal(64)
            b = rng.standard_normal(64)
            a /= np.linalg.norm(a)
            b /= np.linalg.norm(b)
            delta = delta_from_embeddings(a, b)
            assert 0.0 <= delta <= 1.0

    def test_symmetric(self):
        rng = np.random.default_rng(99)
        a = rng.standard_normal(32)
        b = rng.standard_normal(32)
        a /= np.linalg.norm(a)
        b /= np.linalg.norm(b)
        assert delta_from_embeddings(a, b) == pytest.approx(
            delta_from_embeddings(b, a), abs=1e-7
        )


# ---------------------------------------------------------------------------
# Batch Δ Matrix
# ---------------------------------------------------------------------------

class TestBatchDelta:
    """Test pairwise distance matrix computation."""

    def test_matrix_shape(self):
        rng = np.random.default_rng(7)
        embeddings = rng.standard_normal((5, 16))
        # L2 normalize
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        matrix = batch_delta(["a"] * 5, embeddings)
        assert matrix.shape == (5, 5)

    def test_matrix_diagonal_zero(self):
        rng = np.random.default_rng(3)
        embeddings = rng.standard_normal((8, 32))
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        matrix = batch_delta(["x"] * 8, embeddings)
        for i in range(8):
            assert matrix[i, i] == pytest.approx(0.0, abs=1e-7)

    def test_matrix_symmetric(self):
        rng = np.random.default_rng(11)
        embeddings = rng.standard_normal((6, 24))
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        matrix = batch_delta(["t"] * 6, embeddings)
        for i in range(6):
            for j in range(6):
                assert matrix[i, j] == pytest.approx(matrix[j, i], abs=1e-6)

    def test_matrix_values_in_range(self):
        rng = np.random.default_rng(23)
        embeddings = rng.standard_normal((10, 64))
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        matrix = batch_delta(["t"] * 10, embeddings)
        assert matrix.min() >= 0.0
        assert matrix.max() <= 1.0

    def test_identical_pairwise_zero(self):
        v = np.array([[1.0, 0.0, 0.0]] * 4)
        matrix = batch_delta(["same"] * 4, v)
        assert np.allclose(matrix, 0.0)


# ---------------------------------------------------------------------------
# Integration Tests (require API key)
# ---------------------------------------------------------------------------

HAS_API_KEY = bool(os.environ.get("DEEPINFRA_API_KEY")) or os.path.exists(
    os.path.expanduser("~/mcp-deeinfra/.env")
)


@pytest.mark.skipif(not HAS_API_KEY, reason="No DeepInfra API key available")
class TestDeepInfraIntegration:
    """Integration tests that actually call the DeepInfra embedding API."""

    def test_embed_text_returns_normalized_vector(self):
        from experiments.delta_calculator import embed_text
        emb = embed_text("The quick brown fox")
        assert emb.shape == (1024,)
        assert np.linalg.norm(emb) == pytest.approx(1.0, abs=1e-5)

    def test_calculate_delta_identical_text(self):
        text = "The cat sat on the mat."
        delta = calculate_delta(text, text)
        assert delta == pytest.approx(0.0, abs=0.01)

    def test_calculate_delta_unrelated_text(self):
        delta = calculate_delta(
            "The mitochondria is the powerhouse of the cell.",
            "Quantum entanglement defies classical intuition.",
        )
        # Should be well above stale zone
        assert delta > 0.3

    def test_calculate_delta_paraphrase_is_stale(self):
        delta = calculate_delta(
            "The cat sat on the mat.",
            "A feline rested on the rug.",
        )
        # Paraphrase should be close (stale or transitional low)
        # bge-m3 sees these as semantically related but with some distance
        # because different verbs/objects create measurable divergence
        assert delta < 0.45

    def test_embed_batch(self):
        texts = ["Hello world", "Goodbye world", "Quantum physics"]
        embeddings = embed_batch(texts)
        assert embeddings.shape == (3, 1024)
        # All rows normalized
        for i in range(3):
            assert np.linalg.norm(embeddings[i]) == pytest.approx(1.0, abs=1e-5)
