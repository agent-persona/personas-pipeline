"""Tests for the embedding wrapper."""

import pytest


def test_embedder_importable():
    from persona_eval.embeddings import Embedder
    assert Embedder is not None


@pytest.mark.slow
def test_embed_returns_vector():
    from persona_eval.embeddings import Embedder
    embedder = Embedder()
    vec = embedder.embed("Hello world")
    assert len(vec) > 0
    assert isinstance(vec[0], float)


@pytest.mark.slow
def test_embed_batch():
    from persona_eval.embeddings import Embedder
    embedder = Embedder()
    vecs = embedder.embed_batch(["Hello", "World"])
    assert len(vecs) == 2
    assert len(vecs[0]) == len(vecs[1])


@pytest.mark.slow
def test_similarity():
    from persona_eval.embeddings import Embedder
    embedder = Embedder()
    sim = embedder.similarity("I love dogs", "I love dogs")
    assert sim > 0.9
    sim_diff = embedder.similarity("I love dogs", "Quantum physics is complex")
    assert sim_diff < sim


@pytest.mark.slow
def test_retrieval_score():
    from persona_eval.embeddings import Embedder
    embedder = Embedder()
    passages = [
        "The cat sat on the mat.",
        "Python is a programming language.",
        "Dogs are loyal companions.",
    ]
    results = embedder.retrieval_score("I love my pet dog", passages, top_k=2)
    assert len(results) == 2
    # "Dogs are loyal companions" should rank highest
    assert results[0][0] == 2
