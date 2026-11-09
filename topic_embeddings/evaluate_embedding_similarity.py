"""Compare pairwise topic similarities against a reference embedding space."""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


TOPIC_COLUMNS = ["Topic_1", "Topic_2", "Topic_3"]


def _indexed_vectors(frame, prefix):
    """Return a DataFrame of vectors indexed by topic paths."""
    paths = frame[TOPIC_COLUMNS].fillna("")
    index = pd.MultiIndex.from_frame(paths)
    if not index.is_unique:
        raise ValueError(f"Duplicate topic paths in {prefix} coordinates.")
    vectors = frame.filter(regex=f"^{prefix}").set_axis(index)
    if vectors.shape[1] == 0:
        raise ValueError(f"Expected coordinate columns starting with {prefix!r}.")
    return vectors


def _cosine_matrix(vectors):
    """Return a DataFrame of pairwise cosine similarities for the given vectors."""
    values = vectors.to_numpy(dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or (norms == 0).any():
        raise ValueError("Shared topics must have finite, nonzero embedding norms.")
    unit = values / norms
    return pd.DataFrame(unit @ unit.T, index=vectors.index, columns=vectors.index)


def evaluate_embedding_similarity(topics, reference):
    """Return aligned cosine matrices (topics, reference) and their Spearman rho.

    Match full topic paths using embedding_* columns from topics and dim_*
    columns from reference (e.g. comment means or pretrained topic labels).
    Compare each unordered shared-topic pair once, excluding self-pairs.
    The two embedding spaces may have different dimensionalities.
    """
    topic_vectors = _indexed_vectors(topics, "embedding_")
    reference_vectors = _indexed_vectors(reference, "dim_")
    shared = topic_vectors.index.intersection(reference_vectors.index, sort=False)
    if len(shared) < 3:
        raise ValueError("Spearman correlation needs at least three shared topics.")
    topic_cosines = _cosine_matrix(topic_vectors.loc[shared])
    reference_cosines = _cosine_matrix(reference_vectors.loc[shared])
    upper = np.triu_indices(len(shared), k=1) # upper triangle indices for strict pairs
    scores = [matrix.to_numpy()[upper] for matrix in (topic_cosines, reference_cosines)]
    if any(np.ptp(values) == 0 for values in scores):
        raise ValueError("Spearman correlation is undefined for constant similarities.")
    return topic_cosines, reference_cosines, float(spearmanr(*scores).statistic)
