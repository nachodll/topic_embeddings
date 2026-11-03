"""Explicit shared-user PPMI and PCA, used by notebooks and grid search."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA

TOPIC_COLUMNS = ["Topic_1", "Topic_2", "Topic_3"]

def count_topic_users(comment_chunks, article_mapping, n_topics, verbose=False):
    """Accumulate full-dataset counts; return sparse counts and processing statistics."""
    users = {}
    counts = csr_matrix((n_topics, 0), dtype=np.int64)
    stats = dict(comment_rows=0, valid_comments=0, excluded_unknown_articles=0,
                 excluded_blank_authors=0)
    for chunk_number, chunk in enumerate(comment_chunks, start=1):
        topic_indices = chunk["Article_Id"].map(article_mapping)
        authors = chunk["Author_Id"].str.strip()
        known = topic_indices.notna()
        nonblank = authors.ne("")
        valid = known & nonblank
        stats["comment_rows"] += len(chunk)
        stats["excluded_unknown_articles"] += int((~known).sum())
        stats["excluded_blank_authors"] += int((known & ~nonblank).sum())
        stats["valid_comments"] += int(valid.sum())
        valid_authors = authors.loc[valid]
        for author in valid_authors.unique():
            if author not in users:
                users[author] = len(users)
        counts.resize((n_topics, len(users)))
        chunk_counts = csr_matrix(
            (np.ones(int(valid.sum()), dtype=np.int64),
             (topic_indices.loc[valid].to_numpy(dtype=np.int64),
              valid_authors.map(users).to_numpy(dtype=np.int64))),
            shape=counts.shape,
        )
        counts = counts + chunk_counts
        if verbose and chunk_number % 10 == 0:
            print(f"Processed {stats['comment_rows']:,} comments; {len(users):,} authors.", flush=True)
    if counts.sum() != stats["valid_comments"]:
        raise AssertionError("Sparse counts do not match the valid-comment total.")
    stats.update(users=len(users), topic_user_relationships=counts.nnz)
    return counts, stats

def build_context_counts(counts, cutoff=5, context_dimensions=500):
    presence = (counts > cutoff).astype(np.int64)
    active_users = np.asarray(presence.sum(axis=1)).ravel()
    cooccurrence = (presence @ presence.T).tocsr()
    assert np.array_equal(cooccurrence.diagonal(), active_users)
    cooccurrence.setdiag(0)
    cooccurrence.eliminate_zeros()
    retained_indices = np.flatnonzero(np.asarray(cooccurrence.sum(axis=1)).ravel() > 0)
    if len(retained_indices) < context_dimensions:
        raise ValueError(f"Need {context_dimensions} overlapping topics; found {len(retained_indices)}.")
    context_indices = retained_indices[
        np.argsort(-active_users[retained_indices], kind="stable")[:context_dimensions]
    ]
    context_counts = cooccurrence[retained_indices][:, context_indices].toarray().astype(np.float64)
    return context_counts, retained_indices, context_indices, active_users

def normalize_rows(values):
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    normalized = np.divide(values, norms, out=np.zeros_like(values), where=norms > 0)
    return normalized, np.flatnonzero(norms.ravel() == 0)

def compute_ppmi(context_counts, smoothing=1.0):
    if not np.isfinite(smoothing) or smoothing <= 0:
        raise ValueError("smoothing must be positive and finite")
    smoothed = context_counts + smoothing
    total = smoothed.sum()
    row_totals = smoothed.sum(axis=1, keepdims=True)
    column_totals = smoothed.sum(axis=0, keepdims=True)
    return np.maximum(np.log(smoothed * total / (row_totals * column_totals)), 0.0)

def fit_pca(ppmi, dimensions=150, whiten=False):
    """Normalize PPMI, fit centered full-SVD PCA, then normalize again."""
    pca_input, _ = normalize_rows(ppmi)
    if dimensions > min(pca_input.shape):
        raise ValueError("PCA dimensions exceed the available topics or context dimensions.")
    pca = PCA(n_components=dimensions, svd_solver="full", whiten=whiten)
    embeddings = pca.fit_transform(pca_input)
    normalized, zero_rows = normalize_rows(embeddings)
    if len(zero_rows) or not np.isfinite(normalized).all():
        raise ValueError("PCA produced zero or nonfinite vectors; cosine evaluation is undefined.")
    return pca, embeddings, normalized


def train_explicit(counts, cutoff=5, context_dimensions=500, dimensions=150,
                   smoothing=1.0, whiten=False):
    context, retained, _, _ = build_context_counts(counts, cutoff, context_dimensions)
    ppmi = compute_ppmi(context, smoothing)
    pca, raw, normalized = fit_pca(ppmi, dimensions, whiten)
    return retained, raw, normalized, pca

def embedding_frame(topics, values):
    return pd.concat([
        topics[["Topic_Id", *TOPIC_COLUMNS]].reset_index(drop=True),
        pd.DataFrame(values, columns=[f"embedding_{i:03d}" for i in range(values.shape[1])]),
    ], axis=1)


def export_embeddings(output_dir, topics, embeddings, normalized_embeddings, ppmi=None):
    """Export explicit vectors with topic metadata to CSV.

    Writes topic_embeddings.csv and topic_embeddings_normalized.csv, plus
    topic_embeddings_<dimensions>_ppmi.csv when PPMI vectors are supplied.
    Arrays must follow the row order of the retained topics table.
    """
    vectors = {
        "topic_embeddings.csv": embeddings,
        "topic_embeddings_normalized.csv": normalized_embeddings,
    }
    if ppmi is not None:
        vectors[f"topic_embeddings_{ppmi.shape[1]}_ppmi.csv"] = ppmi
    for values in vectors.values():
        if values.ndim != 2 or values.shape[0] != len(topics):
            raise ValueError("Embedding arrays must have one row per retained topic.")
    if embeddings.shape != normalized_embeddings.shape:
        raise ValueError("Raw and normalized embeddings must have the same shape.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, values in vectors.items():
        embedding_frame(topics, values).to_csv(output_dir / filename, index=False)