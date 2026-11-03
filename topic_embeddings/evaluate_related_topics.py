"""Bidirectional related-topic evaluation and summaries; errors count as misses."""
import numpy as np
import pandas as pd

TOPIC_COLUMNS = ["Topic_1", "Topic_2", "Topic_3"]

DIRECTIONS = {"a_to_b": ("a", "b"), "b_to_a": ("b", "a")}


class RelatedEvaluator:
    def __init__(self, topics, pairs):
        """Initialize with topic embeddings and related-topic pairs."""
        
        self.topic = topics.copy().reset_index(drop=True)
        self.topic["Topic_Id"] = self.topic["Topic_Id"].astype(str)
        columns = sorted(c for c in self.topic if c.startswith("embedding_"))
        self.vectors = self.topic[columns].to_numpy(dtype=np.float64)
        if not columns or not np.isfinite(self.vectors).all():
            raise ValueError("Embeddings must contain finite numeric coordinates.")
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        if np.any(norms == 0) or self.topic["Topic_Id"].duplicated().any():
            raise ValueError("Topic IDs must be unique and vectors must have nonzero length.")
        self.vectors = self.vectors / norms
        self.paths = self.topic[TOPIC_COLUMNS].apply(
            lambda row: "/".join(str(v) for v in row if v), axis=1)
        self.id_to_index = {topic_id: i for i, topic_id in enumerate(self.topic["Topic_Id"])}
        self.pairs = pairs.copy()

    def evaluate_pair(self, pair, top_k=5, verbose=False, direction="a_to_b"):
        """Accept a pair row or ID; return metrics and neighbors for one direction.

        Set verbose=True for manual inspection. Only the query topic is excluded.
        """
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be a positive integer.")
        if direction not in DIRECTIONS:
            raise ValueError(f"direction must be one of {tuple(DIRECTIONS)}.")
        if isinstance(pair, str):
            selected = self.pairs.loc[self.pairs["pair_id"] == pair]
            if len(selected) != 1:
                raise ValueError(f"Expected exactly one pair with ID {pair}; found {len(selected)}.")
            pair = selected.iloc[0]
        indices = {}
        for role in "ab":
            topic_id = pair[f"topic_id_{role}"]
            if topic_id not in self.id_to_index:
                raise ValueError(f"Topic {role.upper()} (ID {topic_id}) is missing from the embeddings.")
            index = self.id_to_index[topic_id]
            if self.paths.iloc[index] != pair[f"topic_path_{role}"]:
                raise ValueError(f"Path mismatch for {role.upper()}: {self.paths.iloc[index]} != "
                                 f"{pair[f'topic_path_{role}']}")
            indices[role] = index
        source_role, target_role = DIRECTIONS[direction]
        source, target = indices[source_role], indices[target_role]
        if source == target:
            raise ValueError("A related-topic pair must contain two distinct topics.")
        scores = self.vectors @ self.vectors[source]
        candidates = np.flatnonzero(np.arange(len(self.topic)) != source)
        ranking = candidates[np.argsort(-scores[candidates], kind="stable")]
        expected_rank = int(np.flatnonzero(ranking == target)[0]) + 1
        top_indices = ranking[:top_k]
        metrics = {
            "pair_id": pair["pair_id"], "relation": pair["relation"], "direction": direction,
            "query_topic_id": pair[f"topic_id_{source_role}"], "query_path": self.paths.iloc[source],
            "expected_topic_id": pair[f"topic_id_{target_role}"], "expected_path": self.paths.iloc[target],
            "expected_rank": expected_rank, "cosine_similarity": float(scores[target]),
            "eligible_topics": len(candidates),
            **{f"top{k}": expected_rank <= k for k in (5, 3, 1)}, "error": "",
        }
        neighbors = pd.DataFrame({
            "Rank": np.arange(1, len(top_indices) + 1),
            "Topic_Id": self.topic.iloc[top_indices]["Topic_Id"].to_numpy(),
            "Topic_path": self.paths.iloc[top_indices].to_numpy(),
            "Cosine_similarity": scores[top_indices], "Expected_answer": top_indices == target,
        })
        if verbose:
            print(f"{pair['pair_id']} | Relation: {pair['relation']} | Direction: {direction}")
            print(f"Comments: {pair['comments'] or '(none)'}")
            print(f"Query: {metrics['query_topic_id']} | {metrics['query_path']}")
            print(f"Expected: {metrics['expected_topic_id']} | {metrics['expected_path']}")
            print(neighbors.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
            print(f"Expected-topic rank: {expected_rank} / {len(candidates)}")
            print(f"Pair cosine similarity: {metrics['cosine_similarity']:.6f}")
            print(" | ".join(f"Top {k}: {metrics[f'top{k}']}" for k in (5, 3, 1)))
        return metrics, neighbors

    def evaluate_all_pairs(self, pair_rows):
        """Evaluate every pair in both directions, retaining validation failures."""
        records = []
        for _, pair in pair_rows.iterrows():
            for direction, (source_role, target_role) in DIRECTIONS.items():
                try:
                    metrics, _ = self.evaluate_pair(pair, direction=direction)
                except ValueError as exc:
                    metrics = {
                        "pair_id": pair["pair_id"], "relation": pair["relation"], "direction": direction,
                        "query_topic_id": pair[f"topic_id_{source_role}"],
                        "query_path": pair[f"topic_path_{source_role}"],
                        "expected_topic_id": pair[f"topic_id_{target_role}"],
                        "expected_path": pair[f"topic_path_{target_role}"],
                        "expected_rank": None,
                        **{f"top{k}": False for k in (5, 3, 1)}, "error": str(exc),
                    }
                records.append(metrics)
        if not records:
            raise ValueError("No pairs to evaluate.")
        results = pd.DataFrame(records)
        results["expected_rank"] = results["expected_rank"].astype("Int64")
        return results



def summarize_pairs(results):
    """Summarize each relation and all pairs, separately for each direction."""
    if results.empty:
        raise ValueError("No results to summarize.")
    groups = list(results.groupby("relation", sort=True)) + [("ALL", results)]
    records = []
    for relation, group in groups:
        for direction, subset in group.groupby("direction", sort=True):
            for k in (5, 3, 1):
                matches = int(subset[f"top{k}"].sum())
                records.append({
                    "Relation": relation, "Direction": direction, "Cutoff": f"Top {k}",
                    "Matches": matches, "Total pairs": len(subset),
                    "Hit rate (%)": 100 * matches / len(subset),
                })
    return pd.DataFrame(records)


def summarize_related_main(results):
    # Main metrics: equally average the two directional top-5 hit rates.
    # Validation errors remain non-hits, as in the full evaluation summary.
    relations = ["close_theme", "taxonomy_relocation"]
    directional_top5 = (
        results.groupby(["relation", "direction"])["top5"]
        .mean()
        .unstack("direction")
        .reindex(index=relations, columns=["a_to_b", "b_to_a"])
    )
    if directional_top5.isna().any().any():
        raise ValueError("Both directions must have results for each requested relation.")

    main_metrics = (
        directional_top5.mean(axis=1).mul(100)
        .rename("Average top-5 hit rate (%)")
        .rename_axis("Relation")
        .reset_index()
    )
    main_metrics["Relation"] = main_metrics["Relation"].map({
        "close_theme": "Close themes",
        "taxonomy_relocation": "Taxonomy relocations",
    })
    return main_metrics
