"""Analogy evaluation and summaries; validation errors count as misses."""
import numpy as np
import pandas as pd

TOPIC_COLUMNS = ["Topic_1", "Topic_2", "Topic_3"]


class AnalogyEvaluator:
    def __init__(self, topics, analogies):
        """Initialize with topic embeddings and analogy pairs."""
        
        self.topics = topics.copy().reset_index(drop=True)
        self.topics["Topic_Id"] = self.topics["Topic_Id"].astype(str)
        columns = sorted(c for c in self.topics if c.startswith("embedding_"))
        self.vectors = self.topics[columns].to_numpy(dtype=np.float64)
        if not columns or not np.isfinite(self.vectors).all():
            raise ValueError("Embeddings must contain finite numeric coordinates.")
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        if np.any(norms == 0) or self.topics["Topic_Id"].duplicated().any():
            raise ValueError("Topic IDs must be unique and vectors must have nonzero length.")
        self.paths = self.topics[TOPIC_COLUMNS].apply(
            lambda row: "/".join(str(v) for v in row if v), axis=1)
        self.id_to_index = {topic_id: i for i, topic_id in enumerate(self.topics["Topic_Id"])}
        self.analogies = analogies.copy()

    def evaluate_analogy(self, analogy, top_k=5, verbose=False):
        """
        Accept an analogy row or ID and return (metrics, neighbors).
        Set verbose=True to print the analogy details and nearest topics.
        A:B::C:D analogy evaluation using cosine similarity of normalized embeddings.
        B-A+C should yield a vector similar to D. Ties follow embedding-file row order.
        An expected answer equal to A, B, or C is excluded and counts as a miss.
        """
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be a positive integer.")
        if isinstance(analogy, str):
            analogy_id = analogy
            selected = self.analogies.loc[self.analogies["analogy_id"] == analogy_id]
            if len(selected) != 1:
                raise ValueError(f"Expected exactly one analogy with ID {analogy_id}; found {len(selected)}.")
            analogy = selected.iloc[0]

        # Validate that all four topic IDs exist and match the expected paths
        indices = {}
        for role in "abcd":
            topic_id = analogy[f"topic_id_{role}"]
            if topic_id not in self.id_to_index:
                raise ValueError(f"Topic {role.upper()} (ID {topic_id}) is missing from the embeddings.")
            index = self.id_to_index[topic_id]
            if self.paths.iloc[index] != analogy[f"topic_path_{role}"]:
                raise ValueError(f"Path mismatch for {role.upper()}: {self.paths.iloc[index]} != "
                                 f"{analogy[f'topic_path_{role}']}")
            indices[role] = index

        # Compute the analogy query vector and its cosine similarity to all topics
        query = self.vectors[indices["b"]] - self.vectors[indices["a"]] + self.vectors[indices["c"]]
        query_norm = np.linalg.norm(query)
        if query_norm <= np.finfo(float).eps:
            raise ValueError("The analogy query has zero length; cosine similarity is undefined.")
        scores = self.vectors @ (query / query_norm)
        excluded = {indices[role] for role in "abc"}
        candidates = np.array([i for i in range(len(self.topics)) if i not in excluded], dtype=int)
        if not len(candidates):
            raise ValueError("No eligible topics remain after excluding A, B, and C.")
        ranking = candidates[np.argsort(-scores[candidates], kind="stable")]
        top_indices = ranking[:top_k]
        expected_position = np.flatnonzero(ranking == indices["d"])
        expected_rank = int(expected_position[0]) + 1 if len(expected_position) else None
        metrics = {
            "analogy_id": analogy["analogy_id"],
            "relation": analogy["relation"],
            "label": analogy["label"],
            "expected_topic_id": analogy["topic_id_d"],
            "expected_path": self.paths.iloc[indices["d"]],
            "expected_rank": expected_rank,
            "expected_cosine_similarity": float(scores[indices["d"]]),
            "eligible_topics": len(candidates),
            "expected_excluded": indices["d"] in excluded,
            **{f"top{k}": expected_rank is not None and expected_rank <= k for k in (5, 3, 1)},
            "error": "",
        }
        neighbors = pd.DataFrame({
            "Rank": np.arange(1, len(top_indices) + 1),
            "Topic_Id": self.topics.iloc[top_indices]["Topic_Id"].to_numpy(),
            "Topic_path": self.paths.iloc[top_indices].to_numpy(),
            "Cosine_similarity": scores[top_indices],
            "Expected_answer": top_indices == indices["d"],
        })

        if verbose:
            print(f"{analogy['analogy_id']} | Relation: {analogy['relation']} | Label: {analogy['label']}")
            print(f"Notes: {analogy['notes'] or '(none)'}")
            for role in "abcd":
                purpose = "expected answer" if role == "d" else "query input"
                print(f"{role.upper()} ({purpose}): ID={analogy[f'topic_id_{role}']} | "
                      f"{analogy[f'topic_path_{role}']}")

            print("\n")
            print(neighbors.to_string(index=False, float_format=lambda value: f"{value:.6f}"))

        return metrics, neighbors

    def evaluate_all_analogies(self, analogy_rows):
        """Return one result per row, including validation errors instead of skipping them."""
        records = []
        for _, analogy in analogy_rows.iterrows():
            try:
                metrics, _ = self.evaluate_analogy(analogy)
            except ValueError as exc:
                metrics = {
                    "analogy_id": analogy["analogy_id"],
                    "relation": analogy["relation"],
                    "label": analogy["label"],
                    "expected_topic_id": analogy["topic_id_d"],
                    "expected_path": analogy["topic_path_d"],
                    "expected_rank": None,
                    **{f"top{k}": False for k in (5, 3, 1)},
                    "error": str(exc),
                }
            records.append(metrics)
        if not records:
            raise ValueError("No analogies to evaluate.")
        results = pd.DataFrame(records)
        results["expected_rank"] = results["expected_rank"].astype("Int64")
        return results



def summarize_analogies(results):
    """Count hits using all supplied rows as the denominator, including errors."""
    total = len(results)
    if not total:
        raise ValueError("No results to summarize.")
    return pd.DataFrame([
        {"Cutoff": f"Top {k}", "Matches": int(results[f"top{k}"].sum()),
         "Total analogies": total, "Hit rate (%)": 100 * results[f"top{k}"].sum() / total}
        for k in (5, 3, 1)
    ])
