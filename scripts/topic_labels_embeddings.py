#!/usr/bin/env python3
"""Embed full topic paths with F2LLM-v2-14B for the topic-label benchmark.

From the repository root, run this in a GPU allocation:
    python -u scripts/topic_labels_embeddings.py --device cuda:0
Remove the model from the Hugging Face cache:
    hf cache rm model/codefuse-ai/F2LLM-v2-14B

Requires numpy, pandas, torch, sentence-transformers, transformers, and
huggingface_hub. For offline compute nodes, pre-cache the model on a connected
node, share HF_HOME between jobs, and set HF_HUB_OFFLINE=1.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_ID = "codefuse-ai/F2LLM-v2-330M"
TOPIC_COLUMNS = ["Topic_1", "Topic_2", "Topic_3"]
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def run(source, output, revision_arg, requested_device, batch_size, max_seq_length, force):
    metadata_path = output.with_suffix(".json")
    if len({p.resolve() for p in (source, output, metadata_path)}) != 3:
        raise ValueError("Input CSV, output CSV and metadata must be distinct paths")
    paths = pd.read_csv(source, keep_default_na=False, dtype=str).rename(
        columns={"topic1": "Topic_1", "topic2": "Topic_2", "topic3": "Topic_3"}
    )[TOPIC_COLUMNS]
    paths = paths.loc[paths.ne("").any(axis=1)].drop_duplicates().reset_index(drop=True)
    paths["Topic_path"] = paths[TOPIC_COLUMNS].apply(lambda row: " / ".join(x for x in row if x), axis=1)
    if len(paths) < 3:
        raise ValueError("Expected at least three nonempty topic paths")
    settings = dict(model_id=MODEL_ID, requested_revision=revision_arg,
                    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    separator=" / ", prompt="", normalize_embeddings=True,
                    max_seq_length=max_seq_length)
    print(f"All topic paths: {len(paths):,}")

    # For embeddings to be reused:
    # 1. The output CSV and metadata JSON must exist.
    # 2. The metadata must match the current settings.
    # 3. The saved vectors must pass validation
    # 4. force must not be set
    reuse = False
    if not force and output.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("settings") == settings:
            embeddings = pd.read_csv(output, keep_default_na=False)
            dims = embeddings.filter(regex="^dim_")
            values = dims.to_numpy(dtype=np.float32)
            reuse = (embeddings[TOPIC_COLUMNS].equals(paths[TOPIC_COLUMNS])
                     and values.shape == (len(paths), metadata["dimensions"])
                     and np.isfinite(values).all()
                     and np.allclose(np.linalg.norm(values, axis=1), 1, atol=1e-4))

    if reuse:
        print(f"Reused {output} ({metadata['resolved_revision']})")
    else:
        import torch
        import sentence_transformers
        import transformers
        from huggingface_hub import HfApi, snapshot_download
        from huggingface_hub.constants import HF_HUB_OFFLINE
        from sentence_transformers import SentenceTransformer

        device = requested_device
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else (
                "mps" if torch.backends.mps.is_available() else "cpu")
        dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if device.startswith("cuda") else (
            torch.float16 if device == "mps" else torch.float32)
        if HF_HUB_OFFLINE:
            # explicitly download the snapshot to avoid a network call in offline mode, set HF_HUB_OFFLINE=1 before running the script
            snapshot = Path(snapshot_download(MODEL_ID, revision=revision_arg, local_files_only=True))
            revision = snapshot.name
            model_source = str(snapshot)
        else:
            revision = HfApi().model_info(MODEL_ID, revision=revision_arg).sha
            model_source = MODEL_ID
        print(f"Loading {MODEL_ID}@{revision} on {device}, dtype={dtype}")
        model = SentenceTransformer(model_source, revision=revision, device=device,
                                    model_kwargs={"torch_dtype": dtype})
        model.max_seq_length = max_seq_length
        texts = paths.Topic_path.tolist()
        lengths = model.tokenizer(texts, truncation=False, padding=False, return_length=True)["length"]
        if max(lengths) > max_seq_length:
            raise ValueError("Increase max_seq_length: at least one full topic path would be truncated")
        vectors = model.encode(texts, batch_size=batch_size, prompt="",
                               normalize_embeddings=True, convert_to_numpy=True,
                               show_progress_bar=True).astype(np.float32)
        # Normalize in float32 as well, avoiding half-precision rounding in the export.
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4):
            raise ValueError("Expected finite, unit-length embeddings")
        embeddings = pd.concat([paths, pd.DataFrame(vectors, columns=[
            f"dim_{i + 1}" for i in range(vectors.shape[1])])], axis=1)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".tmp.csv")
        embeddings.to_csv(temporary, index=False)
        temporary.replace(output)
        metadata = dict(settings=settings, resolved_revision=revision, topics=len(paths),
                        dimensions=vectors.shape[1], device=device, dtype=str(dtype),
                        versions=dict(torch=torch.__version__, transformers=transformers.__version__,
                                      sentence_transformers=sentence_transformers.__version__))
        temporary = metadata_path.with_suffix(".tmp.json")
        temporary.write_text(json.dumps(metadata, indent=2) + "\n")
        temporary.replace(metadata_path)
        print(f"Saved {len(embeddings):,} paths × {vectors.shape[1]:,} dimensions to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=DATA_DIR / "topic_summary.csv")
    parser.add_argument("--output", type=Path, default=DATA_DIR / "topic_path_embeddings.csv")
    parser.add_argument("--revision", default="main", help="Hugging Face model revision or commit")
    parser.add_argument("--device", default="auto", help="Torch device, e.g. cuda:0, mps, cpu, or auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--force", action="store_true", help="Recompute even if a matching export exists")
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_seq_length < 1:
        parser.error("--batch-size and --max-seq-length must be positive")
    run(args.input, args.output, args.revision, args.device, args.batch_size, args.max_seq_length, args.force)


if __name__ == "__main__":
    main()
