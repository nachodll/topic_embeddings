"""Pinned word2vecf SGNS with reusable input preparation and training."""
from pathlib import Path
from tempfile import TemporaryDirectory
from collections import Counter
import subprocess
import numpy as np
import pandas as pd

TOPIC_COLUMNS = ["Topic_1", "Topic_2", "Topic_3"]

TOOL_COMMIT = "2d503b095546660e0bb8b0594c7b442c7c533e49"
TOOL_DIR = Path(".tools/word2vecf")

def build_trainer(output_dir, tool_dir=TOOL_DIR, commit=TOOL_COMMIT):
    """Obtain the pinned word2vecf source and compile its C training executable."""

    tool_dir, output_dir = Path(tool_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not tool_dir.exists():
        subprocess.run(["git", "clone", "https://github.com/BIU-NLP/word2vecf.git", str(tool_dir)], check=True)
        subprocess.run(["git", "-C", str(tool_dir), "checkout", commit], check=True)
    actual = subprocess.check_output(["git", "-C", str(tool_dir), "rev-parse", "HEAD"], text=True).strip()
    if actual != commit:
        raise ValueError("word2vecf version differs from the pinned implementation")
    trainer = (output_dir / "word2vecf").resolve()
    with (output_dir / "build.log").open("w") as log:
        subprocess.run(["cc", "-O3", "-pthread", *[str(tool_dir / name) for name in
                        ("word2vecf.c", "vocab.c", "io.c")], "-lm", "-o", str(trainer)],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    return trainer


def prepare_training_pairs(comment_chunks, topics, article_to_topic, directory,
                           shuffle_seed=42, chunk_size=250_000):
    """Convert comment chunks into shuffled topic-author pairs and vocabularies. 
    Files written:
        ``pairs.txt``, ``topics.vocab`` and ``users.vocab`` in ``directory``.
        The temporary ``pairs.bin`` is deleted after successful preparation.
    """
    topic_to_index = {str(t): i for i, t in enumerate(topics["Topic_Id"])}
    rng = np.random.default_rng(shuffle_seed)
    user_codes = {} # maps author IDs to int codes for the trainer (more efficient)
    user_counts = Counter() # maps user codes to comment counts for the vocab file
    topic_counts = np.zeros(len(topics), dtype=np.int64) # counts comments per topic for the vocab file
    total_pairs = blank_authors = 0
    # Create a temporary binary file (pairs.bin) of (topic_id, user_code) pairs, one per valid comment (read by chunks)
    scratch = Path(directory)
    scratch.mkdir(parents=True, exist_ok=True)
    binary_pairs = scratch / "pairs.bin"
    with binary_pairs.open("wb") as output:
        for chunk in comment_chunks:
            # Raw shards also contain articles outside the selected topics
            topic_ids = chunk["Article_Id"].map(article_to_topic).map(topic_to_index)
            chunk = chunk.loc[topic_ids.notna()]
            topic_ids = topic_ids.dropna()

            # Check for blank authors and count them, then filter them out
            valid = chunk["Author_Id"].str.strip().ne("")
            blank_authors += int((~valid).sum())
            authors = chunk.loc[valid, "Author_Id"]
            topic_ids = topic_ids.loc[valid].to_numpy(dtype=np.int32)

            # Map authors to user codes for efficient training and shuffling
            for author in authors.unique():
                if author not in user_codes:
                    user_codes[author] = len(user_codes)
            users = authors.map(user_codes).to_numpy(dtype=np.int32)

            # Write the (topic, author) pairs to the binary file and update counts
            np.column_stack((topic_ids, users)).astype(np.int32).tofile(output)

            # Update counts for topics and users
            topic_counts += np.bincount(topic_ids, minlength=len(topics))
            ids, counts = np.unique(users, return_counts=True)
            user_counts.update(dict(zip(ids.tolist(), counts.tolist())))
            total_pairs += len(users)

    n_users = len(user_codes)
    if total_pairs == 0 or n_users < 2:
        raise ValueError("SGNS needs valid pairs from at least two authors")

    # Access pais.bin as a memory-mapped array and shuffle the pairs
    pairs = np.memmap(binary_pairs, dtype=np.int32, mode="r+", shape=(total_pairs, 2))
    rng.shuffle(np.asarray(pairs).view(np.uint64).reshape(-1))
    pairs.flush()

    # The trainer expects a txt file so we write pairs.bin to pairs.txt
    with (scratch / "pairs.txt").open("w") as output:
        output.write("\n") # first line is skipped by the trainer
        for start in range(0, total_pairs, chunk_size):
            # Pairs writtern as: t<topic_id> u<user_code> for convenience
            output.writelines(f"t{topic} u{user}\n" for topic, user in np.asarray(pairs[start:start + chunk_size]))
    del pairs

    # Write vocab files for topics and users
    with (scratch / "topics.vocab").open("w") as output:
        output.writelines(f"t{i} {count}\n" for i, count in enumerate(topic_counts))
    with (scratch / "users.vocab").open("w") as output:
        output.writelines(f"u{i} {user_counts[i]}\n" for i in range(n_users))

    binary_pairs.unlink()

    #Return statistics and the user code mapping for later use
    return dict(training_pairs=total_pairs, users=n_users, excluded_blank_authors=blank_authors,
                topic_comment_counts=topic_counts.tolist()), user_codes


def train_prepared(trainer, prepared_dir, output_dir, dimensions=150, negative=35,
                   alpha=0.18, sample=0.0043, epochs=1, threads=4, dump_context=True):
    """Run SGNS on prepared files and retain the learned vectors and log.
    Files written:
        ``topic_vectors.txt`` and ``training.log`` in ``output_dir``, plus
        ``user_vectors.txt`` when ``dump_context`` is true. Existing files
        with those names are overwritten; temporary working files are removed.
    """
    output_dir, prepared_dir = Path(output_dir).resolve(), Path(prepared_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="trainer-", dir=output_dir) as work:
        work = Path(work)
        for name in ("pairs.txt", "topics.vocab", "users.vocab"):
            (work / name).symlink_to(prepared_dir / name)
        command = [str(Path(trainer).resolve()), "-train", "pairs.txt", "-wvocab", "topics.vocab",
                   "-cvocab", "users.vocab", "-output", "topic_vectors.txt",
                   "-size", str(dimensions), "-negative", str(negative), "-alpha", str(alpha),
                   "-sample", str(sample), "-iters", str(epochs), "-threads", str(threads),
                   "-binary", "0", "-debug", "0"]
        if dump_context:
            command += ["-dumpcv", "user_vectors.txt"]
        with (output_dir / "training.log").open("w") as log:
            subprocess.run(command, cwd=work, stdout=log, stderr=subprocess.STDOUT, check=True)
        for name in (["topic_vectors.txt", "user_vectors.txt"] if dump_context else ["topic_vectors.txt"]):
            (work / name).replace(output_dir / name)


def train_embeddings(comment_chunks, topics, article_to_topic, trainer, output_dir,
                     dimensions=150, negative=35, alpha=0.18, sample=0.0043, epochs=1,
                     threads=4, shuffle_seed=42, chunk_size=250_000):
    """Prepare comment pairs, train topic and author embeddings, and clean up."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="sgns-", dir=output_dir) as scratch:
        stats, users = prepare_training_pairs(comment_chunks, topics, article_to_topic, scratch,
                                              shuffle_seed, chunk_size)
        train_prepared(trainer, scratch, output_dir, dimensions, negative, alpha, sample,
                       epochs, threads)

    # Save the user mapping and topic mapping for later use in embedding export
    pd.DataFrame({
        "Author_Id": list(users),
        "User_Token": [f"u{code}" for code in users.values()],
    }).to_csv(output_dir / "user_mapping.csv", index=False)
    topic_mapping = topics[["Topic_Id", *TOPIC_COLUMNS]].copy()
    topic_mapping.insert(0, "Topic_Token", [f"t{i}" for i in range(len(topics))])
    topic_mapping.to_csv(output_dir / "topic_mapping.csv", index=False)
    return stats


def export_embeddings(output_dir):
    """Load saved SGNS vectors and export raw and normalized embedding CSVs.
    Writes ``topic_embeddings.csv``, ``topic_embeddings_normalized.csv``,
        ``user_embeddings.csv`` and ``user_embeddings_normalized.csv`` to
        ``output_dir``. Author tables contain original IDs, not internal tokens.
    """
    output_dir = Path(output_dir)
    topics = pd.read_csv(output_dir / "topic_mapping.csv", dtype=str, keep_default_na=False)
    topic_vectors = read_vectors(output_dir / "topic_vectors.txt",
                                 topics["Topic_Token"])
    normalized_topics, zero_topics = normalize_rows(topic_vectors)
    if len(zero_topics):
        raise ValueError("SGNS produced zero topic vectors")

    users = pd.read_csv(output_dir / "user_mapping.csv", dtype=str, keep_default_na=False)
    user_vectors = read_vectors(output_dir / "user_vectors.txt", users["User_Token"])
    normalized_users, _ = normalize_rows(user_vectors)
    columns = [f"embedding_{i:03d}" for i in range(user_vectors.shape[1])]
    user_frame = pd.DataFrame(user_vectors, columns=columns)
    user_frame.insert(0, "Author_Id", users["Author_Id"].to_numpy())
    normalized_user_frame = pd.DataFrame(normalized_users, columns=columns)
    normalized_user_frame.insert(0, "Author_Id", users["Author_Id"].to_numpy())

    embedding_frame(topics, topic_vectors).to_csv(output_dir / "topic_embeddings.csv", index=False)
    embedding_frame(topics, normalized_topics).to_csv(
        output_dir / "topic_embeddings_normalized.csv", index=False)
    user_frame.to_csv(output_dir / "user_embeddings.csv", index=False)
    normalized_user_frame.to_csv(output_dir / "user_embeddings_normalized.csv", index=False)


def embedding_frame(topics, values):
    return pd.concat([
        topics[["Topic_Id", *TOPIC_COLUMNS]].reset_index(drop=True),
        pd.DataFrame(values, columns=[f"embedding_{i:03d}" for i in range(values.shape[1])]),
    ], axis=1)


def read_vectors(path, tokens):
    """Load word2vecf text vectors and select rows in the requested token order."""
    vectors = {}
    with Path(path).open() as source:
        rows, dimensions = map(int, source.readline().split())
        for line in source:
            token, *values = line.split()
            if token in vectors or len(values) != dimensions:
                raise ValueError("Invalid word2vecf vector output")
            vectors[token] = np.asarray(values, dtype=np.float32)
    if len(vectors) != rows:
        raise ValueError("Vector row count does not match header")
    values = np.stack([vectors[token] for token in tokens])
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite vectors from trainer")
    return values

def normalize_rows(values):
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    normalized = np.divide(values, norms, out=np.zeros_like(values), where=norms > 0)
    return normalized, np.flatnonzero(norms.ravel() == 0)