#!/usr/bin/env python3
"""Stream comment embeddings, then roll up sums/counts to every ancestor topic.

Requires Python >=3.10 and NumPy. Run --help for command-line options.
Each worker retains one monthly comment lookup and a dense float64 sum matrix.

python3 scripts/comment_embeddings_mean.py \
  --articles 'data/raw/articles_file_*.tsv' \
  --embeddings 'data/raw/embeddings_*.tsv' \
  --workers 8 \
  --output data/processed/comment_embedding_means.csv
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import dataclass
import glob
import multiprocessing as mp
import os
from pathlib import Path
import re
import sys
import tempfile
import warnings

# Workers use elementwise NumPy operations, not BLAS  (Basic Linear Algebra Subprograms). Avoid nested thread pools.
for _name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_name] = '1'
import numpy as np

NUMBER = rb'[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?'
VECTOR = re.compile(NUMBER + rb'(?:\t' + NUMBER + rb')*')
ARTICLES = {}
N_TOPICS = DIMENSIONS = 0
VALIDATE = True
COMMENT_PATH = None
COMMENTS = {}


@dataclass(frozen=True)
class Task:
    """Represents a byte-range task for processing a single embedding file."""
    embedding: str
    comments: str
    start: int
    end: int
    header_end: int


def expand(patterns):
    """Expand wildcard patterns. Support both shell-expanded arguments and quoted wildcard patterns."""
    paths = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern)) # find all files matching the wildcard pattern
        if not matches:
            raise ValueError(f'No files match: {pattern}')
        paths.extend(str(Path(p).resolve()) for p in matches)
    if len(set(paths)) != len(paths):
        raise ValueError('Duplicate input paths would count data twice')
    return paths


def load_articles(paths):
    articles, topic_ids = {}, {}
    labels, parents = [('', '', '')], [0]
    for path in paths:
        with open(path, 'rb') as stream:
            header = stream.readline().rstrip(b'\r\n').split(b'\t')
            if header[:4] != [b'Article_Id', b'Topic_1', b'Topic_2', b'Topic_3']:
                raise ValueError(f'Invalid article header: {path}')
            for line_no, line in enumerate(stream, 2):
                fields = line.rstrip(b'\r\n').split(b'\t')
                if len(fields) != 6 or not fields[0]:
                    raise ValueError(f'Invalid article row: {path}:{line_no}')
                topic_path = tuple(x.decode('utf-8') for x in fields[1:4])
                topic = 0
                for depth in range(1, 4):
                    if not topic_path[depth - 1]:
                        continue
                    key = topic_path[:depth] + ('',) * (3 - depth)
                    if key not in topic_ids:
                        topic_ids[key] = len(labels)
                        labels.append(key)
                        parents.append(topic)
                    topic = topic_ids[key]
                previous = articles.get(fields[0])
                if previous is not None and previous != topic:
                    raise ValueError(f'Conflicting article ID: {fields[0]!r}')
                articles[fields[0]] = topic
    return articles, labels, parents


def initialize(articles, n_topics, dimensions, validate):
    # Global variables are inherited by child processes, each worker retains a copy
    global ARTICLES, N_TOPICS, DIMENSIONS, VALIDATE, COMMENT_PATH, COMMENTS
    ARTICLES, N_TOPICS, DIMENSIONS, VALIDATE = articles, n_topics, dimensions, validate
    COMMENT_PATH, COMMENTS = None, {}
    warnings.filterwarnings('error', category=DeprecationWarning)


def load_comments(path):
    global COMMENT_PATH, COMMENTS
    if COMMENT_PATH == path:
        return
    # Release the previous month before allocating the next lookup.
    COMMENTS = {}
    COMMENT_PATH = None
    with open(path, 'rb') as stream:
        header = stream.readline().rstrip(b'\r\n').split(b'\t')
        if len(header) < 8 or header[0] != b'Comment_Id' or header[7] != b'Article_Id':
            raise ValueError(f'Invalid comment header: {path}')
        for line_no, line in enumerate(stream, 2):
            fields = line.rstrip(b'\r\n').split(b'\t')
            if len(fields) != 10 or not fields[0] or not fields[7]:
                raise ValueError(f'Invalid comment row: {path}:{line_no}')
            topic = ARTICLES.get(fields[7], -1)
            previous = COMMENTS.get(fields[0])
            if previous is not None and previous != topic:
                raise ValueError(f'Conflicting comment ID: {fields[0]!r}')
            COMMENTS[fields[0]] = topic
    COMMENT_PATH = path


def make_tasks(paths, comments_dir, block_bytes):
    dimensions = None
    tasks = []
    for path in paths:
        name = Path(path).name
        if not re.fullmatch(r'embeddings_[0-9]{2}_[0-9]{2}\.tsv', name):
            raise ValueError(f'Expected embeddings_MM_YY.tsv: {path}')
        comments = (Path(comments_dir) if comments_dir else Path(path).parent) / ('comments_file_' + name[11:])
        if not comments.is_file():
            raise ValueError(f'Missing comment file: {comments}')
        with open(path, 'rb') as stream:
            header = stream.readline().rstrip(b'\r\n').split(b'\t') # column names, not data
            header_end = stream.tell() # byte offset of the first data row, after the header line
        d = len(header) - 1
        if d < 1 or header != [b'Post_Id'] + [f'dim_{i}'.encode() for i in range(1, d + 1)]:
            raise ValueError(f'Invalid embedding header: {path}')
        if dimensions is not None and d != dimensions:
            raise ValueError(f'Inconsistent dimensions: {path}')
        dimensions = d
        size = os.path.getsize(path) # total file size in bytes
        stride = block_bytes or max(1, size - header_end)
        for start in range(header_end, size, stride):
            # One task per file if block_bytes=0; otherwise, split into byte ranges of block_bytes.
            tasks.append(Task(path, str(comments), start, min(size, start + stride), header_end))
    return dimensions, tasks


def process_task(task):
    load_comments(task.comments)
    sums = np.zeros((N_TOPICS, DIMENSIONS), dtype=np.float64)
    counts = np.zeros(N_TOPICS, dtype=np.int64)
    stats = dict(rows=0, used=0, missing_comment=0, missing_article=0, no_topic=0)
    with open(task.embedding, 'rb', buffering=1024 * 1024) as stream:
        if task.start == task.header_end:
            stream.seek(task.start)
        else:
            stream.seek(task.start - 1)
            stream.readline()
        while stream.tell() < task.end:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            stats['rows'] += 1
            comment, sep, payload = line.partition(b'\t')
            if not sep:
                raise ValueError(f'Invalid embedding row: {task.embedding} byte {offset}')
            topic = COMMENTS.get(comment, -2)
            if topic <= 0:
                stats[{-2: 'missing_comment', -1: 'missing_article', 0: 'no_topic'}[topic]] += 1
                continue
            payload = payload.rstrip(b'\r\n')
            if VALIDATE and VECTOR.fullmatch(payload) is None:
                raise ValueError(f'Invalid numeric vector: {task.embedding} byte {offset}')
            # Parsing and all coordinate additions execute in compiled NumPy, no Python loop over the 896 dimensions
            vector = np.fromstring(payload, dtype=np.float64, sep='\t')
            if vector.size != DIMENSIONS or payload.count(b'\t') != DIMENSIONS - 1:
                raise ValueError(f'Wrong dimension count: {task.embedding} byte {offset}')
            if VALIDATE and not np.isfinite(vector).all():
                raise ValueError(f'Non-finite vector: {task.embedding} byte {offset}')
            sums[topic] += vector
            counts[topic] += 1
            stats['used'] += 1
    # Only ship active topics between processes; don't serialize a full dense matrix of zeros for inactive topics
    active = np.flatnonzero(counts)
    return active, sums[active], counts[active], stats


def task_results(tasks, workers, initargs, start_method):
    if workers == 1:
        initialize(*initargs)
        for task in tasks:
            yield process_task(task)
        return
    # Results are yielded in the same order as tasks, even if some tasks take longer than others
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context(start_method),
                             initializer=initialize, initargs=initargs) as pool:
        pending = {}
        for i in range(min(workers, len(tasks))):
            pending[i] = pool.submit(process_task, tasks[i])
        for i in range(len(tasks)):
            result = pending.pop(i).result()
            yield result
            following = i + workers
            if following < len(tasks):
                pending[following] = pool.submit(process_task, tasks[following])


def write_output(path, labels, sums, counts):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='',
                                         dir=path.parent, prefix=path.name + '.', suffix='.tmp', delete=False) as stream:
            temp_path = Path(stream.name)
            writer = csv.writer(stream, delimiter=',', lineterminator='\n')
            writer.writerow(['Topic_1', 'Topic_2', 'Topic_3', 'N_embeddings'] + [f'dim_{i}' for i in range(1, sums.shape[1] + 1)])
            for topic, label in enumerate(labels):
                if counts[topic]:
                    writer.writerow([*label, int(counts[topic]), *(format(x, '.17g') for x in sums[topic] / counts[topic])])
        os.replace(temp_path, path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def main(argv=None):
    # Parse command-line arguments and validate them
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--articles', nargs='+', required=True, help='All article files (wildcards accepted)')
    parser.add_argument('--embeddings', nargs='+', required=True, help='Embedding files (wildcards accepted)')
    parser.add_argument('--output', required=True, help='Output CSV; replaced atomically on success')
    parser.add_argument('--comments-dir', help='Defaults to each embedding file directory')
    parser.add_argument('--workers', type=int, default=1, help='Worker processes (default: 1)')
    parser.add_argument('--block-mib', type=int, default=0, help='Split files into byte ranges of this size; 0 = one task per file (default)')
    parser.add_argument('--no-validate', action='store_true', help='Skip numeric syntax and finite checks for trusted data')
    parser.add_argument('--start-method', choices=mp.get_all_start_methods(), default='spawn', help='Process start method (default: spawn; fork available on Linux/macOS)')
    args = parser.parse_args(argv)
    if args.workers < 1 or args.block_mib < 0:
        parser.error('workers must be positive; block-mib must be nonnegative')

    # Expand wildcards, validate input files, and prepare tasks
    article_paths, embedding_paths = expand(args.articles), expand(args.embeddings)
    inputs = set(article_paths + embedding_paths)
    dimensions, tasks = make_tasks(embedding_paths, args.comments_dir, args.block_mib * 1024**2)
    inputs.update(task.comments for task in tasks)
    if str(Path(args.output).resolve()) in {str(Path(p).resolve()) for p in inputs}:
        parser.error('output must not overwrite an input file')
    if not tasks:
        raise ValueError('No embedding data rows')

    # Load articles and topics into memory for fast lookup by comment ID. Each worker will retain a copy.
    articles, labels, parents = load_articles(article_paths)
    workers = min(args.workers, len(tasks))
    print(f'workers={workers} tasks={len(tasks)} topics={len(labels)-1} dimensions={dimensions}', file=sys.stderr, flush=True)

    # Process tasks in parallel and accumulate sums and counts for each topic
    sums = np.zeros((len(labels), dimensions), dtype=np.float64)
    counts = np.zeros(len(labels), dtype=np.int64)
    initargs = articles, len(labels), dimensions, not args.no_validate
    for task, result in zip(tasks, task_results(tasks, workers, initargs, args.start_method)):
        active, partial_sums, partial_counts, stats = result
        sums[active] += partial_sums
        counts[active] += partial_counts
        print(f'{task.embedding} bytes={task.start}:{task.end} ' + ' '.join(f'{k}={v}' for k,v in stats.items()), file=sys.stderr, flush=True)
    if not counts.sum():
        raise ValueError('No embeddings matched a topic')
    for topic in range(len(labels) - 1, 0, -1):
        parent = parents[topic]
        if parent and counts[topic]:
            sums[parent] += sums[topic]
            counts[parent] += counts[topic]
    write_output(args.output, labels, sums, counts)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, DeprecationWarning) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
