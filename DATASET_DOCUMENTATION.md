# derStandard Articles, Topics, and Comments

## AI-ready dataset description

This directory contains data from the Austrian news site **derStandard** covering
**2013–2022**. It combines article metadata, hierarchical topic labels, reader
comments, and user-level comment/vote aggregates. The data is stored primarily as
tab-separated values (TSV); comma-separated summary tables are derived outputs.

## Scale and time encoding

- Article records: **586,942**
- Comment records: **75,644,850**
- Comment threads: **21,957,527** (from `summary_comments.tsv`)
- Replies: **53,617,134**
- Article years: `2013` through `2022`
- Raw timestamps: Unix epoch seconds
- Comments are split into 12 monthly files per year (`comments_file_MM_YY.tsv`).

## Raw files

### `articles_file_YY.tsv`

One row per article. `YY` is the two-digit year (`13` = 2013, ..., `22` = 2022).

| Column | Meaning |
|---|---|
| `Article_Id` | Unique article identifier; treat as a string |
| `Topic_1` | Top-level topic label |
| `Topic_2` | Second-level topic label |
| `Topic_3` | Third-level topic label |
| `Topic_Labels` | Human-readable topic path, formatted as `id,label;id,label;...` |
| `Timestamp` | Article publication timestamp in Unix seconds |

Topic fields are hierarchical and may be empty. An article with only `Topic_1`
belongs to that top-level topic; an empty topic field means that level is absent,
not an unknown numeric value.

### `comments_file_MM_YY.tsv`

One row per comment. There are up to 12 monthly shards per year.

| Column | Meaning |
|---|---|
| `Comment_Id` | Unique comment identifier |
| `Author_Id` | User identifier; treat as a string |
| `Is_Reply` | Reply indicator (`0`/`1`) |
| `Parent_Id` | Direct parent comment identifier, or `NA` for a root comment |
| `Thread_Level` | Nesting depth within the comment thread |
| `Origin_Parent_Id` | Root comment/thread identifier |
| `Timestamp` | Comment timestamp in Unix seconds |
| `Article_Id` | Article associated with the comment |
| `Votes_Pos` | Positive votes received |
| `Votes_Neg` | Negative votes received |

Join comments to articles with `comments.Article_Id = articles.Article_Id`.
Comments with an article ID that does not match an article row should be treated
as unmatched records and investigated rather than silently reassigned.

### `users_file`

One row per user with aggregated activity:

`User_Id`, `N_comments`, `First_Comment`, `Last_Comment`, `R_pos`, `R_neg`,
`G_pos`, `G_neg`, `First_vote`, `Last_vote`.

The timestamp fields are Unix seconds. The `R_` and `G_` vote columns are
precomputed aggregates; consult the source pipeline before assigning a more
specific interpretation to those prefixes.

## Derived topic summaries

### `topic_summary_2013_2022.csv`

Aggregates article and comment activity by topic path across all years. Columns:

`topic1`, `topic2`, `topic3`, `num_articles`, `num_comments`,
`num_articles_2013`, ..., `num_articles_2022`.

The aggregation logic produces rows at three granularities:

1. `topic1` only
2. `topic1` + `topic2`
3. `topic1` + `topic2` + `topic3`

Unused lower-level topic columns are empty strings. `num_articles` counts unique
articles, and `num_comments` sums comments linked to those articles. The same
article can therefore contribute to multiple rows at different hierarchy levels;
do **not** sum all rows together as if they were mutually exclusive categories.

### `topic_summary_18.csv`

The same style of topic summary for **2018** only.

### `topic1_counts_posts.tsv`, `topic2_counts_posts.tsv`,
`topic3_counts_posts.tsv`

Topic-level count tables. Each row contains a count followed by a topic label.
These are post/comment-count summaries, not raw article or comment records; verify
the exact measure from the producing notebook before comparing them with the
topic summary CSVs.

## Recommended interpretation and cautions

- Topic labels are categorical strings, mostly German and often lowercase
  identifiers (for example `inland`, `parteien`, `spoe`).
- Use `Article_Id`, `Comment_Id`, and `User_Id` as strings to preserve identity
  and leading zeroes.
- Convert Unix timestamps explicitly to a timezone before time-based analysis.
- Empty topic values are structural missingness at that hierarchy level.
- Article-topic counts and comment-topic counts are article-level aggregates:
  comments are assigned to a topic through their article, not through a
  comment-level topic classifier.
- The summary pipeline counts comments by `Article_Id` in chunks, then maps those
  counts onto articles. Articles with no matched comments receive zero comments.
- Avoid loading all comments into memory; process monthly files in chunks and
  retain only the columns needed for the analysis.
- The dataset supports analyses of publication volume, topic prevalence, comment
  engagement, thread structure, replies, votes, and user activity. It does not
  by itself provide article text or comment text in the files documented here.

## Minimal AI context

> Analyze a 2013–2022 derStandard news dataset. Articles are in
> `data/articles_file_YY.tsv` and comments are in monthly
> `data/comments_file_MM_YY.tsv` files. Join them on the string column
> `Article_Id`. Articles have a three-level hierarchical topic path
> (`Topic_1`–`Topic_3`), optional human-readable `Topic_Labels`, and a Unix
> publication timestamp. Comments have author, reply/thread structure, Unix
> timestamp, article ID, and positive/negative vote counts. There are 586,942
> articles and 75,644,850 comments. Derived topic summaries count unique
> articles and sum comments by topic and year; hierarchy levels overlap, so
> summary rows must not be summed across granularities. Treat empty topic fields
> as absent hierarchy levels, preserve IDs as strings, and process comment shards
> incrementally.
