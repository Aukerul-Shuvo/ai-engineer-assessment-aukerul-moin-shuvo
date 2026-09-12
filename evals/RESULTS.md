# Evaluation results

The runs that set this service's retrieval parameters, and the one that decided its shape. Each
table names the command that produced it; the JSON reports land in `evals/reports/` (not
committed) and every run is reproducible from the committed corpus with a free Gemini key.

Corpus: the first 20 articles of the SQuAD v1.1 development set, 888 paragraphs, 4,913 gold
questions. Each question records the paragraph it was written from, so "recall at k" below means
"the gold paragraph is within the top k", measured exactly rather than judged.

Samples: every question costs one query embedding and the free tier allows 1,000 a day, so the
tables below are a 300-question random sample, seed 42, not the full 4,913. The corpus size is
printed beside each table because retrieval numbers are not comparable across corpora.

## 1. The pipeline as it ships

`python -m evals.retrieval --limit 300 --rerank --pool 50`
(888 paragraphs, 300 questions, `gemini-embedding-001` at 768 dimensions,
`ms-marco-MiniLM-L-12-v2` at 512 tokens over the nearest 50; 648 s)

| stage | R@1 | R@3 | R@5 | R@10 | R@20 | R@50 | MRR |
|---|---|---|---|---|---|---|---|
| vector search | 0.803 | 0.940 | 0.963 | 0.987 | 0.993 | 1.000 | 0.875 |
| after reranking | **0.877** | 0.963 | 0.977 | 0.990 | 1.000 | 1.000 | **0.922** |

Reading: the vector search puts the gold paragraph in its top 50 for **every** question in the
sample, so the reranker always has the right passage to promote, and it promotes it to first
place 88% of the time. The relevance grader sees the top 10, which holds the gold paragraph 99%
of the time, and the synthesiser gets six passages.

## 2. Why there is no keyword index

The plan was hybrid retrieval: BM25 and dense vectors fused with reciprocal rank fusion, then
reranked. It was built that way, measured, and then removed. All four rows below come from the
same 300 questions and the same 888-paragraph corpus, the first three from the hybrid
implementation before it was deleted:

| ranking | R@1 | R@5 | R@10 | R@50 | MRR |
|---|---|---|---|---|---|
| BM25 alone | 0.763 | 0.917 | 0.953 | 0.990 | 0.833 |
| vector search alone | 0.803 | 0.963 | 0.987 | 1.000 | 0.875 |
| the two fused (RRF, k=60) | 0.817 | 0.963 | 0.983 | 1.000 | 0.887 |
| fused, then reranked | 0.870 | 0.980 | 0.997 | 1.000 | 0.920 |
| **vector search, then reranked (shipped)** | **0.877** | 0.977 | 0.990 | 1.000 | **0.922** |

Reading: the vector search beats BM25 on every column. Fusion adds 1.4 points of recall@1 over
the vector search alone, and that margin disappears once the cross-encoder runs: reranking the
vector search's top 50 scores *higher* than reranking the fused top 50 (0.877 against 0.870,
MRR 0.922 against 0.920). The reason is in the R@50 column, which is 1.000 for the vector search
on its own: BM25 could not add a passage the pool was missing, because the pool was never
missing one. A whole retriever, its index, its dependencies and the fusion step earned nothing
measurable, so they were deleted.

For the record, BM25 over the full 48-article corpus reached recall@1 0.776 and MRR 0.844
(all 10,570 questions, `retrieval_20260908T171854Z.json`), and 0.751 / 0.830 over this
20-article corpus (4,913 questions, `retrieval_20260909T141245Z.json`). Those runs cannot be
reproduced from the current code, which has no lexical index.

## 3. What a rerank call costs

`python -m evals.rerank_timing` (real candidates from the vector search for 8 gold questions,
median wall time per call, idle 8-core desktop CPU, Ryzen 7 7700, ONNX runtime on CPU)

| model | max length | 100 candidates | 50 | 30 | 20 |
|---|---|---|---|---|---|
| ms-marco-MiniLM-L-12-v2 | 512 | 3.58 s | **1.37 s** | 0.77 s | 0.34 s |
| ms-marco-MiniLM-L-12-v2 | 256 | 1.99 s | 0.97 s | 0.55 s | 0.34 s |
| ms-marco-TinyBERT-L-2-v2 | 512 | 0.15 s | 0.06 s | 0.03 s | 0.01 s |

Reading: cost grows faster than linearly with the number of candidates, because the batch pads
to its longest passage and a bigger pool is likelier to contain a long one. Five seconds per
sub-query is too much for an interactive endpoint, so the depth was chosen against recall rather
than taken from the literature.

## 4. Recall under the cheaper settings

The depth question is settled by the recall table in section 1 rather than by a sweep: the
vector search's recall@50 is 1.000, so reranking 100 candidates instead of 50 cannot find a
paragraph that reranking 50 missed, and it costs 2.6 times the CPU. Reranking the top 50 of the
vector search also scored higher at rank one than reranking the fused top 100 did in section 2.

The two remaining reranker choices were swept on the previous pipeline and corpus (2,067
paragraphs, BM25 pool, same 300-question protocol, `retrieval_20260908T18*.json`). Both are
properties of the cross-encoder rather than of whatever produced its candidates, so the
direction carries over, but they are labelled as what they are:

| setting | R@1 | R@5 | R@10 | MRR | s per call |
|---|---|---|---|---|---|
| MiniLM-L-12, 512 tokens, 100 candidates | 0.873 | 0.963 | 0.987 | 0.916 | 4.97 |
| MiniLM-L-12, 512 tokens, 50 candidates | 0.873 | 0.957 | 0.973 | 0.912 | 1.83 |
| MiniLM-L-12, 256 tokens, 50 candidates | 0.863 | 0.957 | 0.973 | 0.906 | 0.97 |
| TinyBERT-L-2, 512 tokens, 100 candidates | 0.817 | 0.943 | 0.970 | 0.874 | 0.21 |

Reading: truncating inputs to 256 tokens costs a point of recall@1, because some paragraphs run
past 256 tokens and the reranker stops seeing the part that holds the answer. TinyBERT is 24
times faster and gives back most of the reranker's gain. A same-pipeline re-run of these two
rows is pending: each costs 300 query embeddings and the free tier allows 1,000 a day.

## 5. Settings chosen

| setting | value | reason |
|---|---|---|
| `DENSE_TOP_K` | 100 | The vector search is one matrix product over 888 rows; depth is free and sets the ceiling. |
| `RERANK_CANDIDATES` | 50 | Section 4: the same recall@1 and MRR as reranking 100, for a third of the CPU. R@50 is already 1.000. |
| `RERANKER_MAX_LENGTH` | 512 | Full paragraphs. 256 is the documented knob for latency at a measured cost of one point of recall@1. |
| `RERANKER_MODEL` | ms-marco-MiniLM-L-12-v2 | Section 4: TinyBERT is 24 times faster and gives back most of the reranker's gain. |
| `RERANK_TOP_K` | 20 | Recall@20 is 1.000; keeping more cannot help. |
| `GRADE_TOP_K` | 10 | The grader's window, which holds the gold paragraph 99% of the time. |
| `EVIDENCE_PER_SUB_QUERY` | 6 | Each sub-query is narrow, and six passages keep a multi-part answer inside a modest prompt. |

## 6. Not yet measured

- **Routing accuracy** (`python -m evals.routing`) and **end-to-end answers**
  (`python -m evals.answers --limit 50`) call the live model and spend the daily model quota;
  their harness is tested in `tests/test_evals.py`.
- The recall tables are a 300-question sample. A full 4,913-question run costs 4,913 query
  embeddings, five days of the free tier, and would narrow the confidence interval rather than
  change any decision above.
