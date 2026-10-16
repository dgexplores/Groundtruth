# Groundtruth

**Don't guess whether your search works. Measure it.**

Groundtruth is a retrieval-evaluation harness: it scores how well a search system finds the right documents, gates regressions in CI, and shows the numbers on an interactive dashboard. Built for developers shipping RAG and search — the people who need proof, not vibes, that retrieval works and stays working.

**Live dashboard:** https://dgexplores.github.io/Groundtruth/dashboard/

## The headline finding

Same 50 documents, two ways of asking. Lexical queries reuse the answer's own words; paraphrase queries ask the same thing with no vocabulary overlap:

| Lane | recall@5 | recall@10 |
| :--- | :--- | :--- |
| TF-IDF, lexical | 1.0000 | 1.0000 |
| **TF-IDF, paraphrases** | **0.7000** | **0.7000** |
| Dense (all-MiniLM-L6-v2), lexical | 1.0000 | 1.0000 |
| **Dense, paraphrases** | **0.9800** | **1.0000** |

TF-IDF loses 0.30 the moment users reword their questions; embeddings hold. That gap is the reason this project exists — and the reason the benchmark tests both tiers instead of reporting a flattering 1.00.

## Use it

```bash
pip install -r requirements.txt          # torch separately (CPU build in CI)

# 1. Benchmark: 2 retrievers x 2 query tiers -> evals/reports/latest.json
python3 -m evals.runner

# 2. Gate: fail when a lane regresses more than 0.05 below baseline
python3 -m evals.regression_gate evals/baseline_report.json evals/reports/latest.json --metric recall_at_10
python3 -m evals.regression_gate evals/baseline_report.json evals/reports/latest.json --metric tfidf_semantic_recall_at_10

# 3. Score answers: stub lookup by default, live model with a key
python3 -m evals.runner --mode qa
GEMINI_API_KEY=... python3 -m evals.runner --mode qa --predictor gemini --limit 20

# 4. Look: serve the repo root, open /dashboard/ (or use the hosted link above)
python3 -m http.server
```

Bring your own data by replacing `evals/corpus.json` (documents + queries with `relevant_ids` and `tier`). Cap spend on paid runs with `--max-calls` / `--max-cost-usd`; interrupted runs resume from per-answer checkpoints instead of re-spending calls. Every report carries a `run` block (`run_id`, UTC window, `git_sha`, full config) — a number without provenance is not a result. Keys come from the environment only, never touch artifacts (tested).

## What's done

- **Head-to-head benchmark** — TF-IDF vs dense embeddings, lexical vs paraphrase tiers, 50 docs / 100 queries, deterministic tie-breaks, Recall@K scoring
- **Regression gates** — both lanes gated in CI (exit 0 pass / 1 regression / 2 malformed), green on every push
- **Real-model QA** — OpenAI + Gemini predictors, SQuAD-style normalization, usage + cost tracking; live-measured `gemini-3.5-flash-lite` at 0.14 exact-match (see honest notes)
- **Production hardening** — resumable checkpoints, spend guards (fail closed on unpriced models), secrets-hygiene test, run manifests, pinned `requirements.txt`, `ruff` + `mypy` clean
- **Dashboard** — headline, matrix with per-query misses, draggable TF-IDF-vs-dense wipe, gate status, provenance; no build step, no dependencies, hosted on GitHub Pages
- **94 tests**, all passing

## What's left

- **Real-model QA number that means something** — 0.14 measures exact-match harshness on long-form answers, not model quality. Next: short-answer subset or LLM-judged scoring
- **Bigger, real corpus** — 50 hand-written docs is a starter set; dense recall@10 = 1.00 is partly a 50-doc ceiling effect (recall@5 is the discriminating column)
- **More lanes** — BM25, hybrid retrieval; more metrics — MRR, nDCG
- **Multi-label relevance** — one relevant doc per query makes recall binary per query
- **Auto-refresh hosting** — the hosted dashboard reads a committed snapshot; refreshing it after new runs is currently manual

## Honest notes

- **Lexical 1.00 measures the harness, not retrieval strength.** It verifies indexing, ranking, scoring and gating work end to end; the paraphrase tier is where quality separates.
- **TF-IDF's 0.70 is what's left after luck** — zero on true vocabulary mismatch, saved by stop words and near-duplicates. Paraphrases are natural phrasing, not adversarial strings tuned to push the number down.
- **Bug the benchmark caught:** scikit-learn's default `token_pattern` drops single-character tokens, so `2 + 2` scored 0. Fixed with a regression test.
- **Free-tier reality:** Gemini allows ~20 requests/day per model — the 50-question set needs quota headroom or billing to finish in one run (checkpoints make multi-day runs painless).
- **Model cached in CI** (`~/.cache/huggingface`); cold runs download `all-MiniLM-L6-v2` (~90 MB).
