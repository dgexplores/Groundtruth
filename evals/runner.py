"""Evaluate against the golden set and emit a machine-readable report.

Two modes:
  retrieval (default) runs the Recall@K benchmark over evals/corpus.json,
    one cell per retriever (--retrievers) and query tier, with the
    headline cell also reported as top-level recall_at_5 / recall_at_10
  qa scores exact-match answers against evals/golden_dataset.json, using
    the stub lookup by default, an OpenAI model with --predictor real
    (needs OPENAI_API_KEY), or a Gemini model with --predictor gemini
    (needs GEMINI_API_KEY; model from QA_MODEL, else a per-provider default)

Reports are the input contract for ``evals.regression_gate``. The qa
mode reports ``exact_match``; the retrieval mode reports
``recall_at_5`` and ``recall_at_10`` (headline cell). The two must not
be conflated: exact_match scores an answer string, recall scores a
retrieved set.
"""

import argparse
import json
import os
import platform
import re
import subprocess
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from evals.benchmark import (
    DEFAULT_CORPUS,
    DEFAULT_RETRIEVERS,
    load_corpus,
    run_matrix,
)

DEFAULT_DATASET = Path(__file__).resolve().parent / "golden_dataset.json"
DEFAULT_REPORT_DIR = Path(__file__).resolve().parent / "reports"
DEFAULT_RETRIEVER_NAMES = "tfidf,dense"
DEFAULT_HEADLINE_CELL = "dense:semantic"

QA_API_KEY_ENV = "OPENAI_API_KEY"
QA_MODEL_ENV = "QA_MODEL"
DEFAULT_QA_MODEL = "gpt-4o-mini"
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
QA_SYSTEM_PROMPT = "Answer the question briefly. Reply with the answer only, no explanation."

STUB_MODEL_ID = "stub:not-a-real-model"

REPO_ROOT = Path(__file__).resolve().parent.parent

# USD per 1M tokens (input, output). Estimates, October 2026 — verify
# against provider pricing before relying on them for real spend.
PRICE_PER_MTOK = {
    "gpt-4o-mini": (0.15, 0.60),
    "gemini-2.5-flash": (0.30, 2.50),
}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_sha() -> str:
    """Short HEAD sha, or 'unknown' outside a git checkout."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return "unknown"
    sha = completed.stdout.strip()
    return sha if completed.returncode == 0 and sha else "unknown"


def build_run_manifest(extra: dict[str, Any], started_at: str) -> dict[str, Any]:
    """Provenance block attached to every report.

    Records what produced the numbers so a report is reproducible and
    attributable: unique run id, wall-clock window, code version, and
    the caller's config. Never contains secrets (keys stay in env).
    """
    return {
        "run_id": uuid4().hex[:12],
        "started_at": started_at,
        "finished_at": _utcnow(),
        "git_sha": _git_sha(),
        "python": platform.python_version(),
        **extra,
    }


def load_data(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise TypeError(f"dataset must be a JSON list, got {type(data).__name__}")
    return data


def simulate_llm(input_text: str) -> str:
    """Deterministic stand-in for a model call.

    This is a fixed lookup, not a model. Any metric derived from it
    measures the lookup table, so reports always carry
    ``model: stub:not-a-real-model``.
    """
    if "capital of France" in input_text:
        return "Paris"
    if "2 + 2" in input_text:
        return "4"
    if "GIL" in input_text:
        return "Python"
    if "Hermes Agent" in input_text:
        return "Nous Research"
    if "1991" in input_text:
        return "1991"
    if "first version of Python" in input_text:
        return "1991"
    return "Unknown"


def normalize_answer(text: str) -> str:
    """SQuAD-style normalization so trivial surface differences don't fail.

    Lowercase, drop punctuation and the articles a/an/the, collapse
    whitespace. Applied to both sides before exact-match comparison.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def real_llm_predict(
    input_text: str, model: str = DEFAULT_QA_MODEL, api_key: str | None = None
) -> tuple[str, dict[str, int]]:
    """Answer with the chat-completions API. Key from OPENAI_API_KEY.

    Returns the answer plus token usage (prompt/completion counts, 0
    when the API does not report them). The openai import is lazy so
    the module stays importable without the dependency (CI runs the
    stub). Raises ValueError when no key is set.
    """
    key = api_key or os.environ.get(QA_API_KEY_ENV)
    if not key:
        raise ValueError(
            f"real predictor needs {QA_API_KEY_ENV} in the environment"
        )
    from openai import OpenAI

    client = OpenAI(api_key=key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": QA_SYSTEM_PROMPT},
            {"role": "user", "content": input_text},
        ],
        temperature=0,
        max_completion_tokens=256,
        timeout=30,
    )
    content = response.choices[0].message.content
    usage = getattr(response, "usage", None)
    return (content if content is not None else ""), {
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
    }


def gemini_llm_predict(
    input_text: str, model: str = DEFAULT_GEMINI_MODEL, api_key: str | None = None
) -> tuple[str, dict[str, int]]:
    """Answer with the Gemini API. Key from GEMINI_API_KEY.

    Returns the answer plus token usage. The google.genai import is
    lazy so the module stays importable without the dependency.
    Raises ValueError when no key is set.
    """
    key = api_key or os.environ.get(GEMINI_API_KEY_ENV)
    if not key:
        raise ValueError(
            f"gemini predictor needs {GEMINI_API_KEY_ENV} in the environment"
        )
    from google import genai
    from google.genai import errors as genai_errors
    from google.genai import types

    client = genai.Client(api_key=key)
    for attempt in range(4):
        try:
            response = client.models.generate_content(
                model=model,
                contents=input_text,
                config=types.GenerateContentConfig(
                    system_instruction=QA_SYSTEM_PROMPT,
                    temperature=0,
                    max_output_tokens=256,
                ),
            )
            break
        except genai_errors.ServerError:
            if attempt == 3:
                raise
            time.sleep(2**attempt)
        except genai_errors.ClientError as exc:
            if exc.code != 429 or attempt == 3:
                raise
            time.sleep(10 * (attempt + 1))
    metadata = getattr(response, "usage_metadata", None)
    return (response.text if response.text is not None else ""), {
        "prompt_tokens": getattr(metadata, "prompt_token_count", 0) or 0,
        "completion_tokens": getattr(metadata, "candidates_token_count", 0) or 0,
    }


def run_evaluation(
    data: list[dict[str, Any]],
    predict: Callable[[str], str | tuple[str, dict[str, int]]] = simulate_llm,
) -> list[dict[str, Any]]:
    results = []
    for item in data:
        input_text = item["input"]
        expected = item["expected_output"]
        actual = predict(input_text)
        if isinstance(actual, tuple):
            actual = actual[0]
        results.append(
            {
                "input": input_text,
                "expected": expected,
                "actual": actual,
                "match": normalize_answer(actual) == normalize_answer(expected),
            }
        )
    return results


def build_report(
    results: list[dict[str, Any]],
    model: str = STUB_MODEL_ID,
    usage: dict[str, int] | None = None,
    estimated_cost_usd: float | None = None,
) -> dict[str, Any]:
    """Summarise per-item QA results into the gate's input contract."""
    total = len(results)
    matched = sum(1 for r in results if r["match"])
    return {
        "model": model,
        "metric": "exact_match",
        "exact_match": matched / total if total else 0.0,
        "matched": matched,
        "total": total,
        "results": results,
        "usage": usage
        if usage is not None
        else {"prompt_tokens": 0, "completion_tokens": 0, "api_calls": 0},
        "estimated_cost_usd": estimated_cost_usd,
    }


def checkpoint_path_for(out: Path) -> Path:
    """Sidecar holding answered rows so an interrupted run can resume."""
    return Path(str(out) + ".checkpoint.json")


def run_qa_session(
    data: list[dict[str, Any]],
    predict: Callable[[str], str | tuple[str, dict[str, int]]],
    out: Path,
    model: str,
    predictor: str = "stub",
    limit: int | None = None,
    max_calls: int | None = None,
    max_cost_usd: float | None = None,
) -> dict[str, Any]:
    """Answer a dataset with checkpoints and spend guards.

    After every answer the row is written to a checkpoint sidecar, so a
    run killed by quota, keyboard interrupt or crash resumes where it
    stopped instead of re-spending calls. ``limit`` scopes the run to
    the first N items, ``max_calls`` caps new API calls, and
    ``max_cost_usd`` stops before the projected spend exceeds the cap
    (needs a known price for the model — unknown prices fail closed).
    The checkpoint is deleted once every scoped item is answered.
    """
    started_at = _utcnow()
    scoped = data[:limit] if limit is not None else data

    price = PRICE_PER_MTOK.get(model)
    if max_cost_usd is not None and price is None:
        raise ValueError(
            f"no price on file for model {model!r}; cannot enforce --max-cost-usd"
        )

    checkpoint = checkpoint_path_for(out)
    done: dict[str, Any] = {}
    resumed = False
    if checkpoint.exists():
        try:
            with open(checkpoint, encoding="utf-8") as handle:
                saved = json.load(handle)
        except (OSError, ValueError):
            saved = {}
        if isinstance(saved, dict) and saved.get("model") == model:
            rows_by_input = saved.get("rows", {})
            if isinstance(rows_by_input, dict):
                done = rows_by_input
                resumed = True

    rows: list[dict[str, Any]] = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "api_calls": 0}
    spent = 0.0
    calls = 0
    stop_reason: str | None = None

    def note_usage(row_usage: dict[str, int]) -> float:
        usage["prompt_tokens"] += row_usage.get("prompt_tokens", 0)
        usage["completion_tokens"] += row_usage.get("completion_tokens", 0)
        usage["api_calls"] += 1
        if price is None:
            return 0.0
        return (
            row_usage.get("prompt_tokens", 0) * price[0]
            + row_usage.get("completion_tokens", 0) * price[1]
        ) / 1_000_000

    for item in scoped:
        if item["input"] in done:
            row = done[item["input"]]
            rows.append(row)
            spent += note_usage(row.get("usage", {}))
            continue
        if max_calls is not None and calls >= max_calls:
            stop_reason = "max-calls"
            break
        if max_cost_usd is not None and calls > 0:
            projected = spent + spent / calls
            if projected > max_cost_usd:
                stop_reason = "max-cost"
                break
        outcome = predict(item["input"])
        if isinstance(outcome, tuple):
            actual, row_usage = outcome
        else:
            actual, row_usage = outcome, {}
        calls += 1
        row = {
            "input": item["input"],
            "expected": item["expected_output"],
            "actual": actual,
            "match": normalize_answer(actual)
            == normalize_answer(item["expected_output"]),
            "usage": row_usage,
        }
        rows.append(row)
        done[item["input"]] = row
        spent += note_usage(row_usage)
        with open(checkpoint, "w", encoding="utf-8") as handle:
            json.dump({"model": model, "rows": done}, handle)

    completed = len(rows) == len(scoped)
    if completed:
        checkpoint.unlink(missing_ok=True)

    report = build_report(
        rows,
        model=model,
        usage=usage,
        estimated_cost_usd=round(spent, 6) if price is not None else None,
    )
    report["completed"] = completed
    report["stop_reason"] = stop_reason
    report["run"] = build_run_manifest(
        {
            "mode": "qa",
            "predictor": predictor,
            "model": model,
            "limit": limit,
            "max_calls": max_calls,
            "max_cost_usd": max_cost_usd,
            "resumed": resumed,
            "new_calls": calls,
        },
        started_at,
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("retrieval", "qa"),
        default="retrieval",
        help="retrieval: Recall@K benchmark (default); qa: exact-match scoring",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT_DIR / "latest.json")
    parser.add_argument(
        "--retrievers",
        default=DEFAULT_RETRIEVER_NAMES,
        help="comma-separated retriever aliases to run (default: %(default)s)",
    )
    parser.add_argument(
        "--headline",
        default=DEFAULT_HEADLINE_CELL,
        help="retriever:tier cell reported as top-level recall_at_k "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--predictor",
        choices=("stub", "real", "gemini"),
        default="stub",
        help="qa mode only: stub lookup table (default), OpenAI model "
        f"(real, needs {QA_API_KEY_ENV}), or Gemini model "
        f"(gemini, needs {GEMINI_API_KEY_ENV})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="qa mode only: answer the first N dataset items (smoke tests, quotas)",
    )
    parser.add_argument(
        "--max-calls",
        type=int,
        default=None,
        help="qa mode only: stop after N new API calls this run",
    )
    parser.add_argument(
        "--max-cost-usd",
        type=float,
        default=None,
        help="qa mode only: stop before projected spend exceeds this (USD)",
    )
    args = parser.parse_args(argv)

    if args.mode == "retrieval":
        if not args.corpus.exists():
            parser.error(f"corpus not found: {args.corpus}")
        aliases = [alias.strip() for alias in args.retrievers.split(",") if alias.strip()]
        unknown = [alias for alias in aliases if alias not in DEFAULT_RETRIEVERS]
        if unknown:
            parser.error(
                f"unknown retriever(s): {unknown}; "
                f"known: {sorted(DEFAULT_RETRIEVERS)}"
            )
        headline: tuple[str, str] | None = None
        if args.headline:
            parts = args.headline.split(":", 1)
            if len(parts) != 2 or not all(parts):
                parser.error(f"headline must be 'retriever:tier', got {args.headline!r}")
            headline = (parts[0], parts[1])
        retrieval_started = _utcnow()
        report = run_matrix(
            load_corpus(args.corpus),
            retrievers={alias: DEFAULT_RETRIEVERS[alias] for alias in aliases},
            headline=headline,
        )
        report["run"] = build_run_manifest(
            {
                "mode": "retrieval",
                "retrievers": aliases,
                "headline": f"{report['headline_retriever']}:{report['headline_tier']}",
            },
            retrieval_started,
        )
    else:
        if not args.dataset.exists():
            parser.error(f"dataset not found: {args.dataset}")
        if args.limit is not None and args.limit < 1:
            parser.error("--limit must be >= 1")
        if args.max_calls is not None and args.max_calls < 1:
            parser.error("--max-calls must be >= 1")
        if args.max_cost_usd is not None and args.max_cost_usd < 0:
            parser.error("--max-cost-usd must be >= 0")
        if args.predictor in ("real", "gemini"):
            if args.predictor == "real":
                key_env, default_model = QA_API_KEY_ENV, DEFAULT_QA_MODEL
                factory = real_llm_predict
            else:
                key_env, default_model = GEMINI_API_KEY_ENV, DEFAULT_GEMINI_MODEL
                factory = gemini_llm_predict
            if not os.environ.get(key_env):
                parser.error(
                    f"--predictor {args.predictor} needs {key_env} "
                    "in the environment"
                )
            model_id = os.environ.get(QA_MODEL_ENV, default_model)
            predict: Callable[[str], str | tuple[str, dict[str, int]]] = (
                lambda text: factory(text, model=model_id)
            )
        else:
            model_id = STUB_MODEL_ID
            predict = simulate_llm
        report = run_qa_session(
            load_data(args.dataset),
            predict,
            args.out,
            model_id,
            predictor=args.predictor,
            limit=args.limit,
            max_calls=args.max_calls,
            max_cost_usd=args.max_cost_usd,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    if args.mode == "retrieval":
        print(f"headline:    {report['headline_retriever']}:{report['headline_tier']}")
        for k in (5, 10):
            print(f"recall@{k}:  {report[f'recall_at_{k}']:.4f}")
        for alias in report["retrievers"]:
            for tier in sorted(report["cells"][alias]):
                for k in (5, 10):
                    value = report[f"{alias}_{tier}_recall_at_{k}"]
                    print(f"{alias}/{tier} recall@{k}: {value:.4f}")
    else:
        print(f"model:       {report['model']}")
        print(
            f"exact_match: {report['exact_match']:.4f} "
            f"({report['matched']}/{report['total']})"
        )
        usage = report["usage"]
        print(
            f"usage:       {usage['api_calls']} calls, "
            f"{usage['prompt_tokens']} in / {usage['completion_tokens']} out tokens"
        )
        if report["estimated_cost_usd"] is not None:
            print(f"est. cost:   ${report['estimated_cost_usd']:.6f}")
        if not report["completed"]:
            print(f"incomplete:  stopped ({report['stop_reason']}), resume by rerunning")
    print(f"report:      {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
