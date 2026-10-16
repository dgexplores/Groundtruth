"""Evaluate against the golden set and emit a machine-readable report.

Two modes:
  retrieval (default) runs the Recall@K benchmark over evals/corpus.json,
    one cell per retriever (--retrievers) and query tier, with the
    headline cell also reported as top-level recall_at_5 / recall_at_10
  qa scores exact-match answers against evals/golden_dataset.json, using
    the stub lookup by default or a real model with --predictor real
    (needs OPENAI_API_KEY; model from QA_MODEL, default gpt-4o-mini)

Reports are the input contract for ``evals.regression_gate``. The qa
mode reports ``exact_match``; the retrieval mode reports
``recall_at_5`` and ``recall_at_10`` (headline cell). The two must not
be conflated: exact_match scores an answer string, recall scores a
retrieved set.
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

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
QA_SYSTEM_PROMPT = "Answer the question briefly. Reply with the answer only, no explanation."

STUB_MODEL_ID = "stub:not-a-real-model"


def load_data(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"dataset must be a JSON list, got {type(data).__name__}")
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
) -> str:
    """Answer with the chat-completions API. Key from OPENAI_API_KEY.

    The openai import is lazy so the module stays importable without the
    dependency (CI runs the stub). Raises ValueError when no key is set.
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
    return content if content is not None else ""


def run_evaluation(
    data: List[Dict[str, Any]], predict: Callable[[str], str] = simulate_llm
) -> List[Dict[str, Any]]:
    results = []
    for item in data:
        input_text = item["input"]
        expected = item["expected_output"]
        actual = predict(input_text)
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
    results: List[Dict[str, Any]], model: str = STUB_MODEL_ID
) -> Dict[str, Any]:
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
    }


def main(argv: List[str] | None = None) -> int:
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
        choices=("stub", "real"),
        default="stub",
        help="qa mode only: stub lookup table (default) or a real model "
        f"(needs {QA_API_KEY_ENV})",
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
        headline: Tuple[str, str] | None = None
        if args.headline:
            parts = args.headline.split(":", 1)
            if len(parts) != 2 or not all(parts):
                parser.error(f"headline must be 'retriever:tier', got {args.headline!r}")
            headline = (parts[0], parts[1])
        report = run_matrix(
            load_corpus(args.corpus),
            retrievers={alias: DEFAULT_RETRIEVERS[alias] for alias in aliases},
            headline=headline,
        )
    else:
        if not args.dataset.exists():
            parser.error(f"dataset not found: {args.dataset}")
        if args.predictor == "real":
            if not os.environ.get(QA_API_KEY_ENV):
                parser.error(f"--predictor real needs {QA_API_KEY_ENV} in the environment")
            model_id = os.environ.get(QA_MODEL_ENV, DEFAULT_QA_MODEL)
            predict: Callable[[str], str] = lambda text: real_llm_predict(
                text, model=model_id
            )
        else:
            model_id = STUB_MODEL_ID
            predict = simulate_llm
        report = build_report(
            run_evaluation(load_data(args.dataset), predict=predict), model=model_id
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
    print(f"report:      {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
