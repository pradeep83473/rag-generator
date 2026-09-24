"""Measure retrieval and answer quality against a labelled question set.

    python -m app.evaluate eval/handbook.json eval/product.json

Each spec file names a folder of documents and a list of cases:

    {"name": "...", "documents": "../samples/handbook",
     "cases": [{"question": "...", "expected_source": "leave-policy.md",
                "expected_answer": ["5", "31 March"]},
               {"question": "...", "answerable": false}]}

Paths are relative to the spec file. Every spec gets a fresh knowledge base in a
temporary data directory, so evaluation never touches real collections. Any
document set can be evaluated this way; nothing here is specific to the samples.

Metrics, per answerable case:
  retrieval  the expected file is among the retrieved passages
  citation   the expected file is among the passages the answer cites
  answer     every expected_answer term appears in the answer (case-insensitive)
  grounded   the answer is marked grounded
and per unanswerable case:
  declined   the answer is not grounded (the model said it could not find it)
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from app.config import get_settings
from app.ingestion.loaders import SUPPORTED_EXTENSIONS
from app.llm import LLMProvider
from app.service import RAGService

ANSWERABLE_CHECKS = ("retrieval", "citation", "answer", "grounded")


def score_case(case: dict, answer) -> dict[str, bool]:
    if not case.get("answerable", True):
        return {"declined": not answer.grounded}
    expected = case["expected_source"]
    text = answer.answer.lower()
    return {
        "retrieval": any(s["filename"] == expected for s in answer.sources),
        "citation": any(s["filename"] == expected and s["cited"] for s in answer.sources),
        "answer": all(term.lower() in text for term in case.get("expected_answer", [])),
        "grounded": answer.grounded,
    }


def run_spec(service: RAGService, spec_path: Path) -> dict:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    collection = service.collections.create(spec.get("name", spec_path.stem))
    folder = (spec_path.parent / spec["documents"]).resolve()
    for path in sorted(folder.rglob("*")):
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            result = service.ingest(collection, path.name, path.read_bytes())
            if result.status == "error":
                raise RuntimeError(f"{path}: {result.detail}")

    cases = []
    for case in spec["cases"]:
        answer = service.ask(collection, case["question"])
        cases.append(
            {"question": case["question"], "answer": answer.answer,
             "checks": score_case(case, answer)}
        )
    return {"name": spec.get("name", spec_path.stem), "cases": cases}


def summarize(reports: list[dict]) -> dict[str, tuple[int, int]]:
    totals: dict[str, list[int]] = {}
    for report in reports:
        for case in report["cases"]:
            for check, passed in case["checks"].items():
                hit, total = totals.setdefault(check, [0, 0])
                totals[check] = [hit + passed, total + 1]
    order = [*ANSWERABLE_CHECKS, "declined"]
    return {k: tuple(totals[k]) for k in order if k in totals}


def evaluate(spec_paths: list[Path], provider: LLMProvider | None = None) -> tuple[list[dict], dict]:
    with tempfile.TemporaryDirectory() as tmp:
        settings = get_settings().model_copy(update={"data_dir": Path(tmp)})
        service = RAGService(settings, provider)
        reports = [run_spec(service, path) for path in spec_paths]
    return reports, summarize(reports)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.evaluate")
    parser.add_argument("specs", nargs="+", type=Path)
    parser.add_argument("--json", type=Path, help="also write the full report here")
    parser.add_argument("--fail-under", type=float, default=0.0,
                        help="exit non-zero if any metric's pass rate is below this (0-1)")
    args = parser.parse_args(argv)

    reports, summary = evaluate(args.specs)
    for report in reports:
        print(f"\n== {report['name']}")
        for case in report["cases"]:
            failed = [k for k, ok in case["checks"].items() if not ok]
            mark = "PASS" if not failed else "FAIL " + ",".join(failed)
            print(f"  [{mark}] {case['question']}")
            if failed:
                print(f"         -> {case['answer'][:200]}")
    print("\nSummary")
    for check, (hit, total) in summary.items():
        print(f"  {check:<10} {hit}/{total}  ({hit / total:.0%})")
    if args.json:
        args.json.write_text(json.dumps({"summary": summary, "reports": reports}, indent=2),
                             encoding="utf-8")
    return 1 if any(hit / total < args.fail_under for hit, total in summary.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
