"""Command-line access to the same pipeline, handy for scripted evaluation.

    python -m app.cli create "HR policies"
    python -m app.cli ingest <collection_id> ./docs/hr
    python -m app.cli ask <collection_id> "How many days of parental leave?"
    python -m app.cli chat <collection_id>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.config import get_settings
from app.ingestion.loaders import SUPPORTED_EXTENSIONS
from app.service import RAGService


def _print_answer(answer) -> None:
    print(f"\n{answer.answer}\n")
    for source in answer.sources:
        if source["cited"]:
            page = f", p.{source['page']}" if source["page"] else ""
            print(f"  [{source['number']}] {source['filename']}{page}")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rag")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    create = sub.add_parser("create")
    create.add_argument("name")
    create.add_argument("--description", default="")
    create.add_argument("--instructions", default="")
    ingest = sub.add_parser("ingest")
    ingest.add_argument("collection_id")
    ingest.add_argument("paths", nargs="+", type=Path)
    ask = sub.add_parser("ask")
    ask.add_argument("collection_id")
    ask.add_argument("question")
    chat = sub.add_parser("chat")
    chat.add_argument("collection_id")
    args = parser.parse_args(argv)

    service = RAGService(get_settings())

    if args.command == "list":
        for c in service.collections.list():
            s = c.summary()
            print(f"{s['id']}  {s['name']}  ({s['document_count']} docs, {s['chunk_count']} chunks)")
    elif args.command == "create":
        c = service.collections.create(args.name, args.description, args.instructions)
        print(c.id)
    elif args.command == "ingest":
        collection = service.collections.get(args.collection_id)
        files = []
        for path in args.paths:
            if path.is_dir():
                files += [p for p in sorted(path.rglob("*")) if p.suffix.lower() in SUPPORTED_EXTENSIONS]
            else:
                files.append(path)
        for path in files:
            result = service.ingest(collection, path.name, path.read_bytes())
            print(f"{result.status:>9}  {path}  {result.chunks} chunks  {result.detail or ''}")
    elif args.command == "ask":
        _print_answer(service.ask(service.collections.get(args.collection_id), args.question))
    elif args.command == "chat":
        collection = service.collections.get(args.collection_id)
        history: list[dict[str, str]] = []
        print(f"Chatting with '{collection.meta['name']}'. Ctrl-D to exit.")
        for line in sys.stdin if not sys.stdin.isatty() else iter(lambda: input("> "), None):
            question = line.strip()
            if not question:
                continue
            answer = service.ask(collection, question, history)
            _print_answer(answer)
            history += [{"role": "user", "content": question},
                        {"role": "assistant", "content": answer.answer}]
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (EOFError, KeyboardInterrupt):
        sys.exit(0)
