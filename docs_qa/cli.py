"""python -m docs_qa index папка | ask "вопрос" | eval questions.jsonl"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .answer import answer, make_llm
from .index import DocsIndex

DEFAULT_INDEX = "docs.index"


def main(argv=None) -> int:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    ap = argparse.ArgumentParser(prog="docs_qa", description="Вопросы по документам компании")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("index", help="проиндексировать папку с документами")
    p.add_argument("folder")
    p.add_argument("--out", default=DEFAULT_INDEX)
    p = sub.add_parser("ask", help="задать вопрос")
    p.add_argument("question")
    p.add_argument("--index", default=DEFAULT_INDEX)
    p.add_argument("-k", type=int, default=3, help="сколько фрагментов отдавать нейросети")
    p = sub.add_parser("eval", help="проверить качество поиска на наборе вопросов")
    p.add_argument("questions")
    p.add_argument("--index", default=DEFAULT_INDEX)
    a = ap.parse_args(argv)

    if a.cmd == "index":
        idx = DocsIndex.from_folder(a.folder)
        idx.save(a.out)
        print(f"Фрагментов: {len(idx.chunks)} из {len({c.source for c in idx.chunks})} документов. Индекс: {a.out}")
        return 0

    idx = DocsIndex.load(a.index)
    if a.cmd == "ask":
        print(answer(a.question, idx.search(a.question, a.k), make_llm()))
        return 0

    rows = [json.loads(l) for l in Path(a.questions).read_text(encoding="utf-8").splitlines() if l.strip()]
    hit1 = hit3 = 0
    for r in rows:
        found = [c.source for _, c in idx.search(r["question"], 3)]
        hit1 += found[0] == r["source"]
        hit3 += r["source"] in found
        mark = "✓" if found[0] == r["source"] else ("~" if r["source"] in found else "✗")
        print(f"{mark} {r['question']}  ->  {found[0]}")
    print(f"\nВерный документ первым: {hit1}/{len(rows)} ({hit1 / len(rows):.0%}), в первой тройке: {hit3}/{len(rows)} ({hit3 / len(rows):.0%})")
    return 0
