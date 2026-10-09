import json
from pathlib import Path

from docs_qa.answer import answer, build_prompt
from docs_qa.index import DocsIndex, split_into_chunks

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def test_chunks_keep_section_titles():
    text = "# Отпуска\n\n## Деление\n\nМожно делить на части.\n\n## Без сохранения зарплаты\n\nДо пяти дней."
    chunks = split_into_chunks("otpusk.md", text)
    assert [c.section for c in chunks] == ["Отпуска. Деление", "Отпуска. Без сохранения зарплаты"]


def test_search_finds_right_document():
    idx = DocsIndex.from_folder(EXAMPLES / "docs")
    rows = [json.loads(l) for l in (EXAMPLES / "questions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    top3 = sum(r["source"] in [c.source for _, c in idx.search(r["question"], 3)] for r in rows)
    assert top3 == len(rows)


def test_answer_with_llm_cites_sources():
    idx = DocsIndex.from_folder(EXAMPLES / "docs")
    found = idx.search("Какие суточные за рубежом?", 3)
    prompt = build_prompt("Какие суточные за рубежом?", found)
    assert "[1]" in prompt and "komandirovki.md" in prompt
    result = answer("Какие суточные за рубежом?", found, ask=lambda p: "50 евро в день [1]")
    assert result.startswith("50 евро") and "Источники:" in result


def test_no_context_means_no_answer():
    assert answer("что угодно", []) == "В документах этого нет."
