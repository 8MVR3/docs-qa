"""Загрузка документов, нарезка на фрагменты и поиск по ним."""
from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

SUPPORTED = {".md", ".txt", ".pdf", ".docx"}


@dataclass
class Chunk:
    source: str
    section: str
    text: str


def read_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in (".md", ".txt"):
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".pdf":
        import pdfplumber

        with pdfplumber.open(path) as pdf:
            return "\n\n".join(page.extract_text() or "" for page in pdf.pages)
    if suffix == ".docx":
        import docx

        document = docx.Document(path)
        lines = []
        for p in document.paragraphs:
            style = (p.style.name or "").lower() if p.style is not None else ""
            text = p.text.strip()
            if text and style.startswith("heading"):
                level = "".join(ch for ch in style if ch.isdigit()) or "1"
                lines.append("#" * int(level) + " " + text)
            else:
                lines.append(text)
        return "\n\n".join(lines)
    raise ValueError(f"Формат {suffix} не поддерживается")


def split_into_chunks(source: str, text: str, max_chars: int = 700) -> list[Chunk]:
    """Режет текст по абзацам. К каждому фрагменту привязывает заголовок раздела:
    без него фраза «до пяти дней» теряет смысл, а с ним поиск понимает, о чём речь."""
    chunks, title, section, buf = [], "", "", []

    def flush():
        if buf:
            chunks.append(Chunk(source, section or title, "\n".join(buf).strip()))
            buf.clear()

    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if not block:
            continue
        heading = re.match(r"^(#{1,6})\s+(.*)", block)
        if heading:
            flush()
            if len(heading.group(1)) == 1:
                title = heading.group(2).strip()
                section = title
            else:
                section = f"{title}. {heading.group(2).strip()}" if title else heading.group(2).strip()
            continue
        if buf and sum(len(b) for b in buf) + len(block) > max_chars:
            flush()
        buf.append(block)
    flush()
    return chunks


class DocsIndex:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        # Символьные n-граммы терпимы к русским окончаниям: «отпуска», «отпуском» и «отпуск» совпадают по большей части.
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, lowercase=True)
        self.matrix = self.vectorizer.fit_transform([f"{c.section}\n{c.text}" for c in chunks])

    @classmethod
    def from_folder(cls, folder: str | Path) -> "DocsIndex":
        chunks = []
        for path in sorted(Path(folder).rglob("*")):
            if path.suffix.lower() in SUPPORTED and path.is_file():
                chunks.extend(split_into_chunks(path.name, read_text(path)))
        if not chunks:
            raise ValueError(f"В папке {folder} нет документов {', '.join(sorted(SUPPORTED))}")
        return cls(chunks)

    def search(self, query: str, k: int = 3) -> list[tuple[float, Chunk]]:
        scores = linear_kernel(self.vectorizer.transform([query]), self.matrix).ravel()
        best = scores.argsort()[::-1][:k]
        return [(float(scores[i]), self.chunks[i]) for i in best]

    def save(self, path: str | Path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "DocsIndex":
        with open(path, "rb") as f:
            return pickle.load(f)
