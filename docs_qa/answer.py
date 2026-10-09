"""Ответ на вопрос по найденным фрагментам: через нейросеть или без неё."""
from __future__ import annotations

import os

from .index import Chunk

MIN_SCORE = 0.08  # ниже этого порога считаем, что в документах ответа нет

PROMPT = """Ты помощник сотрудников компании. Отвечай только по фрагментам документов ниже.
Если ответа во фрагментах нет, так и скажи: «В документах этого нет» — и ничего не придумывай.
Отвечай коротко, по-русски, и в конце укажи номера фрагментов, на которые опирался, например [1] или [1, 3].

Фрагменты:
{context}

Вопрос: {question}"""


def build_prompt(question: str, found: list[tuple[float, Chunk]]) -> str:
    context = "\n\n".join(f"[{i}] {c.source}, раздел «{c.section}»:\n{c.text}" for i, (_, c) in enumerate(found, 1))
    return PROMPT.format(context=context, question=question)


def make_llm():
    """Возвращает функцию «текст запроса → ответ» или None, если нейросеть не настроена."""
    provider = os.getenv("LLM_PROVIDER", "none").lower()
    if provider == "gigachat":
        from gigachat import GigaChat
        from gigachat.models import Chat, Messages, MessagesRole

        client = GigaChat(credentials=os.environ["GIGACHAT_CREDENTIALS"],
                          scope=os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS"),
                          verify_ssl_certs=os.getenv("GIGACHAT_VERIFY_SSL", "false").lower() == "true")
        model = os.getenv("GIGACHAT_MODEL") or client.get_models().data[0].id_

        def ask(prompt):
            chat = Chat(model=model, temperature=0.1, messages=[Messages(role=MessagesRole.USER, content=prompt)])
            return client.chat(chat).choices[0].message.content
        return ask
    if provider == "openai":
        from openai import OpenAI

        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=os.getenv("OPENAI_BASE_URL") or None)

        def ask(prompt):
            r = client.chat.completions.create(model=os.environ["OPENAI_MODEL"], temperature=0.1,
                                               messages=[{"role": "user", "content": prompt}])
            return r.choices[0].message.content
        return ask
    return None


def answer(question: str, found: list[tuple[float, Chunk]], ask=None) -> str:
    sources = "\n".join(f"[{i}] {c.source} — {c.section}" for i, (_, c) in enumerate(found, 1))
    if not found or found[0][0] < MIN_SCORE:
        return "В документах этого нет."
    if ask is None:
        # Без нейросети показываем самый подходящий фрагмент как есть.
        top = found[0][1]
        return f"{top.text}\n\nИсточник: {top.source}, раздел «{top.section}»"
    return f"{ask(build_prompt(question, found)).strip()}\n\nИсточники:\n{sources}"
