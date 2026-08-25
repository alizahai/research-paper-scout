"""Phase 4: retrieval-augmented synthesis via the Gemini API.

Takes the matches from :mod:`embeddings` and asks Gemini to answer a question
using only those papers as evidence. The grounding rules live in
:data:`SYSTEM_INSTRUCTION`; the retrieved papers are formatted into the user turn
so the model can quote titles back when it cites.

Conversation history, when supplied, is rendered into the same user turn ahead of
the question purely so a follow-up like "what about the second one?" can be
understood. It is never evidence, and it never reaches retrieval: the caller has
already chosen the papers from the current question's text alone.

Uses the newer ``google-genai`` SDK (``from google import genai``), not the older
``google-generativeai`` package: a client object, then
``client.models.generate_content(model=..., contents=..., config=...)``.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Final, Sequence

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

__all__ = [
    "HISTORY_TURN_LIMIT",
    "MODEL_NAME",
    "MissingAPIKeyError",
    "NO_PAPERS_MESSAGE",
    "SynthesisError",
    "answer_question",
    "generate_text",
]

MODEL_NAME: Final = "gemini-2.5-flash"
API_KEY_ENV_VAR: Final = "GEMINI_API_KEY"

# Resolved from this file rather than the working directory, so the key is found
# whether uvicorn is started from the repo root or from backend/.
_ENV_PATH: Final = Path(__file__).resolve().parent / ".env"

# Low but non-zero: the answer should follow the sources, not improvise.
_TEMPERATURE: Final = 0.2

# Enough for a follow-up to resolve what it refers to, without letting an old
# exchange grow the prompt — and its cost — on every question.
HISTORY_TURN_LIMIT: Final = 4

# How each role is introduced in the prompt. Doubles as the set of roles that are
# accepted; anything else is dropped.
_ROLE_LABELS: Final[dict[str, str]] = {"user": "Question", "assistant": "Answer"}

NO_PAPERS_MESSAGE: Final = (
    "I don't have any papers to work from, so I can't answer that yet. "
    "Try indexing a paper's references and citations first, or rephrasing the "
    "question so it matches the papers in this session."
)

SYSTEM_INSTRUCTION: Final = """\
You are a research assistant helping someone understand a set of academic papers.

Ground rules:
- Answer using ONLY the papers provided in the user's message. Do not use outside
  knowledge, even if you are confident it is correct.
- Cite the papers you rely on by their exact title, in quotes, next to the claim
  they support.
- If the provided papers do not answer the question, say so plainly and explain
  what they do cover instead. Do not fill the gap with general knowledge.
- Never invent a title, finding, author or number that is not in the provided
  text. If a paper's abstract is unavailable, do not speculate about its contents.
- Where papers disagree or point in different directions, say so rather than
  flattening them into one answer.
- Earlier conversation turns, when included, are there only to show what the
  question refers to. Never treat your own previous answer as evidence, and never
  cite a paper that appears only in those turns and not below.
- Write in clear prose for a technical reader. Be concise; no preamble.\
"""

_client: genai.Client | None = None
_client_lock: Final = threading.Lock()


class SynthesisError(Exception):
    """Raised when the Gemini call fails or comes back without usable text."""


class MissingAPIKeyError(SynthesisError):
    """Raised when no :data:`API_KEY_ENV_VAR` is configured."""


def answer_question(
    query: str,
    retrieved_papers: list[dict[str, Any]],
    *,
    history: Sequence[dict[str, Any]] | None = None,
) -> str:
    """Answer ``query`` using only ``retrieved_papers`` as evidence.

    Args:
        query: The user's natural-language question.
        retrieved_papers: Matches from :func:`embeddings.query_collection`, each
            with ``title``, ``abstract``, ``year`` and ``tldr``. Extra keys such
            as ``similarity`` are ignored. Order is preserved in the prompt, so
            pass them most-relevant-first.
        history: Earlier turns as ``{"role": "user" | "assistant", "content":
            str}`` dicts, oldest first, excluding the question being asked now.
            Only the last :data:`HISTORY_TURN_LIMIT` are used, and only so the
            model can tell what a follow-up refers to. Turns with an unknown role
            or empty content are skipped. This has no bearing on retrieval, which
            the caller has already done from ``query`` alone.

    Returns:
        The model's answer, or :data:`NO_PAPERS_MESSAGE` when
        ``retrieved_papers`` is empty — that case returns early without spending
        an API call.

    Raises:
        ValueError: If ``query`` is blank.
        MissingAPIKeyError: If ``GEMINI_API_KEY`` is not set.
        SynthesisError: If the API rejects the request or returns no text, which
            includes responses cut short by a safety filter or the token limit.
    """
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")

    if not retrieved_papers:
        return NO_PAPERS_MESSAGE

    sections: list[str] = []
    recent = _recent_turns(history)

    if recent:
        sections.append(
            "Earlier turns in this conversation, oldest first. Use them only to "
            "work out what the question below refers to — they are not evidence, "
            "and the papers are still the only material you may draw on:\n\n"
            f"{_format_history(recent)}"
        )

    # Without history this is byte-for-byte the single-turn prompt, so the
    # existing behaviour is untouched when no history is passed.
    sections.append(f"{'Current question' if recent else 'Question'}: {query.strip()}")
    sections.append(
        f"Here are the {len(retrieved_papers)} papers retrieved for this "
        f"question:\n\n{_format_papers(retrieved_papers)}"
    )
    sections.append(
        "Answer the question using only these papers, citing titles as you go."
    )

    return generate_text("\n\n".join(sections), SYSTEM_INSTRUCTION)


def generate_text(
    prompt: str,
    system_instruction: str,
    *,
    temperature: float = _TEMPERATURE,
    response_mime_type: str | None = None,
    response_schema: Any = None,
) -> str:
    """Run one single-turn Gemini call and return its text.

    The shared entry point for every Gemini request in the backend, so that
    client setup, error translation and empty-response handling live in exactly
    one place. :mod:`clustering` uses this for theme labels and landscape
    briefings; only the instruction and prompt differ.

    Deliberately does not set ``max_output_tokens``: on the 2.5 models, internal
    reasoning draws from the same budget, so a small cap can consume it entirely
    and yield an empty response instead of a short one.

    Args:
        prompt: The user turn.
        system_instruction: Instruction defining the model's task and limits.
        temperature: Sampling temperature; the low default keeps output close to
            the supplied sources.
        response_mime_type: Set to ``"application/json"`` to have the model
            constrained to JSON syntax rather than merely asked for it.
        response_schema: Optional schema the reply must satisfy, as a plain dict
            in Gemini's OpenAPI subset or anything else the SDK accepts. Only
            meaningful alongside a JSON ``response_mime_type``. Callers pass
            dicts so they need no ``google.genai`` import of their own.

    Returns:
        The response text, stripped. Still a string when JSON was requested:
        parsing is the caller's business, since only it knows the shape.

    Raises:
        MissingAPIKeyError: If ``GEMINI_API_KEY`` is not set.
        SynthesisError: If the API rejects the request or returns no usable text.
    """
    try:
        response = _get_client().models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=temperature,
                response_mime_type=response_mime_type,
                response_schema=response_schema,
            ),
        )
    except errors.APIError as exc:
        raise SynthesisError(
            f"Gemini rejected the request (code {exc.code}): {exc.message}"
        ) from exc

    text = (response.text or "").strip()
    if not text:
        raise SynthesisError(
            f"Gemini returned no text ({_describe_empty_response(response)}). "
            "This usually means the response was blocked or truncated."
        )

    return text


def _recent_turns(
    history: Sequence[dict[str, Any]] | None,
) -> list[tuple[str, str]]:
    """Keep the last :data:`HISTORY_TURN_LIMIT` usable turns, oldest first.

    Malformed turns are skipped rather than raising: history is a convenience for
    phrasing, so a bad entry should cost the model a little context, not fail the
    whole question. Trimming happens after filtering, so unusable entries cannot
    push real turns out of the window.
    """
    if not history:
        return []

    turns: list[tuple[str, str]] = []
    for entry in history:
        if not isinstance(entry, dict):
            continue
        role = _clean(entry.get("role")).lower()
        content = _clean(entry.get("content"))
        if role in _ROLE_LABELS and content:
            turns.append((role, content))

    return turns[-HISTORY_TURN_LIMIT:]


def _format_history(turns: Sequence[tuple[str, str]]) -> str:
    """Render conversation turns as labelled blocks for the prompt."""
    return "\n\n".join(f"{_ROLE_LABELS[role]}: {content}" for role, content in turns)


def _format_papers(papers: Sequence[dict[str, Any]]) -> str:
    """Render papers as numbered blocks for the prompt.

    A missing abstract is stated explicitly rather than left out, so the model
    can see the gap instead of inferring that a paper said nothing.
    """
    blocks: list[str] = []

    for position, paper in enumerate(papers, start=1):
        title = _clean(paper.get("title")) or "Untitled"
        lines = [f"--- Paper {position} ---", f'Title: "{title}"']

        year = paper.get("year")
        if year is not None:
            lines.append(f"Year: {year}")

        tldr = _clean(paper.get("tldr"))
        if tldr:
            lines.append(f"One-line summary: {tldr}")

        abstract = _clean(paper.get("abstract")) or _clean(paper.get("document"))
        lines.append(f"Abstract: {abstract or '(not available for this paper)'}")

        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


def _clean(value: Any) -> str:
    """Return a stripped string for text-ish values, else an empty string."""
    return value.strip() if isinstance(value, str) else ""


def _describe_empty_response(response: Any) -> str:
    """Summarise why a response had no text, for the error message.

    Reads finish/block reasons defensively: they are absent on some responses and
    the SDK has moved them around between versions.
    """
    details: list[str] = []

    feedback = getattr(response, "prompt_feedback", None)
    block_reason = getattr(feedback, "block_reason", None)
    if block_reason:
        details.append(f"prompt block_reason={block_reason}")

    for candidate in getattr(response, "candidates", None) or []:
        finish_reason = getattr(candidate, "finish_reason", None)
        if finish_reason:
            details.append(f"finish_reason={finish_reason}")

    return ", ".join(details) or "no finish reason reported"


def _get_client() -> genai.Client:
    """Return the shared Gemini client, creating it on first use.

    Cached because a client holds connection state that is wasteful to rebuild
    per request, and lock-guarded since FastAPI may call this from several
    threads at once.

    Raises:
        MissingAPIKeyError: If no API key is configured.
    """
    global _client

    if _client is None:
        with _client_lock:
            if _client is None:
                load_dotenv(_ENV_PATH)
                api_key = os.getenv(API_KEY_ENV_VAR, "").strip()
                if not api_key:
                    raise MissingAPIKeyError(
                        f"{API_KEY_ENV_VAR} is not set. Add it to {_ENV_PATH} or "
                        "the environment before asking questions."
                    )
                _client = genai.Client(api_key=api_key)

    return _client
