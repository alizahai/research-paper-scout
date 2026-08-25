"""Streamlit UI for Research Paper Scout.

Talks to the FastAPI backend over HTTP and owns no analysis logic of its own.

Streamlit re-runs this whole script on every interaction, so anything that has
to outlive a click lives in ``st.session_state`` rather than in a local:

- ``analysis``: the decoded ``POST /analyze`` payload, or ``None`` before the
  first search, which is also the flag for whether to draw the lower sections.
- ``chat``: the running list of chat turns, replayed on each re-run.
- ``paper_input``: the search box, keyed so "New Search" can clear it.

Start the backend first, then run ``streamlit run frontend/app.py`` from the
repo root. Point ``SCOUT_BACKEND_URL`` elsewhere if the backend is not on the
default port.
"""

from __future__ import annotations

import os
from typing import Any, Final, Iterator, Sequence

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BACKEND_URL: Final = os.getenv("SCOUT_BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")

# /analyze walks four Semantic Scholar endpoints and makes a Gemini call per
# theme, and without a Semantic Scholar API key the rate-limit backoff alone can
# run into minutes, so it gets a far longer budget than a single question does.
ANALYZE_TIMEOUT_SECONDS: Final = 600
ASK_TIMEOUT_SECONDS: Final = 180

# Papers that HDBSCAN treated as outliers come back with clusterLabel = null.
UNCLUSTERED_HEADING: Final = "Other"

# How many earlier turns travel with a question. The backend caps this too; the
# limit is repeated here so a long conversation does not grow every request.
HISTORY_TURN_LIMIT: Final = 4

CARD_COLUMNS: Final = 2


# ---------------------------------------------------------------------------
# Backend calls
# ---------------------------------------------------------------------------


class BackendError(Exception):
    """A failed backend call, carrying text that is fit to show a user."""


def _post(path: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    """POST JSON to the backend and return the decoded body.

    Raises:
        BackendError: On any transport failure or non-2xx response, described in
            plain language so callers can hand it straight to ``st.error``.
    """
    try:
        response = requests.post(
            f"{BACKEND_URL}{path}", json=payload, timeout=timeout
        )
    except requests.Timeout as exc:
        raise BackendError(
            f"The backend did not answer within {timeout} seconds. Semantic "
            "Scholar rate limiting can make a first analysis slow — try again, "
            "or set SEMANTIC_SCHOLAR_API_KEY in backend/.env to speed it up."
        ) from exc
    except requests.ConnectionError as exc:
        raise BackendError(
            f"Could not reach the backend at {BACKEND_URL}. Start it with "
            "`uvicorn main:app --reload` from the backend/ directory."
        ) from exc
    except requests.RequestException as exc:
        raise BackendError(f"The request to {path} failed: {exc}") from exc

    if not response.ok:
        raise BackendError(_error_text(response))

    try:
        return response.json()
    except ValueError as exc:
        raise BackendError(
            f"The backend answered {response.status_code} with a body that was "
            "not JSON."
        ) from exc


def _error_text(response: requests.Response) -> str:
    """Pull a readable message out of an error response.

    FastAPI puts the messages raised by the routes into ``detail`` as a string,
    but reports its own request-validation failures as a list of per-field
    objects under that same key, so both shapes have to be unpacked. Anything
    unrecognised falls back to the raw body.
    """
    try:
        body = response.json()
    except ValueError:
        body = None

    detail = body.get("detail") if isinstance(body, dict) else None

    if isinstance(detail, str) and detail.strip():
        return detail

    if isinstance(detail, list):
        problems = []
        for item in detail:
            if not isinstance(item, dict):
                continue
            location = ".".join(
                str(part) for part in item.get("loc", ()) if part != "body"
            )
            message = item.get("msg") or "invalid value"
            problems.append(f"{location}: {message}" if location else message)
        if problems:
            return "; ".join(problems)

    raw = (response.text or "").strip()
    return raw[:500] or f"The backend returned HTTP {response.status_code}."


def run_analysis(paper_input: str) -> dict[str, Any]:
    """Run the full analysis pipeline for one seed input."""
    return _post(
        "/analyze", {"paper_input": paper_input}, ANALYZE_TIMEOUT_SECONDS
    )


def ask_question(
    session_id: str, question: str, history: list[dict[str, str]]
) -> dict[str, Any]:
    """Ask one question against an already-analysed session."""
    return _post(
        "/ask",
        {"sessionId": session_id, "question": question, "history": history},
        ASK_TIMEOUT_SECONDS,
    )


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------


def init_state() -> None:
    """Seed the keys every later section assumes exist."""
    st.session_state.setdefault("analysis", None)
    st.session_state.setdefault("chat", [])
    st.session_state.setdefault("paper_input", "")


def reset_state() -> None:
    """Clear the previous search so the page returns to its opening state.

    Assigning to ``paper_input`` only works because the sidebar is drawn before
    the search box: Streamlit rejects writes to a widget's key after that widget
    has been created in the current run.
    """
    st.session_state.analysis = None
    st.session_state.chat = []
    st.session_state.paper_input = ""


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def render_sidebar() -> None:
    """Show what the current session holds, and the way out of it."""
    with st.sidebar:
        st.header("Session")

        analysis = st.session_state.analysis
        if analysis is None:
            st.caption("Nothing analysed yet.")
        else:
            papers = analysis.get("papers") or []
            left, right = st.columns(2)
            left.metric("Papers", len(papers))
            right.metric("Themes", analysis.get("clusterCount", 0))
            st.caption("Session ID")
            st.code(analysis.get("sessionId", ""), language=None)

        if st.button("New Search", width="stretch"):
            reset_state()
            st.rerun()

        st.divider()
        st.caption(f"Backend: {BACKEND_URL}")


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def render_search() -> None:
    """Draw the search box and run an analysis when it is submitted."""
    with st.form("search"):
        query = st.text_input(
            "arXiv URL or ID",
            key="paper_input",
            placeholder="https://arxiv.org/abs/1706.03762",
            help=(
                "An arXiv URL or bare ID goes straight to that paper. Any other "
                "text is treated as a search, and the closest match becomes the "
                "starting point."
            ),
        )
        submitted = st.form_submit_button("Analyze", type="primary")

    if not submitted:
        return

    query = (query or "").strip()
    if not query:
        st.warning("Enter an arXiv URL, an arXiv ID, or a topic to search for.")
        return

    with st.spinner("Fetching papers, grouping them into themes, writing the briefing…"):
        try:
            analysis = run_analysis(query)
        except BackendError as exc:
            st.error(str(exc))
            return

    # A fresh corpus invalidates the old conversation, so the history goes too.
    st.session_state.analysis = analysis
    st.session_state.chat = []
    st.rerun()


# ---------------------------------------------------------------------------
# Briefing and paper cards
# ---------------------------------------------------------------------------


def render_briefing(analysis: dict[str, Any]) -> None:
    """Show the landscape briefing as a highlighted block."""
    papers = analysis.get("papers") or []
    theme_count = analysis.get("clusterCount", 0)

    st.subheader("Landscape briefing")
    st.caption(
        f"{len(papers)} {_plural(len(papers), 'paper')} · "
        f"{theme_count} {_plural(theme_count, 'theme')}"
    )
    st.info(analysis.get("landscapeBriefing") or "The briefing came back empty.")


def render_papers(papers: Sequence[dict[str, Any]]) -> None:
    """Show every paper as a card, grouped under its theme."""
    st.subheader("Papers by theme")

    if not papers:
        st.caption("No papers were returned for this search.")
        return

    for heading, group in _group_by_theme(papers):
        st.markdown(f"#### {heading} ({len(group)})")
        for row in _in_rows(group, CARD_COLUMNS):
            for column, paper in zip(st.columns(CARD_COLUMNS), row):
                with column:
                    _render_card(paper)


def _group_by_theme(
    papers: Sequence[dict[str, Any]],
) -> list[tuple[str, list[dict[str, Any]]]]:
    """Bucket papers by theme label, largest theme first and outliers last."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for paper in papers:
        heading = paper.get("clusterLabel") or UNCLUSTERED_HEADING
        groups.setdefault(heading, []).append(paper)

    themes = sorted(
        ((name, group) for name, group in groups.items() if name != UNCLUSTERED_HEADING),
        key=lambda item: (-len(item[1]), item[0].lower()),
    )
    if UNCLUSTERED_HEADING in groups:
        themes.append((UNCLUSTERED_HEADING, groups[UNCLUSTERED_HEADING]))

    return themes


def _in_rows(
    papers: Sequence[dict[str, Any]], width: int
) -> Iterator[Sequence[dict[str, Any]]]:
    """Split papers into rows of at most ``width`` cards."""
    for start in range(0, len(papers), width):
        yield papers[start : start + width]


def _render_card(paper: dict[str, Any]) -> None:
    """Draw one paper as a bordered card."""
    with st.container(border=True):
        st.markdown(f"**{paper.get('title') or 'Untitled paper'}**")

        facts = []
        year = paper.get("year")
        if year:
            facts.append(str(year))
        citations = paper.get("citationCount")
        if citations is not None:
            facts.append(f"{citations:,} {_plural(citations, 'citation')}")
        st.caption(" · ".join(facts) if facts else "No year or citation data")

        tldr = paper.get("tldr")
        st.write(tldr if tldr else "_No TL;DR available for this paper._")


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------


def render_chat(session_id: str) -> None:
    """Draw the conversation and handle a newly submitted question.

    History is replayed from session state first, then a new question is drawn
    live so the spinner appears in place rather than after a re-run.
    """
    st.subheader("Ask about these papers")
    st.caption(
        "Answers are grounded in the papers above, not in the model's own "
        "knowledge."
    )

    for turn in st.session_state.chat:
        with st.chat_message(turn["role"]):
            _render_turn(turn)

    question = st.chat_input("What do these papers agree on?")
    if question is None or not question.strip():
        return

    question = question.strip()

    # Read before the new question is appended, so it carries the conversation
    # up to but not including itself.
    history = _history_payload()

    user_turn = {"role": "user", "content": question}
    st.session_state.chat.append(user_turn)
    with st.chat_message("user"):
        _render_turn(user_turn)

    with st.chat_message("assistant"):
        with st.spinner("Reading the retrieved papers…"):
            try:
                payload = ask_question(session_id, question, history)
            except BackendError as exc:
                # Kept in the history so the failure stays attached to the
                # question that caused it after the next re-run.
                answer_turn = {
                    "role": "assistant",
                    "content": str(exc),
                    "failed": True,
                }
            else:
                answer_turn = {
                    "role": "assistant",
                    "content": payload.get("answer") or "",
                    "sources": payload.get("sourcePapers") or [],
                }

        st.session_state.chat.append(answer_turn)
        _render_turn(answer_turn)


def _history_payload() -> list[dict[str, str]]:
    """The last few completed turns, in the shape ``/ask`` expects.

    Failed turns are left out: their content is a local error message, not
    something the model said, and sending it back would have the model treat its
    own outage as part of the conversation.
    """
    completed = [
        {"role": turn["role"], "content": turn["content"]}
        for turn in st.session_state.chat
        if not turn.get("failed")
    ]
    return completed[-HISTORY_TURN_LIMIT:]


def _render_turn(turn: dict[str, Any]) -> None:
    """Draw one turn's body, assuming a chat bubble is already open."""
    if turn.get("failed"):
        st.error(turn["content"])
        return

    st.markdown(turn["content"] or "_The answer came back empty._")

    sources = turn.get("sources")
    if sources:
        with st.expander(f"Sources ({len(sources)})"):
            for title in sources:
                st.markdown(f"- {title}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _plural(count: int, noun: str) -> str:
    """Return ``noun`` pluralised for ``count``, for the simple ``+s`` cases."""
    return noun if count == 1 else f"{noun}s"


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Research Paper Scout", layout="wide")

init_state()

# Drawn before the search box on purpose: see reset_state.
render_sidebar()

st.title("Research Paper Scout")
st.caption(
    "Map the literature around a paper: fetch its neighbourhood, group it into "
    "themes, and ask questions grounded in what was found."
)

render_search()

analysis = st.session_state.analysis

# Everything below depends on an analysis having run, so it stays hidden until
# one has.
if analysis is None:
    st.info("Analyse a paper to see its landscape briefing, themes and chat.")
else:
    st.divider()
    render_briefing(analysis)
    st.divider()
    render_papers(analysis.get("papers") or [])
    st.divider()
    render_chat(analysis["sessionId"])

# Attribution, required by the Semantic Scholar API licence terms.
st.divider()
st.caption(
    "Paper data provided by [Semantic Scholar](https://www.semanticscholar.org/)."
)
