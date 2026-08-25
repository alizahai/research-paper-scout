"""Pydantic schemas for request and response payloads.

Response field names are camelCase to match what the Streamlit frontend and any
JavaScript client would expect, and mirror the keys the phase modules already
produce (``paperId``, ``citationCount``, ``tldr``).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


def _non_blank(value: str) -> str:
    """Strip surrounding whitespace and reject values that were only whitespace."""
    stripped = value.strip()
    if not stripped:
        raise ValueError("must not be blank")
    return stripped


class HealthResponse(BaseModel):
    """Liveness probe payload."""

    status: str


class AnalyzeRequest(BaseModel):
    """Request body for ``POST /analyze``."""

    paper_input: str = Field(
        min_length=1,
        description=(
            "An arXiv URL or bare arXiv ID identifying the seed paper, for "
            "example 'https://arxiv.org/abs/1706.03762' or '1706.03762'."
        ),
    )

    @field_validator("paper_input")
    @classmethod
    def _clean_paper_input(cls, value: str) -> str:
        return _non_blank(value)


class PaperSummary(BaseModel):
    """One paper as returned to the client.

    Every field except ``paperId`` may be absent: Semantic Scholar leaves
    abstracts, years and TL;DRs off many records, and ``clusterLabel`` is unset
    for papers HDBSCAN treated as outliers.
    """

    paperId: str
    title: str | None = None
    year: int | None = None
    citationCount: int | None = None
    tldr: str | None = None
    clusterLabel: str | None = None


class AnalyzeResponse(BaseModel):
    """Result of the full fetch, index, cluster and brief pipeline."""

    sessionId: str = Field(
        description="Pass this to POST /ask to query this corpus. Valid until "
        "the server restarts."
    )
    papers: list[PaperSummary]
    landscapeBriefing: str
    clusterCount: int = Field(
        description="Number of themes found, excluding the outlier group."
    )


class ChatTurn(BaseModel):
    """One earlier turn of a conversation, sent back to resolve references.

    The wire form is a plain ``{"role": ..., "content": ...}`` object; typing it
    here means a malformed turn is reported as a field error rather than being
    quietly dropped or reaching the prompt.
    """

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)

    @field_validator("content")
    @classmethod
    def _clean_content(cls, value: str) -> str:
        return _non_blank(value)


class QuestionRequest(BaseModel):
    """Request body for ``POST /ask``."""

    sessionId: str = Field(min_length=1, description="From a prior /analyze call.")
    question: str = Field(min_length=1, description="Natural-language question.")
    history: list[ChatTurn] = Field(
        default_factory=list,
        description=(
            "Earlier turns of this conversation, oldest first, excluding the "
            "question being asked now. Optional. Only the last few are used, and "
            "only so the model can resolve references like 'that one'; retrieval "
            "still runs on the question text alone."
        ),
    )

    @field_validator("sessionId", "question")
    @classmethod
    def _clean_fields(cls, value: str) -> str:
        return _non_blank(value)


class QuestionResponse(BaseModel):
    """Grounded answer plus the papers it drew on."""

    answer: str
    sourcePapers: list[str] = Field(
        description="Titles of the papers retrieved for this question, in "
        "descending similarity order."
    )
