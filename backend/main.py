"""FastAPI app entrypoint for Research Paper Scout.

Sequences the phase modules and owns no analysis logic of its own: fetching lives
in :mod:`semantic_scholar`, embedding and retrieval in :mod:`embeddings`,
clustering and theme naming in :mod:`clustering`, and grounded answering in
:mod:`synthesis`.

Run from inside ``backend/`` (``uvicorn main:app --reload``), since the modules
import each other by plain name.

Sessions live in a module-level dict, so every indexed corpus is lost when the
process restarts and nothing is shared between workers. Run a single worker while
developing, or the session created by ``/analyze`` may be missing from the worker
that handles the follow-up ``/ask``.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import uuid
from collections import defaultdict
from typing import Any, Final, Iterator

import chromadb
import httpx
from fastapi import FastAPI, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

import clustering
import embeddings
import semantic_scholar
import synthesis
from models import (
    AnalyzeRequest,
    AnalyzeResponse,
    HealthResponse,
    PaperSummary,
    QuestionRequest,
    QuestionResponse,
)

# Sized so a run lands near the 15-40 paper corpora the clustering defaults assume.
_RELATED_LIMIT: Final = 20
_CITATION_LIMIT: Final = 15

# References come back in bibliography order and citations newest-first, neither
# of which tracks importance, so a wide pool is fetched and then ranked by
# citation count down to _CITATION_LIMIT. Widening the pool costs response size
# but no extra requests, so it does not eat further into the rate limit.
_CITATION_FETCH_LIMIT: Final = 100

# Hard ceiling after deduplication. Every extra paper costs an embedding pass and
# inflates the prompts sent to Gemini.
_MAX_CORPUS_PAPERS: Final = 60

_MIN_CLUSTER_SIZE: Final = 2
_ANSWER_TOP_K: Final = 10

_HTTP_TIMEOUT: Final = httpx.Timeout(30.0)

# Streamlit runs on its own port, so origins are matched by pattern rather than
# listed; a bare wildcard is not allowed alongside allow_credentials.
_LOCALHOST_ORIGIN_PATTERN: Final = r"http://(localhost|127\.0\.0\.1)(:\d+)?"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
)
logger = logging.getLogger("scout")

app = FastAPI(title="Research Paper Scout", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=_LOCALHOST_ORIGIN_PATTERN,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# sessionId -> the session's in-memory Chroma collection. Guarded by a lock
# because FastAPI serves requests from a thread pool.
_sessions: Final[dict[str, chromadb.Collection]] = {}
_sessions_lock: Final = threading.Lock()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report that the process is up.

    Does not touch the embedding model, Gemini or Semantic Scholar, so a healthy
    response says nothing about whether those are reachable.
    """
    return HealthResponse(status="ok")


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """Build a themed picture of the literature around one seed paper.

    Runs phases 1 to 3 in order: resolve the input to a seed paper, fetch that
    paper's recommendations, references and citations, deduplicate them, fill in
    the TL;DRs the edge endpoints cannot return, embed and index them into a
    fresh session collection, cluster the vectors, name each theme, and write a
    landscape briefing.

    The input may be an arXiv URL or ID, or free text such as a title or topic,
    in which case the best relevance match becomes the seed.

    The session ID in the response is the handle for asking follow-up questions
    via ``POST /ask``.

    Raises:
        HTTPException:
            - 422 if the input is neither an arXiv identifier nor text that
              matches any paper.
            - 404 if Semantic Scholar has no such paper.
            - 429 if Semantic Scholar rate-limited every retry.
            - 502 if Semantic Scholar or Gemini failed, or nothing was indexable.
            - 503 if no Gemini API key is configured.
    """
    papers = await _fetch_corpus(request.paper_input)

    session_id = str(uuid.uuid4())
    collection = await run_in_threadpool(embeddings.create_session_collection, session_id)
    indexed_count = await run_in_threadpool(embeddings.index_papers, collection, papers)
    logger.info(
        "analyze: indexed %d/%d papers into session %s",
        indexed_count,
        len(papers),
        session_id,
    )

    if not indexed_count:
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            "None of the papers fetched for this input had a title or abstract to "
            "embed, so there is nothing to analyse.",
        )

    indexed_papers, vectors = _read_indexed(collection)
    labels = await run_in_threadpool(
        clustering.cluster_papers, vectors, _MIN_CLUSTER_SIZE
    )

    grouped = _group_by_cluster(indexed_papers, labels)
    theme_ids = [cid for cid in sorted(grouped) if cid != clustering.NOISE_LABEL]
    outlier_count = len(grouped.get(clustering.NOISE_LABEL, []))
    logger.info(
        "analyze: clustered %d papers into %d themes (%d outliers)",
        len(indexed_papers),
        len(theme_ids),
        outlier_count,
    )

    with _as_http_error():
        # One call for every theme, then one for the briefing: two Gemini
        # requests per analysis regardless of how many themes were found.
        theme_labels = await run_in_threadpool(clustering.label_clusters, grouped)
        for cluster_id in theme_ids:
            logger.info(
                "analyze: theme %d (%d papers) -> %r",
                cluster_id,
                len(grouped[cluster_id]),
                theme_labels.get(cluster_id),
            )

        briefing = await run_in_threadpool(
            clustering.generate_landscape_summary, grouped, theme_labels
        )
    logger.info("analyze: briefing generated (%d characters)", len(briefing))

    with _sessions_lock:
        _sessions[session_id] = collection

    return AnalyzeResponse(
        sessionId=session_id,
        papers=[
            _to_summary(paper, theme_labels.get(label))
            for paper, label in zip(indexed_papers, labels)
        ],
        landscapeBriefing=briefing,
        clusterCount=len(theme_ids),
    )


@app.post("/ask", response_model=QuestionResponse)
async def ask(request: QuestionRequest) -> QuestionResponse:
    """Answer a question using only the papers indexed in one session.

    Retrieves the closest papers from the session's collection and hands just
    those to Gemini, so the answer is grounded in the corpus rather than in the
    model's own knowledge. An unrecognised question simply retrieves the nearest
    papers anyway; the model is instructed to say when they do not cover it.

    Any ``history`` on the request is passed through to the generation step so
    follow-ups can refer back to earlier turns. Retrieval deliberately ignores
    it and matches on the question's own text, so each question is answered from
    the papers that fit it rather than the papers that fit the conversation.

    Raises:
        HTTPException:
            - 404 if the session ID is unknown, which includes any session
              created before the last server restart.
            - 400 if the session's collection cannot be queried.
            - 502 if Gemini failed.
            - 503 if no Gemini API key is configured.
    """
    collection = _lookup_session(request.sessionId)

    try:
        matches = await run_in_threadpool(
            embeddings.query_collection, collection, request.question, _ANSWER_TOP_K
        )
    except ValueError as exc:
        logger.warning("ask: rejected query for session %s: %s", request.sessionId, exc)
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc

    logger.info(
        "ask: session %s retrieved %d papers for %r (%d history turns supplied)",
        request.sessionId,
        len(matches),
        request.question,
        len(request.history),
    )

    with _as_http_error():
        answer = await run_in_threadpool(
            synthesis.answer_question,
            request.question,
            matches,
            history=[turn.model_dump() for turn in request.history],
        )

    return QuestionResponse(
        answer=answer,
        sourcePapers=[title for match in matches if (title := match.get("title"))],
    )


async def _resolve_seed(
    paper_input: str, *, client: httpx.AsyncClient
) -> dict[str, Any]:
    """Resolve user input to a seed paper, by identifier first and text second.

    An arXiv URL or ID is looked up directly. Anything else — a title, a topic,
    a pasted abstract — is handed to relevance search.
    """
    try:
        paper_id = semantic_scholar.parse_paper_id(paper_input)
    except ValueError as exc:
        logger.info(
            "analyze: %r is not an arXiv identifier (%s), searching by text",
            paper_input,
            exc,
        )
        return await _search_for_seed(paper_input, client=client)

    seed = await semantic_scholar.fetch_paper_details(paper_id, client=client)
    # The rest of the pipeline addresses the seed by its S2 hash, which is what
    # search results carry too; the ARXIV: form also works as an ID, so it
    # stands in on the rare record that has no hash.
    seed["paperId"] = seed.get("paperId") or paper_id
    logger.info("analyze: resolved %r to %s", paper_input, seed["paperId"])
    return seed


async def _search_for_seed(
    query: str, *, client: httpx.AsyncClient
) -> dict[str, Any]:
    """Seed from the best text match, or fail with a 422 if there is none."""
    matches = await semantic_scholar.search_papers_by_text(query, client=client)
    seed = next((paper for paper in matches if paper.get("paperId")), None)

    if seed is None:
        logger.warning("analyze: text search found nothing for %r", query)
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "That input is not an arXiv identifier, and searching Semantic "
            "Scholar for it matched no papers. Try an arXiv URL such as "
            "'https://arxiv.org/abs/1706.03762', or a shorter and more specific "
            "phrase — long passages match poorly.",
        )

    logger.info(
        "analyze: text search matched %d papers, seeding with %r (%s)",
        len(matches),
        seed.get("title"),
        seed["paperId"],
    )
    return seed


async def _fetch_corpus(paper_input: str) -> list[dict[str, Any]]:
    """Fetch the seed paper and its neighbourhood, deduplicated and capped.

    All five requests share one HTTP client. They run in sequence because the
    Semantic Scholar rate limit is cumulative across endpoints, so nothing is
    gained by overlapping them; the required gap between requests is enforced
    inside :mod:`semantic_scholar` rather than here.
    """
    with _as_http_error():
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
            seed = await _resolve_seed(paper_input, client=client)
            seed_id = seed["paperId"]
            related = await semantic_scholar.fetch_related_papers(
                seed_id, _RELATED_LIMIT, client=client
            )
            neighbourhood = await semantic_scholar.fetch_references_and_citations(
                seed_id, limit=_CITATION_FETCH_LIMIT, client=client
            )

            references = _most_cited(neighbourhood["references"])
            citations = _most_cited(neighbourhood["citations"])
            logger.info(
                "analyze: fetched seed + %d related + %d/%d references + "
                "%d/%d citations (most cited kept)",
                len(related),
                len(references),
                len(neighbourhood["references"]),
                len(citations),
                len(neighbourhood["citations"]),
            )

            combined = [seed, *related, *references, *citations]
            unique = _deduplicate(combined)
            logger.info(
                "analyze: %d papers fetched, %d unique after deduplication",
                len(combined),
                len(unique),
            )

            if len(unique) > _MAX_CORPUS_PAPERS:
                logger.info(
                    "analyze: trimming corpus from %d to %d papers",
                    len(unique),
                    _MAX_CORPUS_PAPERS,
                )
                unique = unique[:_MAX_CORPUS_PAPERS]

            # Runs after the trim so no lookup is spent on a paper that was just
            # dropped, and before indexing so the TL;DRs reach Chroma's metadata.
            # References and citations arrive without one, so for most of the
            # corpus this is the only chance to get it.
            await _backfill_tldrs(unique, client=client)

    return unique


async def _backfill_tldrs(
    papers: list[dict[str, Any]], *, client: httpx.AsyncClient
) -> None:
    """Fill in missing TL;DRs and log how many were recovered."""
    missing_before = sum(1 for paper in papers if paper.get("tldr") is None)
    if not missing_before:
        logger.info("analyze: every paper already has a TL;DR, skipping backfill")
        return

    await semantic_scholar.backfill_tldrs(papers, client=client)

    still_missing = sum(1 for paper in papers if paper.get("tldr") is None)
    logger.info(
        "analyze: backfilled %d of %d missing TL;DRs (%d have none on record)",
        missing_before - still_missing,
        missing_before,
        still_missing,
    )


def _most_cited(
    papers: list[dict[str, Any]], limit: int = _CITATION_LIMIT
) -> list[dict[str, Any]]:
    """Keep the ``limit`` most-cited papers, best first.

    Sorting before trimming is what makes a wide fetch worthwhile: it turns an
    arbitrary slice of a heavily cited paper's neighbourhood into its most
    influential corner. Papers with no citation count sort last.
    """
    ranked = sorted(
        papers, key=lambda paper: paper.get("citationCount") or 0, reverse=True
    )
    return ranked[:limit]


def _deduplicate(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop repeats, keeping first appearance so the seed paper stays first.

    References and citations overlap often, and a recommendation is frequently
    also a reference. Papers without an ID fall back to their title.
    """
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []

    for paper in papers:
        key = paper.get("paperId") or (paper.get("title") or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(paper)

    return unique


def _read_indexed(
    collection: chromadb.Collection,
) -> tuple[list[dict[str, Any]], list[list[float]]]:
    """Read back what was indexed, so clustering sees exactly those papers.

    Reusing the stored vectors avoids embedding everything a second time and
    keeps the papers aligned with their vectors by construction. The stored
    document is the text that was embedded — the abstract, or the title when the
    abstract was missing — so it stands in for the abstract downstream.
    """
    record = collection.get(include=["embeddings", "documents", "metadatas"])

    ids = list(record.get("ids") or [])
    metadatas = list(record.get("metadatas") or [])
    documents = list(record.get("documents") or [])

    # Chroma hands embeddings back as a numpy array, so this cannot use `or []`
    # as the other fields do: that would evaluate the array's truth value.
    raw_vectors = record.get("embeddings")
    if raw_vectors is None:
        raw_vectors = []

    papers: list[dict[str, Any]] = []
    for position, record_id in enumerate(ids):
        metadata = metadatas[position] if position < len(metadatas) else None
        metadata = metadata or {}
        papers.append(
            {
                "paperId": metadata.get("paperId") or record_id,
                "title": metadata.get("title"),
                "year": metadata.get("year"),
                "citationCount": metadata.get("citationCount"),
                "tldr": metadata.get("tldr"),
                "abstract": documents[position] if position < len(documents) else None,
            }
        )

    vectors = [[float(value) for value in row] for row in raw_vectors]
    return papers, vectors


def _group_by_cluster(
    papers: list[dict[str, Any]], labels: list[int]
) -> dict[int, list[dict[str, Any]]]:
    """Bucket papers by cluster label, preserving order within each bucket."""
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for paper, label in zip(papers, labels):
        grouped[label].append(paper)
    return dict(grouped)


def _to_summary(paper: dict[str, Any], cluster_label: str | None) -> PaperSummary:
    """Project an internal paper dict onto the client-facing schema."""
    return PaperSummary(
        paperId=str(paper["paperId"]),
        title=paper.get("title"),
        year=paper.get("year"),
        citationCount=paper.get("citationCount"),
        tldr=paper.get("tldr"),
        clusterLabel=cluster_label,
    )


def _lookup_session(session_id: str) -> chromadb.Collection:
    """Return a session's collection, or raise a 404 explaining why it is gone."""
    with _sessions_lock:
        collection = _sessions.get(session_id)

    if collection is None:
        logger.warning("ask: unknown session %s", session_id)
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"No session {session_id!r}. Sessions are held in memory only, so it "
            "may have been lost to a server restart. Call POST /analyze again to "
            "start a new one.",
        )

    return collection


@contextlib.contextmanager
def _as_http_error() -> Iterator[None]:
    """Translate phase-module failures into HTTP responses.

    Keeps the mapping in one place and stops upstream tracebacks reaching the
    client, while still logging them server-side. Subclasses are listed before
    their base classes so the specific handler wins.
    """
    try:
        yield
    except semantic_scholar.PaperNotFoundError as exc:
        logger.warning("upstream: paper not found (%s)", exc)
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            "Semantic Scholar has no record of that paper. Very recent arXiv "
            "submissions can take a while to be indexed.",
        ) from exc
    except semantic_scholar.RateLimitError as exc:
        logger.warning("upstream: rate limited (%s)", exc)
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "Semantic Scholar is rate-limiting this server. Wait a moment and try "
            "again, or set SEMANTIC_SCHOLAR_API_KEY to raise the limit.",
        ) from exc
    except semantic_scholar.SemanticScholarError as exc:
        logger.exception("upstream: Semantic Scholar failed")
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            f"Could not reach Semantic Scholar: {exc}",
        ) from exc
    except synthesis.MissingAPIKeyError as exc:
        logger.error("config: %s", exc)
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "This server has no Gemini API key configured, so it cannot label "
            "themes or answer questions. Set GEMINI_API_KEY in backend/.env.",
        ) from exc
    except synthesis.SynthesisError as exc:
        logger.exception("upstream: Gemini failed")
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            f"The language model call failed: {exc}",
        ) from exc
