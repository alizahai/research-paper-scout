"""Phase 1: fetching papers from the Semantic Scholar API.

Endpoints used, all verified against the live API:

- ``GET /graph/v1/paper/{id}`` — single paper metadata, returned bare.
- ``GET /graph/v1/paper/search`` — relevance search over free text, as
  ``{"total": N, "offset": 0, "next": N, "data": [{...}]}``, with ``data``
  omitted altogether when nothing matched.
- ``POST /graph/v1/paper/batch`` — many papers at once, as a bare array aligned
  with the submitted IDs, using a ``null`` element for an ID it cannot match.
- ``GET /graph/v1/paper/{id}/references`` — papers this one cites, as
  ``{"data": [{"citedPaper": {...}}]}``.
- ``GET /graph/v1/paper/{id}/citations`` — papers citing this one, as
  ``{"data": [{"citingPaper": {...}}]}``.
- ``GET /recommendations/v1/papers/forpaper/{id}`` — related papers, as
  ``{"recommendedPapers": [...]}``.

Every endpoint wraps its papers differently, but the paper objects inside are
identically shaped, so :func:`_normalize_paper` handles all of them and each
fetch function only has to know where to find the list.

Two API quirks drive the rest of the module. First, ``tldr`` is unavailable on
the edge endpoints: ``/references`` and ``/citations`` reject it with HTTP 400, so
they are queried with :data:`CITATION_EDGE_FIELDS` and their results always
arrive with ``tldr=None`` — :func:`backfill_tldrs` exists to fill those in
afterwards, since ``/paper/batch`` does support the field. Second, the rate limit
is cumulative across every endpoint
rather than per endpoint — one request per second with a key, and something
stricter and burstier without one — and no ``Retry-After`` header is sent. Two
things follow from that: every request passes through :data:`_PACER`, which
spaces requests process-wide, and :func:`_request` retries what still comes back
as a 429 with exponential backoff. :data:`API_KEY_ENV_VAR` is sent as an
``x-api-key`` header when set, which raises the limit substantially.

Each function accepts an optional ``client`` so callers can reuse one
``httpx.AsyncClient`` across several calls; when omitted, a short-lived client is
created for the duration of the call.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Final, Sequence
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv

__all__ = [
    "PaperNotFoundError",
    "RateLimitError",
    "SemanticScholarError",
    "backfill_tldrs",
    "fetch_paper_details",
    "fetch_references_and_citations",
    "fetch_related_papers",
    "parse_paper_id",
    "search_papers_by_text",
]

logger = logging.getLogger(__name__)

GRAPH_API_BASE: Final = "https://api.semanticscholar.org/graph/v1"
RECOMMENDATIONS_API_BASE: Final = "https://api.semanticscholar.org/recommendations/v1"

API_KEY_ENV_VAR: Final = "SEMANTIC_SCHOLAR_API_KEY"

# Resolved from this file rather than the working directory, so the key is found
# whether uvicorn is started from the repo root or from backend/.
_ENV_PATH: Final = Path(__file__).resolve().parent / ".env"

PAPER_FIELDS: Final[tuple[str, ...]] = (
    "paperId",
    "title",
    "abstract",
    "year",
    "citationCount",
    "tldr",
    "authors",
)
CITATION_EDGE_FIELDS: Final[tuple[str, ...]] = tuple(
    field for field in PAPER_FIELDS if field != "tldr"
)

# Relevance search caps its page size lower than the other endpoints: 100 is
# accepted, 101 is rejected.
SEARCH_MAX_LIMIT: Final = 100

# Most IDs /paper/batch accepts in one request.
BATCH_MAX_IDS: Final = 500

# Retries are sized for the unauthenticated limit: worst case is roughly 30s of
# waiting before giving up. Lower _MAX_ATTEMPTS if you would rather fail fast.
_MAX_ATTEMPTS: Final = 5
_INITIAL_BACKOFF_SECONDS: Final = 2.0
_MAX_BACKOFF_SECONDS: Final = 30.0
_TIMEOUT: Final = httpx.Timeout(30.0)

# A key buys one request per second, so requests are spaced slightly wider than
# that to leave room for clock skew and for the server's own accounting.
_MIN_REQUEST_INTERVAL_SECONDS: Final = 1.1

# https://arxiv.org/help/arxiv_identifier
_ARXIV_MODERN_ID: Final = re.compile(r"(\d{4}\.\d{4,5})(?:v\d+)?")
_ARXIV_LEGACY_ID: Final = re.compile(
    r"([a-z][a-z\-]*(?:\.[a-z]{2})?/\d{7})(?:v\d+)?", re.IGNORECASE
)
_ARXIV_URL_PATH: Final = re.compile(r"^/(?:abs|pdf)/(.+)$", re.IGNORECASE)
_UNSUPPORTED_FIELDS: Final = re.compile(
    r"unsupported fields:\s*\[([^\]]*)\]", re.IGNORECASE
)


class SemanticScholarError(Exception):
    """Base class for Semantic Scholar failures.

    Attributes:
        status_code: HTTP status that triggered the error, when there was one.
        body: Response body, truncated, for error statuses that returned one.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        body: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class PaperNotFoundError(SemanticScholarError):
    """Raised when the API has no record of the requested paper (HTTP 404)."""


class RateLimitError(SemanticScholarError):
    """Raised when rate limiting (HTTP 429) persists across every retry."""


def parse_paper_id(input_str: str) -> str:
    """Normalize an arXiv URL or bare arXiv ID into a Semantic Scholar paper ID.

    Accepts abs/pdf URLs, an ``arXiv:`` prefixed ID, or a bare ID, in either the
    modern (``2310.12345``) or legacy (``hep-th/9901001``) identifier scheme. Any
    version suffix is dropped, since Semantic Scholar indexes papers rather than
    individual versions.

    Args:
        input_str: For example ``https://arxiv.org/abs/2310.12345v2``,
            ``arXiv:2310.12345``, or ``2310.12345``.

    Returns:
        An ID of the form ``ARXIV:2310.12345``, ready to interpolate into an API
        path.

    Raises:
        ValueError: If no arXiv identifier can be recovered from the input.
    """
    if not input_str or not input_str.strip():
        raise ValueError("paper identifier must be a non-empty string")

    candidate = input_str.strip()

    if candidate.upper().startswith("ARXIV:"):
        candidate = candidate.split(":", 1)[1].strip()

    if "arxiv.org" in candidate.lower():
        url = candidate if "://" in candidate else f"https://{candidate}"
        path_match = _ARXIV_URL_PATH.match(urlparse(url).path)
        if path_match is None:
            raise ValueError(
                f"{input_str!r} is an arXiv URL but has no /abs/ or /pdf/ path"
            )
        candidate = re.sub(r"\.pdf$", "", path_match.group(1), flags=re.IGNORECASE)
        candidate = candidate.rstrip("/")

    match = _ARXIV_MODERN_ID.fullmatch(candidate) or _ARXIV_LEGACY_ID.fullmatch(
        candidate
    )
    if match is None:
        raise ValueError(f"could not parse an arXiv identifier from {input_str!r}")

    return f"ARXIV:{match.group(1)}"


async def fetch_paper_details(
    paper_id: str, *, client: httpx.AsyncClient | None = None
) -> dict[str, Any]:
    """Fetch metadata for a single paper.

    Args:
        paper_id: A Semantic Scholar paper ID, such as the ``ARXIV:2310.12345``
            produced by :func:`parse_paper_id`, or a bare S2 paper hash.
        client: Optional client to reuse.

    Returns:
        A dict with ``paperId``, ``title``, ``abstract``, ``year``,
        ``citationCount``, ``tldr`` and ``authors``. ``tldr`` is flattened from
        the API's ``{"model": ..., "text": ...}`` object down to the text string.
        ``authors`` is a list of ``{"authorId", "name"}`` dicts. Any field may be
        ``None`` when Semantic Scholar has no value for it, which is common for
        ``abstract`` and ``tldr`` on older or non-open-access papers.

    Raises:
        PaperNotFoundError: If the paper is not indexed.
        RateLimitError: If rate limiting outlasts the retries.
        SemanticScholarError: On any other API or transport failure.
    """
    payload = await _fetch_with_fields(
        f"{GRAPH_API_BASE}/paper/{paper_id}", PAPER_FIELDS, client=client
    )
    return _normalize_paper(payload)


async def search_papers_by_text(
    query: str, limit: int = 10, *, client: httpx.AsyncClient | None = None
) -> list[dict[str, Any]]:
    """Find papers matching free text, most relevant first.

    Use this when there is no identifier to work from — a title, a topic, or a
    pasted abstract. The API matches the query against titles and abstracts, and
    relevance degrades as the query gets longer, so a title or a phrase works
    better than several paragraphs.

    Args:
        query: Non-empty search text.
        limit: How many results to request, from 1 to :data:`SEARCH_MAX_LIMIT`.
        client: Optional client to reuse.

    Returns:
        A list of paper dicts shaped exactly like :func:`fetch_paper_details`
        results, ordered by the API's relevance ranking. Empty when nothing
        matched, which is an ordinary answer rather than an error.

    Raises:
        ValueError: If ``query`` is blank or ``limit`` is out of range.
        RateLimitError: If rate limiting outlasts the retries.
        SemanticScholarError: On any other API or transport failure.
    """
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")
    if not 1 <= limit <= SEARCH_MAX_LIMIT:
        raise ValueError(f"limit must be between 1 and {SEARCH_MAX_LIMIT}, got {limit}")

    # Unlike /paper/{id}, this endpoint wraps its results in a paged envelope;
    # the paper objects inside it carry the same fields. A zero-match search
    # answers 200 with `data` omitted entirely rather than empty, hence `or []`.
    payload = await _fetch_with_fields(
        f"{GRAPH_API_BASE}/paper/search",
        PAPER_FIELDS,
        {"query": query.strip(), "limit": limit},
        client=client,
    )
    return [_normalize_paper(paper) for paper in payload.get("data") or [] if paper]


async def fetch_related_papers(
    paper_id: str, limit: int = 20, *, client: httpx.AsyncClient | None = None
) -> list[dict[str, Any]]:
    """Fetch semantically related papers via the recommendations API.

    Note that this endpoint is far less reliable than the graph endpoints: it
    answers 200 with an empty ``recommendedPapers`` list for many seed papers,
    including heavily cited ones, so treat an empty result as normal and fall
    back to :func:`fetch_references_and_citations` for discovery.

    Args:
        paper_id: Seed paper to find neighbours for.
        limit: How many recommendations to request, from 1 to 500.
        client: Optional client to reuse.

    Returns:
        A list of paper dicts shaped like :func:`fetch_paper_details` results,
        ordered by the API's relevance ranking, and possibly empty.

    Raises:
        ValueError: If ``limit`` is outside the range the API accepts.
        PaperNotFoundError: If the seed paper is not indexed.
        RateLimitError: If rate limiting outlasts the retries.
        SemanticScholarError: On any other API or transport failure.
    """
    if not 1 <= limit <= 500:
        raise ValueError(f"limit must be between 1 and 500, got {limit}")

    payload = await _fetch_with_fields(
        f"{RECOMMENDATIONS_API_BASE}/papers/forpaper/{paper_id}",
        PAPER_FIELDS,
        {"limit": limit},
        client=client,
    )
    return [
        _normalize_paper(paper)
        for paper in payload.get("recommendedPapers") or []
        if paper
    ]


async def fetch_references_and_citations(
    paper_id: str, *, limit: int = 100, client: httpx.AsyncClient | None = None
) -> dict[str, list[dict[str, Any]]]:
    """Fetch the papers a paper cites and the papers that cite it.

    The two endpoints are called in sequence rather than concurrently, because
    the rate limit is shared between them; :data:`_PACER` adds the required gap,
    so expect this to take a second or so per direction at minimum.

    Args:
        paper_id: The paper whose neighbourhood to walk.
        limit: Maximum entries per direction, from 1 to 1000. Anything beyond
            this is dropped, and the API returns citations newest-first rather
            than by importance, so for a well-cited paper treat the citation list
            as an arbitrary sample rather than a census.
        client: Optional client to reuse.

    Returns:
        A dict with keys ``references`` (papers this one cites) and ``citations``
        (papers citing this one), each a list of paper dicts shaped like
        :func:`fetch_paper_details` results. ``tldr`` is always ``None`` here,
        because these endpoints reject that field; re-fetch an individual paper
        with :func:`fetch_paper_details` if you need its summary.

    Raises:
        ValueError: If ``limit`` is outside the range the API accepts.
        PaperNotFoundError: If the paper is not indexed.
        RateLimitError: If rate limiting outlasts the retries.
        SemanticScholarError: On any other API or transport failure.
    """
    if not 1 <= limit <= 1000:
        raise ValueError(f"limit must be between 1 and 1000, got {limit}")

    references = await _fetch_with_fields(
        f"{GRAPH_API_BASE}/paper/{paper_id}/references",
        CITATION_EDGE_FIELDS,
        {"limit": limit},
        client=client,
    )
    citations = await _fetch_with_fields(
        f"{GRAPH_API_BASE}/paper/{paper_id}/citations",
        CITATION_EDGE_FIELDS,
        {"limit": limit},
        client=client,
    )

    return {
        "references": _unwrap_citation_edges(references, "citedPaper"),
        "citations": _unwrap_citation_edges(citations, "citingPaper"),
    }


async def backfill_tldrs(
    papers: list[dict[str, Any]], *, client: httpx.AsyncClient | None = None
) -> list[dict[str, Any]]:
    """Fill in missing ``tldr`` values from ``/paper/batch``, in place.

    Papers gathered from ``/references`` and ``/citations`` never have a TL;DR,
    because those endpoints reject the field outright. This asks for the missing
    ones in a single extra request, which ``/paper/batch`` does support.

    Only papers whose ``tldr`` is already ``None`` are looked up, so a corpus
    that is already complete costs no request at all. A paper the API has no
    TL;DR for keeps its ``None``: absence is a real answer here, not a failure.

    Args:
        papers: Paper dicts as produced elsewhere in this module. Mutated in
            place; papers with no ``paperId`` are skipped, since there is nothing
            to look up.
        client: Optional client to reuse.

    Returns:
        The same list, for convenience.

    Raises:
        RateLimitError: If rate limiting outlasts the retries.
        SemanticScholarError: On any other API or transport failure, including a
            response that is not the expected array.
    """
    missing = [
        paper
        for paper in papers
        if paper.get("tldr") is None and paper.get("paperId")
    ]
    if not missing:
        return papers

    # dict.fromkeys keeps first-seen order while dropping repeats, so the
    # positional alignment of the response stays meaningful.
    ids = list(dict.fromkeys(str(paper["paperId"]) for paper in missing))

    found: dict[str, str] = {}
    for start in range(0, len(ids), BATCH_MAX_IDS):
        chunk = ids[start : start + BATCH_MAX_IDS]
        payload = await _request(
            f"{GRAPH_API_BASE}/paper/batch",
            # Deliberately narrow: the corpus is already built from the other
            # endpoints' data, so re-fetching titles or abstracts here would
            # only invite overwriting good values with a second opinion.
            {"fields": "paperId,tldr"},
            method="POST",
            json_body={"ids": chunk},
            client=client,
        )
        found.update(_tldrs_from_batch(payload, chunk))

    for paper in missing:
        text = found.get(str(paper["paperId"]))
        if text:
            paper["tldr"] = text

    return papers


def _tldrs_from_batch(payload: Any, requested_ids: Sequence[str]) -> dict[str, str]:
    """Map IDs to TL;DR text for the batch entries that have one.

    Entries are keyed under both the ID that was requested and the ``paperId``
    that came back, because those differ whenever the request used an alias such
    as ``ARXIV:1706.03762``. Positional keys are only used when the response
    length matches the request, since alignment is the sole thing tying an entry
    to an alias.
    """
    if not isinstance(payload, list):
        raise SemanticScholarError(
            f"/paper/batch returned {type(payload).__name__}, expected a list"
        )

    aligned = len(payload) == len(requested_ids)
    tldrs: dict[str, str] = {}

    for position, entry in enumerate(payload):
        # A null element means the ID matched nothing.
        if not isinstance(entry, dict):
            continue

        raw = entry.get("tldr")
        text = raw.get("text") if isinstance(raw, dict) else raw
        if not isinstance(text, str) or not text.strip():
            continue

        keys = [entry.get("paperId")]
        if aligned:
            keys.append(requested_ids[position])
        for key in keys:
            if key:
                tldrs[str(key)] = text.strip()

    return tldrs


async def _fetch_with_fields(
    url: str,
    fields: Sequence[str],
    extra_params: dict[str, Any] | None = None,
    *,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """GET ``url`` with a ``fields`` parameter, retrying once without bad fields.

    Field support varies by endpoint and is not consistently documented, so a 400
    naming unsupported fields is retried with those fields removed rather than
    failing the whole request.
    """
    params: dict[str, Any] = {"fields": ",".join(fields), **(extra_params or {})}

    try:
        return await _request(url, params, client=client)
    except SemanticScholarError as exc:
        rejected = _rejected_fields(exc)
        remaining = [field for field in fields if field not in rejected]
        if not rejected or not remaining:
            raise
        params["fields"] = ",".join(remaining)
        return await _request(url, params, client=client)


def _rejected_fields(exc: SemanticScholarError) -> frozenset[str]:
    """Extract field names from an ``unsupported fields: [a, b]`` 400 response."""
    if exc.status_code != 400 or not exc.body:
        return frozenset()
    match = _UNSUPPORTED_FIELDS.search(exc.body)
    if match is None:
        return frozenset()
    return frozenset(
        field.strip() for field in match.group(1).split(",") if field.strip()
    )


def _unwrap_citation_edges(payload: dict[str, Any], key: str) -> list[dict[str, Any]]:
    """Pull the nested paper out of each /references or /citations entry.

    Entries whose nested paper is missing are skipped, which happens when the
    cited or citing work is not itself indexed.
    """
    return [
        _normalize_paper(entry[key])
        for entry in payload.get("data") or []
        if entry.get(key)
    ]


def _normalize_paper(paper: dict[str, Any]) -> dict[str, Any]:
    """Reduce a raw paper object to :data:`PAPER_FIELDS`, flattening ``tldr``.

    The API returns extra keys such as ``openAccessPdf`` even when they were not
    requested; dropping them keeps every paper dict in this module identical in
    shape regardless of which endpoint produced it.
    """
    normalized: dict[str, Any] = {field: paper.get(field) for field in PAPER_FIELDS}
    tldr = paper.get("tldr")
    normalized["tldr"] = tldr.get("text") if isinstance(tldr, dict) else tldr
    normalized["authors"] = paper.get("authors") or []
    return normalized


async def _request(
    url: str,
    params: dict[str, Any],
    *,
    method: str = "GET",
    json_body: Any = None,
    client: httpx.AsyncClient | None = None,
) -> Any:
    """Call ``url``, reusing ``client`` when given or creating a temporary one.

    Returns whatever the endpoint decodes to: the graph object endpoints answer
    with a JSON object, ``/paper/batch`` with an array.
    """
    if client is not None:
        return await _request_with_retries(client, url, params, method, json_body)

    async with httpx.AsyncClient(timeout=_TIMEOUT) as owned_client:
        return await _request_with_retries(
            owned_client, url, params, method, json_body
        )


class _RequestPacer:
    """Spaces outbound requests to at most one per ``interval`` seconds.

    The quota is cumulative across every Semantic Scholar endpoint rather than
    per endpoint, so this gate is process-wide: two pipelines running at once
    take turns instead of each getting a full request per second. Only the wait
    is serialized, not the request itself, so the interval measures the gap
    between requests starting.

    The lock is built on first use and rebuilt if a different event loop turns
    up, because an :class:`asyncio.Lock` binds to the loop that first awaits it;
    one created at import time would fail on a second call to ``asyncio.run``.
    """

    def __init__(self, interval: float) -> None:
        self._interval = interval
        self._lock: asyncio.Lock | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._last_start: float | None = None

    async def wait(self) -> None:
        """Block until enough time has passed since the previous request."""
        async with self._gate():
            if self._last_start is not None:
                delay = self._interval - (time.monotonic() - self._last_start)
                if delay > 0:
                    await asyncio.sleep(delay)
            self._last_start = time.monotonic()

    def _gate(self) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        if self._lock is None or self._loop is not loop:
            self._lock = asyncio.Lock()
            self._loop = loop
            self._last_start = None
        return self._lock


_PACER: Final = _RequestPacer(_MIN_REQUEST_INTERVAL_SECONDS)


async def _request_with_retries(
    client: httpx.AsyncClient,
    url: str,
    params: dict[str, Any],
    method: str = "GET",
    json_body: Any = None,
) -> Any:
    """Call ``url``, retrying rate limits, server errors and transport blips.

    Waits on :data:`_PACER` before every attempt, so every endpoint shares the
    one-request-per-second budget without callers having to sleep themselves.

    404s and other 4xx responses are treated as permanent and raise immediately.

    Raises:
        PaperNotFoundError: On HTTP 404.
        RateLimitError: If the final attempt was rate limited.
        SemanticScholarError: On other HTTP errors, transport failures, or an
            unparseable body.
    """
    headers = _auth_headers()
    backoff = _INITIAL_BACKOFF_SECONDS
    last_error: SemanticScholarError | None = None

    for attempt in range(1, _MAX_ATTEMPTS + 1):
        retry_delay: float | None = None

        try:
            await _PACER.wait()
            response = await client.request(
                method, url, params=params, json=json_body, headers=headers
            )
        except httpx.TransportError as exc:
            last_error = SemanticScholarError(f"could not reach {url}: {exc}")
        else:
            status = response.status_code

            if status == 404:
                raise PaperNotFoundError(
                    f"Semantic Scholar has no paper at {url}",
                    status_code=status,
                    body=response.text[:500],
                )

            if status == 429:
                last_error = RateLimitError(
                    f"rate limited by Semantic Scholar after {attempt} attempt(s)",
                    status_code=status,
                    body=response.text[:500],
                )
                retry_delay = _retry_after(response)
            elif status >= 500:
                last_error = SemanticScholarError(
                    f"Semantic Scholar returned {status} for {url}",
                    status_code=status,
                    body=response.text[:500],
                )
            elif status >= 400:
                raise SemanticScholarError(
                    f"Semantic Scholar rejected the request with {status}: "
                    f"{response.text[:200]}",
                    status_code=status,
                    body=response.text[:500],
                )
            else:
                try:
                    return response.json()
                except ValueError as exc:
                    raise SemanticScholarError(
                        f"malformed JSON from {url}: {exc}", status_code=status
                    ) from exc

        if attempt < _MAX_ATTEMPTS:
            # Jitter keeps concurrent callers from retrying in lockstep and
            # re-colliding on the shared unauthenticated quota.
            await asyncio.sleep(retry_delay or backoff * random.uniform(0.8, 1.3))
            backoff = min(backoff * 2, _MAX_BACKOFF_SECONDS)

    raise last_error or SemanticScholarError(f"request to {url} failed")


def _auth_headers() -> dict[str, str]:
    """Return an ``x-api-key`` header when a key is configured, else nothing.

    The endpoints here all work unauthenticated; a key only raises the rate limit.
    """
    api_key = _api_key()
    return {"x-api-key": api_key} if api_key else {}


@lru_cache(maxsize=1)
def _api_key() -> str:
    """Read the API key from the environment or ``backend/.env``, once.

    Caching keeps the missing-key warning to one line per process instead of one
    per request; the trade-off is that adding a key needs a restart to take
    effect.
    """
    load_dotenv(_ENV_PATH)
    api_key = os.getenv(API_KEY_ENV_VAR, "").strip()

    if not api_key:
        logger.warning(
            "%s is not set, so requests go out unauthenticated against a quota "
            "shared with everyone else doing the same. Expect frequent 429s and "
            "slow retries. Keys are free at "
            "https://www.semanticscholar.org/product/api#api-key-form — add one "
            "to %s.",
            API_KEY_ENV_VAR,
            _ENV_PATH,
        )

    return api_key


def _retry_after(response: httpx.Response) -> float | None:
    """Read the ``Retry-After`` header as seconds, ignoring HTTP-date form.

    Semantic Scholar does not currently send this header, so this is a courtesy
    for if that changes.
    """
    raw = response.headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        return None
