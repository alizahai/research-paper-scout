"""Phase 2: generating embeddings and storing them in ChromaDB.

Papers are embedded with a locally-run SentenceTransformer and held in an
ephemeral, in-memory Chroma collection scoped to one browsing session. Nothing
touches disk, so a session's index disappears when the process exits.

``sentence_transformers`` is imported lazily inside :func:`get_embedding_model`
rather than at module scope, because it pulls in torch and costs seconds of
import time. That keeps FastAPI startup fast and lets this module be imported
(and tested) without the heavy dependency present.
"""

from __future__ import annotations

import hashlib
import re
import threading
from typing import TYPE_CHECKING, Any, Final, Sequence

import chromadb

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

__all__ = [
    "EMBEDDING_MODEL_NAME",
    "create_session_collection",
    "get_embedding_model",
    "index_papers",
    "query_collection",
]

EMBEDDING_MODEL_NAME: Final = "all-MiniLM-L6-v2"

# Metadata copied from each paper dict onto its Chroma record. These ride
# alongside the vector and are returned with matches; they are not embedded.
METADATA_FIELDS: Final[tuple[str, ...]] = (
    "paperId",
    "title",
    "year",
    "citationCount",
    "tldr",
)

_COLLECTION_NAME_PREFIX: Final = "session-"
_MAX_COLLECTION_NAME_LENGTH: Final = 512
_INVALID_NAME_CHARS: Final = re.compile(r"[^a-zA-Z0-9_-]")

_model: SentenceTransformer | None = None
_model_lock: Final = threading.Lock()


def get_embedding_model() -> SentenceTransformer:
    """Return the shared SentenceTransformer, loading it on first use.

    The model is a process-wide singleton: it is roughly 90 MB and takes a few
    seconds to initialise, so reloading it per request would dominate latency.
    Loading is guarded by a lock because FastAPI runs sync endpoints in a thread
    pool, where concurrent first calls would otherwise each load a copy.

    Returns:
        The cached :class:`~sentence_transformers.SentenceTransformer` for
        :data:`EMBEDDING_MODEL_NAME`.
    """
    global _model

    if _model is None:
        with _model_lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer

                _model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    return _model


def create_session_collection(session_id: str) -> chromadb.Collection:
    """Create (or reopen) an in-memory Chroma collection for one session.

    Uses ``chromadb.Client()``, whose default settings are ephemeral, so the
    index lives in memory only. Chroma caches its in-memory system per settings
    object, so repeated calls in the same process reach the same store — which is
    what lets one request index a session and a later request query it.

    Args:
        session_id: Opaque session identifier. It is sanitised into a legal
            Chroma collection name, so a UUID, hash or short slug all work.

    Returns:
        The collection for this session, configured for cosine distance and
        empty unless it was already populated in this process. Note that Chroma
        ignores ``configuration`` when the collection already exists, so a
        collection first created elsewhere keeps whatever space it was made with.

    Raises:
        ValueError: If ``session_id`` is empty or sanitises away to nothing.
    """
    client = chromadb.Client()
    return client.get_or_create_collection(
        name=_collection_name(session_id),
        # MiniLM embeddings are compared by angle, not magnitude, and Chroma's
        # default space is L2: for unit vectors that puts distances on a 0-4
        # scale instead of the 0-2 cosine scale query_collection assumes.
        configuration={"hnsw": {"space": "cosine"}},
    )


def index_papers(
    collection: chromadb.Collection, papers: list[dict[str, Any]]
) -> int:
    """Embed papers and upsert them into ``collection``.

    Each paper is embedded from its abstract, falling back to its title when the
    abstract is missing — common for older or paywalled records. Papers with
    neither are skipped, since there is nothing to embed. Duplicates are collapsed
    by paper ID, because reference and citation lists routinely overlap and Chroma
    rejects a batch containing repeated IDs.

    Args:
        collection: Target collection, typically from
            :func:`create_session_collection`.
        papers: Paper dicts as returned by ``semantic_scholar``, each with
            ``paperId``, ``title``, ``abstract``, ``year``, ``citationCount`` and
            ``tldr``.

    Returns:
        The number of papers actually written, which may be less than
        ``len(papers)`` after skips and deduplication.
    """
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, str | int | float | bool]] = []
    seen: set[str] = set()

    for paper in papers:
        text = _embeddable_text(paper)
        if text is None:
            continue

        paper_key = paper.get("paperId") or _fallback_id(text)
        if paper_key in seen:
            continue
        seen.add(paper_key)

        ids.append(paper_key)
        documents.append(text)
        metadatas.append(_build_metadata(paper, paper_key))

    if not ids:
        return 0

    embeddings = _encode(documents)
    # upsert rather than add so re-indexing a session overwrites cleanly instead
    # of depending on how the installed Chroma treats an already-present ID.
    collection.upsert(
        ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
    )
    return len(ids)


def query_collection(
    collection: chromadb.Collection, query_text: str, top_k: int = 5
) -> list[dict[str, Any]]:
    """Return the ``top_k`` papers most similar to ``query_text``.

    Args:
        collection: Collection to search.
        query_text: Natural-language query, embedded with the same model as the
            indexed papers.
        top_k: Maximum number of matches to return.

    Returns:
        Matches ordered most similar first. Each dict carries the
        :data:`METADATA_FIELDS` (with ``None`` for values the paper lacked), the
        embedded ``document`` text, the raw Chroma ``distance``, and a
        ``similarity`` of ``1 - distance``. Under cosine distance similarity runs
        from 1.0 (identical direction) down to -1.0 (opposite), so treat it as a
        ranking signal rather than a probability. Returns an empty list when the
        collection holds nothing.

    Raises:
        ValueError: If ``query_text`` is blank, ``top_k`` is not positive, or the
            collection was built with a distance space other than cosine, which
            would make the returned ``similarity`` meaningless.
    """
    if not query_text or not query_text.strip():
        raise ValueError("query_text must be a non-empty string")
    if top_k < 1:
        raise ValueError(f"top_k must be at least 1, got {top_k}")

    _require_cosine_space(collection)

    # Chroma clamps n_results to the collection size itself, but asking for no
    # more than exists keeps the intent obvious and skips the empty case.
    available = collection.count()
    if available == 0:
        return []

    response = collection.query(
        query_embeddings=_encode([query_text.strip()]),
        n_results=min(top_k, available),
        include=["metadatas", "documents", "distances"],
    )
    return _unpack_matches(response)


def _require_cosine_space(collection: chromadb.Collection) -> None:
    """Reject collections whose distance space breaks the similarity contract.

    ``similarity = 1 - distance`` only holds for cosine. An L2 collection returns
    squared distances on a 0-4 scale, which would silently yield similarities
    below -1 rather than an obvious error. The space is left unchecked when Chroma
    does not report one, since that varies by index type and version.
    """
    configuration = getattr(collection, "configuration", None) or {}
    space = (configuration.get("hnsw") or {}).get("space")

    if space is not None and space != "cosine":
        raise ValueError(
            f"collection {collection.name!r} uses {space!r} distance, but "
            "query_collection reports cosine similarity. Recreate it with "
            "create_session_collection, which configures cosine."
        )


def _unpack_matches(response: Any) -> list[dict[str, Any]]:
    """Flatten Chroma's per-query nested lists into one list of matches.

    ``query`` returns each field as a list of results *per query embedding*; only
    one query is ever sent here, so the first row is taken.
    """

    def first_row(key: str) -> list[Any]:
        rows = response.get(key) or []
        return list(rows[0]) if rows and rows[0] is not None else []

    documents = first_row("documents")
    distances = first_row("distances")
    metadatas = first_row("metadatas")

    matches: list[dict[str, Any]] = []
    for index, metadata in enumerate(metadatas):
        distance = distances[index] if index < len(distances) else None
        match: dict[str, Any] = {
            field: (metadata or {}).get(field) for field in METADATA_FIELDS
        }
        match["document"] = documents[index] if index < len(documents) else None
        match["distance"] = distance
        match["similarity"] = None if distance is None else 1.0 - distance
        matches.append(match)

    return matches


def _encode(texts: Sequence[str]) -> list[list[float]]:
    """Embed ``texts`` in one batch and return plain lists for Chroma.

    Normalising here is what makes cosine distance meaningful. The texts are
    passed positionally on purpose: sentence-transformers renamed that parameter
    from ``sentences`` to ``inputs`` in 6.0, but its position never moved.
    """
    vectors = get_embedding_model().encode(
        list(texts), normalize_embeddings=True, convert_to_numpy=True
    )
    return [vector.tolist() for vector in vectors]


def _embeddable_text(paper: dict[str, Any]) -> str | None:
    """Pick the text to embed: abstract if present, else title, else nothing."""
    for key in ("abstract", "title"):
        value = paper.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _build_metadata(
    paper: dict[str, Any], paper_key: str
) -> dict[str, str | int | float | bool]:
    """Copy :data:`METADATA_FIELDS` across, keeping only storable scalar values.

    ``None`` and empty-string entries — a missing ``tldr`` or ``year``, say — are
    omitted rather than stored, and :func:`query_collection` restores them as
    ``None`` on the way out. ``paperId`` is always written last, which also keeps
    the dict non-empty; Chroma rejects a record whose metadata dict has no keys.
    """
    metadata: dict[str, str | int | float | bool] = {}

    for field in METADATA_FIELDS:
        value = paper.get(field)
        if isinstance(value, (str, int, float, bool)) and value != "":
            metadata[field] = value

    metadata["paperId"] = paper_key
    return metadata


def _fallback_id(text: str) -> str:
    """Derive a stable ID for a paper the API returned without a ``paperId``."""
    return f"sha1-{hashlib.sha1(text.encode('utf-8')).hexdigest()}"


def _collection_name(session_id: str) -> str:
    """Turn a session ID into a name Chroma will accept.

    Chroma requires 3-512 characters, alphanumeric at both ends, and only
    alphanumerics, underscores and hyphens between. The prefix guarantees a legal
    start and minimum length whatever the caller passes in.
    """
    if not session_id or not session_id.strip():
        raise ValueError("session_id must be a non-empty string")

    slug = _INVALID_NAME_CHARS.sub("-", session_id.strip())
    slug = slug.strip("-_")
    if not slug:
        raise ValueError(
            f"session_id {session_id!r} contains no usable characters for a "
            "collection name"
        )

    name = f"{_COLLECTION_NAME_PREFIX}{slug}"[:_MAX_COLLECTION_NAME_LENGTH]
    return name.rstrip("-_")
