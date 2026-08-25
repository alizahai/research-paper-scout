"""Phase 3: clustering paper embeddings and labeling the resulting themes.

Clustering uses ``sklearn.cluster.HDBSCAN`` rather than the standalone
``hdbscan`` package: scikit-learn is already a dependency and the two have
different constructor arguments, so mixing them invites confusion. Only
``min_cluster_size``, ``min_samples``, ``metric`` and ``copy`` are used here, all
of which mean the same thing in both.

HDBSCAN is built for larger datasets than a 15-40 paper corpus, and on small or
topically tight sets it very often labels *everything* noise. :func:`cluster_papers`
absorbs that: it clamps parameters into the range HDBSCAN accepts and falls back
to a single cluster when no structure is found, so callers never have to handle a
label list that is entirely ``-1``.

Naming themes and writing the landscape briefing both go through
:func:`synthesis.generate_text`, so there is one Gemini client and one place where
API errors are translated. That is exactly two Gemini calls per analysis whatever
the cluster count: :func:`label_clusters` names every theme in one JSON-shaped
call, and :func:`generate_landscape_summary` writes the briefing in another.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Final, Sequence

import numpy as np
from sklearn.cluster import HDBSCAN

from synthesis import SynthesisError, generate_text

__all__ = [
    "NOISE_LABEL",
    "UNLABELED_THEME",
    "cluster_papers",
    "generate_landscape_summary",
    "label_clusters",
]

logger = logging.getLogger(__name__)

# HDBSCAN's own convention for "this point is not in any cluster". Passed through
# to callers unchanged.
NOISE_LABEL: Final = -1

# HDBSCAN rejects min_cluster_size below this.
_MIN_CLUSTER_SIZE_FLOOR: Final = 2

# Papers described per cluster in the labeling call. More context does not
# sharpen a 3-6 word phrase and only makes the prompt longer.
_MAX_LABEL_PAPERS: Final = 5

# Tries for the labeling call. The second is a re-prompt quoting what was wrong
# with the first, which only happens if the JSON constraint somehow yields
# something unusable.
_LABEL_ATTEMPTS: Final = 2

# Models wrap JSON in a fence even when told not to.
_CODE_FENCE: Final = re.compile(
    r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL | re.IGNORECASE
)

_LABEL_PREFIX: Final = re.compile(
    r"^\s*(?:theme|label|common theme)\s*[:\-]\s*", re.IGNORECASE
)

# Representative titles per theme in the landscape briefing.
_MAX_TITLES_PER_THEME: Final = 4

_MAX_LABEL_WORDS: Final = 6

UNLABELED_THEME: Final = "Unlabeled theme"

NO_CLUSTERS_MESSAGE: Final = (
    "There are no papers to summarise yet. Fetch and index a paper's references "
    "and citations first."
)

# Slightly higher than the grounded-answer default: naming a theme is a small
# creative act, and near-zero temperature tends to produce stilted noun piles.
_LABEL_TEMPERATURE: Final = 0.4

LABEL_INSTRUCTION: Final = """\
You name themes in collections of academic papers.

You are given several numbered clusters of papers, each grouped together by
similarity. Name every cluster in one reply.

Reply with a single JSON object mapping each cluster's number, as a string key, to
its name. Nothing else: no markdown fence, no commentary before or after. For
clusters 0 and 3 the reply would look like:

{"0": "Attention-based sequence models", "3": "Retrieval-augmented generation"}

Rules for each name:
- 3 to 6 words. No quotes, no trailing punctuation, no "Theme:" prefix, no
  markdown.
- Describe that cluster's shared subject matter, as specifically as its papers
  allow.
- Prefer the papers' own terminology over invented umbrella terms.
- If one cluster's papers have little in common, name the loosest honest
  connection rather than inventing a tighter one.
- Since you see every cluster at once, make the names distinguish the clusters
  from each other. Never reuse a name.
- Include a key for every cluster you were given, even one that is hard to name.\
"""

LANDSCAPE_INSTRUCTION: Final = """\
You write short briefings on the state of a research area, for someone deciding
what to read next.

You are given themes discovered by clustering a set of papers, with representative
titles and summaries for each.

Cover, in flowing prose with no headings or bullet points:
- The overall shape of the set: how many papers, how many themes, and what the
  themes are.
- What looks established: where several papers agree or build on each other.
- What looks contested or open: disagreements, competing approaches, or themes
  represented by only one or two papers.
- Any direct contradictions between specific papers, naming their titles.

Rules:
- Use ONLY the themes, titles and summaries provided. Do not add papers, findings
  or outside knowledge, and do not speculate about papers whose summaries are
  missing.
- Use the paper and theme counts exactly as given; do not recount or estimate.
- Refer to papers by their titles in quotes.
- If the material is too thin to judge what is established or contested, say so
  plainly instead of padding.
- Aim for 150-250 words. No preamble, no closing summary line.\
"""


def cluster_papers(
    embeddings: list[list[float]], min_cluster_size: int = 2
) -> list[int]:
    """Group paper embeddings into themes with HDBSCAN.

    Args:
        embeddings: One vector per paper. Expected to be unit-normalised, as
            :func:`embeddings._encode` produces.
        min_cluster_size: Smallest group HDBSCAN will call a cluster. The default
            of 2 suits the 15-40 paper corpora this backend works with; HDBSCAN's
            own default of 5 would find almost nothing at that scale. Values below
            2 are raised to 2, and values above the number of papers are lowered
            to fit, both of which HDBSCAN would otherwise reject.

    Returns:
        One label per paper, in the input order. Clustered papers get 0, 1, 2 ...
        and outliers get :data:`NOISE_LABEL` (-1), passed through as HDBSCAN
        reports it.

        Two cases are smoothed over rather than passed through. Fewer than two
        papers cannot be clustered at all, so every paper is put in cluster 0. And
        when HDBSCAN finds no clusters whatsoever — everything noise, which is
        routine for small or topically tight sets — all papers are likewise put in
        cluster 0, since "one undifferentiated group" is more useful downstream
        than "no themes at all". A mix of clusters and noise is left untouched.

    Raises:
        ValueError: If the vectors are not all the same length, or any value is
            not finite.
    """
    if not embeddings:
        return []

    if len(embeddings) < 2:
        return [0] * len(embeddings)

    matrix = _as_matrix(embeddings)
    paper_count = matrix.shape[0]

    effective_size = max(_MIN_CLUSTER_SIZE_FLOOR, min(min_cluster_size, paper_count))

    labels = HDBSCAN(
        min_cluster_size=effective_size,
        # Defaults to min_cluster_size, which HDBSCAN rejects when it exceeds the
        # sample count; pinning it keeps the clamp above sufficient.
        min_samples=min(effective_size, paper_count),
        # Matches the cosine space the Chroma collection uses. For unit-normalised
        # vectors this ranks identically to euclidean.
        metric="cosine",
        # Safe because _as_matrix always hands over a freshly allocated array;
        # set explicitly to silence a scikit-learn 1.10 deprecation warning.
        copy=False,
    ).fit_predict(matrix)

    if all(label == NOISE_LABEL for label in labels):
        return [0] * paper_count

    return [int(label) for label in labels]


def label_clusters(
    clustered_papers: dict[int, list[dict[str, Any]]],
) -> dict[int, str]:
    """Name every theme in one Gemini call.

    One call for the whole set rather than one per cluster, which keeps an
    analysis to two Gemini requests however many themes were found — enough to
    matter against a free-tier per-minute limit. Showing the model all the
    clusters together also lets it pick names that distinguish them, which
    independent calls could not do.

    Args:
        clustered_papers: Cluster ID to the papers in it, as
            :func:`cluster_papers` grouped them. A :data:`NOISE_LABEL` key is
            ignored, since outliers share no theme, and so are empty clusters.
            Only the first :data:`_MAX_LABEL_PAPERS` of each cluster are
            described, so pass the most representative first.

    Returns:
        Cluster ID to theme name, covering every non-noise cluster that had
        papers. Names are stripped of the quoting and prefixes models add and cut
        to :data:`_MAX_LABEL_WORDS` words. Any cluster the model failed to name,
        or whose papers carry no usable text, gets :data:`UNLABELED_THEME` rather
        than being left out. Returns an empty dict when there is nothing to name,
        without spending an API call.

    Raises:
        MissingAPIKeyError: If ``GEMINI_API_KEY`` is not set.
        SynthesisError: If the Gemini call itself fails. A reply that arrives but
            cannot be parsed is retried once and then degraded to
            :data:`UNLABELED_THEME`, so bad output costs labels rather than the
            whole analysis.
    """
    themes = sorted(
        cluster_id
        for cluster_id, papers in clustered_papers.items()
        if cluster_id != NOISE_LABEL and papers
    )
    if not themes:
        return {}

    blocks = {
        cluster_id: block
        for cluster_id in themes
        if (block := _describe_cluster(cluster_id, clustered_papers[cluster_id]))
    }
    if not blocks:
        return {cluster_id: UNLABELED_THEME for cluster_id in themes}

    expected = sorted(blocks)
    base_prompt = _label_prompt(blocks)
    prompt = base_prompt
    labels: dict[int, str] = {}

    for attempt in range(1, _LABEL_ATTEMPTS + 1):
        reply = generate_text(
            prompt,
            LABEL_INSTRUCTION,
            temperature=_LABEL_TEMPERATURE,
            # Constrains the reply to JSON at the API level, so the parsing below
            # is a safety net rather than the only line of defence.
            response_mime_type="application/json",
            response_schema=_label_schema(expected),
        )

        found, problem = _read_labels(reply, expected)
        labels.update(found)

        if problem is None:
            break

        unnamed = [cluster_id for cluster_id in expected if cluster_id not in labels]
        if attempt == _LABEL_ATTEMPTS:
            logger.warning(
                "cluster labeling: %s after %d attempts; leaving %d of %d themes "
                "unlabeled",
                problem,
                attempt,
                len(unnamed),
                len(expected),
            )
            break

        logger.warning("cluster labeling: %s; re-prompting", problem)
        prompt = _reprompt(base_prompt, reply, problem, unnamed)

    return {
        cluster_id: labels.get(cluster_id, UNLABELED_THEME) for cluster_id in themes
    }


def _describe_cluster(cluster_id: int, papers: Sequence[dict[str, Any]]) -> str | None:
    """Render one cluster's representative papers, or ``None`` if it has no text."""
    described = [
        block for paper in papers[:_MAX_LABEL_PAPERS] if (block := _describe_paper(paper))
    ]
    if not described:
        return None

    heading = f"--- Cluster {cluster_id} ({_plural(len(papers), 'paper')}) ---"
    return f"{heading}\n" + "\n\n".join(described)


def _label_prompt(blocks: dict[int, str]) -> str:
    """Assemble the one-shot labeling prompt for every cluster."""
    keys = ", ".join(f'"{cluster_id}"' for cluster_id in sorted(blocks))
    return (
        f"Name each of these {_plural(len(blocks), 'cluster')}.\n\n"
        + "\n\n".join(blocks[cluster_id] for cluster_id in sorted(blocks))
        + f"\n\nReply with a JSON object with exactly these keys: {keys}."
    )


def _label_schema(cluster_ids: Sequence[int]) -> dict[str, Any]:
    """Schema pinning the reply to one string per cluster.

    A plain dict in Gemini's OpenAPI subset, so this module needs no
    ``google.genai`` import to constrain the response shape.
    """
    keys = [str(cluster_id) for cluster_id in cluster_ids]
    return {
        "type": "object",
        "properties": {key: {"type": "string"} for key in keys},
        "required": keys,
        "propertyOrdering": keys,
    }


def _read_labels(
    reply: str, expected: Sequence[int]
) -> tuple[dict[int, str], str | None]:
    """Parse the labeling reply, salvaging whatever it does contain.

    Returns the labels that could be read along with a description of what was
    wrong, or ``None`` when every expected cluster was named. Partial results are
    kept deliberately: with one call covering every cluster, a reply that names
    three themes out of four should cost one label rather than all of them.
    """
    match = _CODE_FENCE.match(reply or "")
    text = (match.group(1) if match else reply or "").strip()

    try:
        payload = json.loads(text)
    except ValueError as exc:
        return {}, f"reply was not valid JSON ({exc})"

    if not isinstance(payload, dict):
        return {}, f"reply was a JSON {type(payload).__name__}, not an object"

    labels: dict[int, str] = {}
    for cluster_id in expected:
        value = _clean(payload.get(str(cluster_id)))
        if value:
            labels[cluster_id] = _tidy_label(value)

    missing = [cluster_id for cluster_id in expected if cluster_id not in labels]
    if missing:
        listed = ", ".join(str(cluster_id) for cluster_id in missing)
        return labels, f"no usable label for cluster(s) {listed}"

    return labels, None


def _reprompt(
    base_prompt: str, reply: str, problem: str, missing: Sequence[int]
) -> str:
    """Rebuild the prompt with a note about what was wrong with the last reply."""
    keys = ", ".join(f'"{cluster_id}"' for cluster_id in missing)
    return (
        f"{base_prompt}\n\n"
        f"Your previous reply could not be used: {problem}. It began: "
        f"{_truncate(reply.strip(), 200)!r}\n"
        f"Reply again with only a JSON object whose keys are exactly {keys}, "
        "each mapped to a 3 to 6 word name. No markdown fence, no commentary."
    )


def generate_landscape_summary(
    clustered_papers: dict[int, list[dict[str, Any]]],
    labels: dict[int, str] | None = None,
) -> str:
    """Write a short briefing on the themes across a whole clustered corpus.

    Args:
        clustered_papers: Cluster ID to the papers in that cluster. A
            :data:`NOISE_LABEL` key is treated as unclustered outliers rather
            than as a theme, and is reported separately from the theme count.
        labels: Optional cluster ID to theme name, as returned by
            :func:`label_clusters`. When omitted, each cluster's name is read from
            a ``clusterLabel`` key on its papers; clusters with neither are
            described as :data:`UNLABELED_THEME`.

    Returns:
        A prose briefing covering the corpus size, its themes, what looks
        established, what looks contested, and any contradictions between named
        papers. Returns :data:`NO_CLUSTERS_MESSAGE` when there are no papers at
        all, without spending an API call.

    Raises:
        MissingAPIKeyError: If ``GEMINI_API_KEY`` is not set.
        SynthesisError: If the Gemini call fails.
    """
    populated = {
        cluster_id: papers for cluster_id, papers in clustered_papers.items() if papers
    }
    if not populated:
        return NO_CLUSTERS_MESSAGE

    theme_ids = sorted(cid for cid in populated if cid != NOISE_LABEL)
    outliers = populated.get(NOISE_LABEL, [])
    paper_count = sum(len(papers) for papers in populated.values())

    # Counts are computed here and handed to the model, rather than left for it to
    # work out from the list, because language models miscount.
    header = (
        f"Corpus: {_plural(paper_count, 'paper')} in {_plural(len(theme_ids), 'theme')}"
    )
    if outliers:
        header += f", plus {_plural(len(outliers), 'unclustered outlier paper')}"

    sections = [
        _describe_theme(cluster_id, populated[cluster_id], labels)
        for cluster_id in theme_ids
    ]
    if outliers:
        sections.append(
            _format_theme_block(
                "Unclustered outliers (no shared theme)", outliers, len(outliers)
            )
        )

    prompt = f"{header}.\n\n" + "\n\n".join(sections) + "\n\nWrite the briefing."
    return generate_text(prompt, LANDSCAPE_INSTRUCTION)


def _describe_theme(
    cluster_id: int,
    papers: list[dict[str, Any]],
    labels: dict[int, str] | None,
) -> str:
    """Render one theme's name, size and representative papers for the prompt."""
    name = (labels or {}).get(cluster_id) or _label_from_papers(papers)
    return _format_theme_block(name, papers, len(papers))


def _format_theme_block(name: str, papers: Sequence[dict[str, Any]], size: int) -> str:
    """Format a theme heading followed by a few of its papers."""
    lines = [f"Theme: {name} ({_plural(size, 'paper')})"]

    for paper in papers[:_MAX_TITLES_PER_THEME]:
        title = _clean(paper.get("title")) or "Untitled"
        summary = _clean(paper.get("tldr")) or _clean(paper.get("abstract"))
        lines.append(f'  - "{title}"')
        if summary:
            lines.append(f"      {_truncate(summary, 300)}")

    remaining = size - min(size, _MAX_TITLES_PER_THEME)
    if remaining > 0:
        lines.append(f"  - (and {_plural(remaining, 'more paper')} in this theme)")

    return "\n".join(lines)


def _plural(count: int, noun: str) -> str:
    """Render ``count`` with ``noun``, pluralised.

    The prompt tells the model to reuse these counts verbatim, so "1 papers"
    would end up in the briefing.
    """
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _label_from_papers(papers: Sequence[dict[str, Any]]) -> str:
    """Read a cluster's theme name off its papers' ``clusterLabel`` key."""
    for paper in papers:
        label = _clean(paper.get("clusterLabel"))
        if label:
            return label
    return UNLABELED_THEME


def _describe_paper(paper: dict[str, Any]) -> str | None:
    """Render one paper for the labeling prompt, or ``None`` if it has no text."""
    title = _clean(paper.get("title"))
    summary = _clean(paper.get("tldr")) or _clean(paper.get("abstract"))

    if not title and not summary:
        return None

    lines = [f'Title: "{title or "Untitled"}"']
    if summary:
        lines.append(f"Summary: {_truncate(summary, 400)}")
    return "\n".join(lines)


def _tidy_label(raw: str) -> str:
    """Reduce a model reply to a bare phrase of at most six words.

    Models reliably wrap short answers in quotes, prefix them with "Theme:", bold
    them, or add a full stop, none of which belong in a UI label.
    """
    text = raw.strip().splitlines()[0] if raw.strip() else ""
    text = text.replace("*", "").replace("#", "")

    # Peeled in a loop because the decorations nest, and each one hides the next
    # from a single pass: in '**Theme: "Neural scaling laws".**' the prefix sits
    # behind the markdown and the closing quote behind the full stop.
    for _ in range(3):
        peeled = _LABEL_PREFIX.sub("", text).strip()
        peeled = peeled.strip("\"'“”‘’").strip()
        peeled = re.sub(r"[.,;:]+$", "", peeled).strip()
        if peeled == text:
            break
        text = peeled

    text = re.sub(r"\s+", " ", text)

    if not text:
        return UNLABELED_THEME

    words = text.split(" ")
    return " ".join(words[:_MAX_LABEL_WORDS]) if len(words) > _MAX_LABEL_WORDS else text


def _clean(value: Any) -> str:
    """Return a stripped string for text-ish values, else an empty string."""
    return value.strip() if isinstance(value, str) else ""


def _truncate(text: str, limit: int) -> str:
    """Shorten ``text`` to ``limit`` characters on a word boundary."""
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "..."


def _as_matrix(embeddings: list[list[float]]) -> np.ndarray:
    """Build a fresh float64 matrix, rejecting ragged or non-finite input.

    Always allocates, which is what makes ``copy=False`` safe to pass to HDBSCAN.
    """
    try:
        matrix = np.array(embeddings, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"embeddings must be a rectangular list of numeric vectors: {exc}"
        ) from exc

    if matrix.ndim != 2:
        raise ValueError(
            "embeddings must be a list of equal-length vectors, got an array with "
            f"shape {matrix.shape}"
        )
    if not np.isfinite(matrix).all():
        raise ValueError("embeddings must not contain NaN or infinite values")

    return matrix
