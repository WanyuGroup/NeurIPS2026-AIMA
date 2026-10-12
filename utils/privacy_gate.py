import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

_EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    re.I,
)
_PHONE_RE = re.compile(
    r"(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b|\b\d{10,15}\b",
)
_URL_RE = re.compile(r"https?://[^\s]+|www\.[^\s]+", re.I)
_SSN_LIKE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b|\b\d{9}\b")
_CREDIT_LIKE = re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")

_STOPWORDS = {
    "the",
    "and",
    "for",
    "that",
    "this",
    "with",
    "from",
    "have",
    "was",
    "were",
    "are",
    "been",
    "being",
    "their",
    "they",
    "them",
    "what",
    "when",
    "where",
    "which",
    "who",
    "will",
    "would",
    "could",
    "should",
    "about",
    "into",
    "your",
    "you",
    "not",
    "but",
    "can",
    "all",
    "any",
    "has",
    "had",
    "how",
    "its",
    "our",
    "out",
    "than",
    "then",
    "too",
    "very",
}


def _unique_preserve(seq: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in seq:
        k = x.strip()
        if not k or k.lower() in seen:
            continue
        seen.add(k.lower())
        out.append(k)
    return out


def extract_sensitive_entities(profile: str, max_entities: int = 96) -> List[str]:
    if not profile:
        return []
    found: List[str] = []
    found += _EMAIL_RE.findall(profile)
    found += _PHONE_RE.findall(profile)
    found += _URL_RE.findall(profile)
    found += _SSN_LIKE.findall(profile)
    found += _CREDIT_LIKE.findall(profile)

    cap = re.findall(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", profile)
    found += cap

    for tok in re.findall(r"[A-Za-z][A-Za-z0-9'\-]{3,}", profile):
        tl = tok.lower()
        if tl in _STOPWORDS:
            continue
        if len(tok) >= 5:
            found.append(tok)

    found = _unique_preserve(found)
    return found[:max_entities]


def score_pii(query: str) -> Tuple[float, List[str]]:
    hits: List[str] = []
    if not query:
        return 0.0, hits
    for label, rx in [
        ("email", _EMAIL_RE),
        ("phone", _PHONE_RE),
        ("url", _URL_RE),
        ("ssn_like", _SSN_LIKE),
        ("card_like", _CREDIT_LIKE),
    ]:
        for m in rx.finditer(query):
            hits.append(f"{label}:{m.group(0)[:64]}")
    return float(len(hits)), hits


def score_ner_overlap(query: str, entities: List[str]) -> Tuple[float, List[str]]:
    if not query or not entities:
        return 0.0, []
    ql = query.lower()
    leaked: List[str] = []
    for e in entities:
        el = e.strip()
        if len(el) < 3:
            continue
        if el.lower() in ql:
            leaked.append(el[:120])
    return float(len(leaked)), leaked


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return dot / (na * nb)


class _LazyEmbedder:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self._model = None

    def encode(self, texts: List[str]) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_id)
        return self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)


def score_semantic_leak(
    query: str,
    entities: List[str],
    embedder: Optional[_LazyEmbedder],
    delta_sem: float,
) -> Tuple[float, List[str]]:
    if not query or not entities or embedder is None:
        return 0.0, []
    try:
        vecs = embedder.encode([query] + entities)
        vq = vecs[0]
        flagged: List[str] = []
        for i, e in enumerate(entities):
            c = _cosine(list(vq), list(vecs[i + 1]))
            if c > delta_sem:
                flagged.append(f"{e[:80]} (cos={c:.3f})")
        return float(len(flagged)), flagged
    except Exception:
        return _score_sem_fallback(query, entities, delta_sem)


def _score_sem_fallback(query: str, entities: List[str], delta_sem: float) -> Tuple[float, List[str]]:
    q_tokens = set(re.findall(r"[a-z]{4,}", query.lower()))
    flagged: List[str] = []
    for e in entities:
        et = set(re.findall(r"[a-z]{4,}", e.lower()))
        if not q_tokens or not et:
            continue
        inter = len(q_tokens & et)
        union = len(q_tokens | et) or 1
        jacc = inter / union
        if jacc > delta_sem:
            flagged.append(f"{e[:80]} (lex_overlap={jacc:.3f})")
    return float(len(flagged)), flagged


@dataclass
class PrivacyScoreDetail:
    total: float
    s_pii: float
    s_ner: float
    s_sem: float
    pii_hits: List[str]
    ner_hits: List[str]
    sem_hits: List[str]


def compute_privacy_score(
    query: str,
    entities: List[str],
    embedder: Optional[_LazyEmbedder],
    delta_sem: float,
) -> PrivacyScoreDetail:
    spii, pii_hits = score_pii(query)
    sner, ner_hits = score_ner_overlap(query, entities)
    if embedder is not None:
        ssem, sem_hits = score_semantic_leak(query, entities, embedder, delta_sem)
    else:
        ssem, sem_hits = _score_sem_fallback(query, entities, delta_sem)
    total = spii + sner + ssem
    return PrivacyScoreDetail(
        total=total,
        s_pii=spii,
        s_ner=sner,
        s_sem=ssem,
        pii_hits=pii_hits,
        ner_hits=ner_hits,
        sem_hits=sem_hits,
    )


def build_privacy_feedback_en(detail: PrivacyScoreDetail) -> str:
    lines = [
        "Privacy risk score S(q) = S_PII + S_NER + S_SEM (composite).",
        f"Numerical breakdown: S_PII={detail.s_pii:.0f}, S_NER={detail.s_ner:.0f}, S_SEM={detail.s_sem:.0f}; total={detail.total:.0f}.",
    ]
    if detail.pii_hits:
        lines.append("PII-style pattern triggers (regex): " + "; ".join(detail.pii_hits[:12]))
    else:
        lines.append("PII-style pattern triggers: none detected.")
    if detail.ner_hits:
        lines.append("Lexical overlap with sensitive phrases from private context: " + "; ".join(detail.ner_hits[:12]))
    else:
        lines.append("Lexical overlap with extracted sensitive phrases: none detected.")
    if detail.sem_hits:
        lines.append("Semantic proximity flags (embedding or lexical fallback): " + "; ".join(detail.sem_hits[:12]))
    else:
        lines.append("Semantic proximity flags: none above threshold.")
    lines.append(
        "Revise the candidate so that task intent is preserved but none of the above leakage modes remain. "
        "Generalize or omit specifics; never paste private post text verbatim."
    )
    return "\n".join(lines)


def pri_rewrite_user_block(
    round_idx: int,
    feedback_en: str,
    candidate_output: str,
    aug_label: str,
) -> str:
    schema_hint = (
        "Preserve the exact JSON-in-markdown structure required by the system prompt "
        "(e.g., refined_question or answer fields as specified)."
    )
    return f"""

# Privacy-constrained rewriting (step {round_idx})

You must regenerate your previous structured output to eliminate privacy leakage while keeping utility for the downstream cloud model.

Task branch (unchanged): {aug_label}

{feedback_en}

{schema_hint}

Previous candidate output (to rewrite — do not repeat its unsafe parts):
{candidate_output}
"""


def get_embedder(model_id: Optional[str]) -> Optional[_LazyEmbedder]:
    if not model_id:
        return None
    return _LazyEmbedder(model_id)
