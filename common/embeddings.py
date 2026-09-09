"""Embeddings für die RAG-Komplettkurs-Labs.

Zwei Backends, per Umgebungsvariable umschaltbar:
  LAB_EMBED_BACKEND   "local" (Default) | "openai"

Lokal läuft sentence-transformers (offline, mehrsprachig). "openai" nutzt die
OpenAI-Embeddings-API mit dem Key aus .env. Beide liefern L2-normierte Vektoren,
sodass das Skalarprodukt direkt die Cosine-Similarity ist (siehe Teil V).

Umgebungsvariablen:
  LAB_EMBED_BACKEND       "local" (Default) | "openai"
  LAB_EMBED_MODEL         lokal, Default "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
  LAB_OPENAI_EMBED_MODEL  OpenAI, Default "text-embedding-3-small"
  OPENAI_API_KEY          nur für Backend "openai" nötig
"""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

try:  # .env laden (override=True: .env gewinnt gegen veralteten Shell-Key), optional
    from dotenv import load_dotenv  # type: ignore

    load_dotenv(override=True)
except Exception:  # python-dotenv ist nicht zwingend
    pass


EMBED_BACKEND = os.environ.get("LAB_EMBED_BACKEND", "local").lower()
EMBED_MODEL = os.environ.get(
    "LAB_EMBED_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
OPENAI_EMBED_MODEL = os.environ.get("LAB_OPENAI_EMBED_MODEL", "text-embedding-3-small")


@lru_cache(maxsize=2)
def _model(name: str):
    from sentence_transformers import SentenceTransformer  # lazy import

    return SentenceTransformer(name)


def _l2norm(mat: np.ndarray) -> np.ndarray:
    return mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)


_openai_client = None


def _embed_openai(texts: list[str], model: str, batch: int = 256) -> np.ndarray:
    """Embeddings über die OpenAI-API, in Batches (Reihenfolge bleibt erhalten)."""
    global _openai_client
    if _openai_client is None:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "LAB_EMBED_BACKEND=openai gewählt, aber das Paket 'openai' ist "
                "nicht installiert (pip install openai)."
            ) from exc
        _openai_client = OpenAI()
    out: list[list[float]] = []
    for i in range(0, len(texts), batch):
        resp = _openai_client.embeddings.create(model=model, input=texts[i:i + batch])
        out.extend(row.embedding for row in sorted(resp.data, key=lambda d: d.index))
    return np.asarray(out, dtype=np.float32)


def embed(texts, model: str | None = None, normalize: bool = True,
          backend: str | None = None) -> np.ndarray:
    """Embedde eine Liste von Strings zu einer (n, d)-Matrix.

    backend wählt "local" (sentence-transformers) oder "openai"; ohne Angabe gilt
    LAB_EMBED_BACKEND. normalize=True liefert L2-normierte Vektoren, sodass das
    Skalarprodukt direkt die Cosine-Similarity ist (siehe Teil V der Folien).
    """
    if isinstance(texts, str):
        texts = [texts]
    texts = list(texts)
    backend = (backend or EMBED_BACKEND).lower()

    if backend == "openai":
        vecs = _embed_openai(texts, model or OPENAI_EMBED_MODEL)
        return _l2norm(vecs) if normalize else vecs
    if backend == "local":
        vecs = _model(model or EMBED_MODEL).encode(
            texts, normalize_embeddings=normalize, show_progress_bar=False
        )
        return np.asarray(vecs, dtype=np.float32)
    raise ValueError(f"unbekanntes Embedding-Backend {backend!r} (erlaubt: local, openai)")


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine-Similarity zwischen Zeilen von a und Zeilen von b -> (len(a), len(b))."""
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return an @ bn.T


def active_model() -> str:
    """Name des aktiven Embedding-Modells (je nach Backend)."""
    return OPENAI_EMBED_MODEL if EMBED_BACKEND == "openai" else EMBED_MODEL


def active_embed() -> dict:
    """Kleine Selbstauskunft: Backend + Modell."""
    return {"backend": EMBED_BACKEND, "model": active_model()}
