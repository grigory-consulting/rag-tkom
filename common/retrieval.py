"""Wiederverwendbare Retrieval-Bausteine für die angewandten Labs.

Die Teaching-Labs bauen ihre Kernfunktion bewusst selbst (Chunking in Teil V,
RRF und Reranking in Teil VI, die Metriken in Teil VII). Die angewandten Labs
(Pipeline, ACL, Injection, Evaluation, Betrieb) nutzen dagegen diese Engine,
damit sie sich auf ihr eigenes Thema konzentrieren können.

Alles läuft lokal: BM25 über rank_bm25, Dense über common.embeddings
(sentence-transformers), der optionale Reranker über einen Cross-Encoder.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from .embeddings import embed

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Einfache, sprachunabhängige Tokenisierung für BM25 (lowercase, \\w+)."""
    return _TOKEN_RE.findall(text.lower())


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    acl: str = "all"

    def for_embedding(self) -> str:
        return f"{self.title}. {self.text}"


def chunk_corpus(docs, max_chars: int = 550, overlap_paras: int = 0) -> list[Chunk]:
    """Absatzbasiertes Chunking: packt Absätze bis max_chars in einen Chunk.

    Bewusst schlicht und absatzgrenzentreu, weil der P-12-Korpus kurz und gut
    strukturiert ist. Teil V vergleicht ausführlichere Strategien. Die doc_id-
    und acl-Zuordnung jedes Chunks bleibt erhalten (wichtig für das ACL-Lab).
    """
    chunks: list[Chunk] = []
    for d in docs:
        paras = [p.strip() for p in re.split(r"\n\s*\n", d.text) if p.strip()]
        buf: list[str] = []
        size = 0
        idx = 0

        def flush(buf, idx):
            if not buf:
                return idx
            chunks.append(Chunk(
                chunk_id=f"{d.doc_id}#{idx}",
                doc_id=d.doc_id,
                title=d.title,
                text="\n\n".join(buf),
                acl=d.acl,
            ))
            return idx + 1

        for p in paras:
            if buf and size + len(p) > max_chars:
                idx = flush(buf, idx)
                buf = buf[-overlap_paras:] if overlap_paras else []
                size = sum(len(x) for x in buf)
            buf.append(p)
            size += len(p)
        flush(buf, idx)
    return chunks


def _rank(scores: np.ndarray, k: int) -> list[int]:
    return list(np.argsort(scores)[::-1][:k])


class BM25Retriever:
    """Lexikalische Suche. Trifft exakte Begriffe (Teile-Nummern, Fehlercodes)."""

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self._bm25 = BM25Okapi([tokenize(c.for_embedding()) for c in chunks])

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        scores = self._bm25.get_scores(tokenize(query))
        return [(self.chunks[i], float(scores[i])) for i in _rank(scores, k)]


class DenseRetriever:
    """Semantische Suche über lokale Embeddings (Cosine via Skalarprodukt)."""

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self._mat = embed([c.for_embedding() for c in chunks])  # (n, d), L2-normiert

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        q = embed(query)[0]
        scores = self._mat @ q
        return [(self.chunks[i], float(scores[i])) for i in _rank(scores, k)]


def reciprocal_rank_fusion(rankings: list[list[str]], rrf_k: int = 60) -> dict[str, float]:
    """RRF über mehrere Ergebnislisten (je eine geordnete Liste von chunk_ids).

    score(d) = sum_l 1 / (rrf_k + rank_l(d)). Rangbasiert, daher robust gegen
    unterschiedliche Score-Skalen von BM25 und Dense.
    """
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, start=1):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (rrf_k + rank)
    return fused


class HybridRetriever:
    """BM25 + Dense, fusioniert per Reciprocal Rank Fusion."""

    def __init__(self, chunks: list[Chunk], rrf_k: int = 60, pool: int = 20):
        self.chunks = chunks
        self._by_id = {c.chunk_id: c for c in chunks}
        self.bm25 = BM25Retriever(chunks)
        self.dense = DenseRetriever(chunks)
        self.rrf_k = rrf_k
        self.pool = pool

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        bm = [c.chunk_id for c, _ in self.bm25.search(query, self.pool)]
        dn = [c.chunk_id for c, _ in self.dense.search(query, self.pool)]
        fused = reciprocal_rank_fusion([bm, dn], self.rrf_k)
        top = sorted(fused.items(), key=lambda x: -x[1])[:k]
        return [(self._by_id[cid], score) for cid, score in top]


def make_retriever(chunks: list[Chunk], mode: str = "hybrid", **kwargs):
    """Retriever nach Modus: 'sparse' (nur BM25), 'dense' (nur Embeddings) oder 'hybrid'.

    Ein Umschalter für Demos und Labs. Alle drei teilen dieselbe Schnittstelle
    search(query, k), nur der Modus wechselt. Zusatzargumente (rrf_k, pool) gehen
    an den HybridRetriever.
    """
    m = mode.lower()
    if m in ("sparse", "bm25", "lexical"):
        return BM25Retriever(chunks)
    if m in ("dense", "semantic", "embeddings"):
        return DenseRetriever(chunks)
    if m == "hybrid":
        return HybridRetriever(chunks, **kwargs)
    raise ValueError(f"unbekannter Modus {mode!r} (erlaubt: sparse, dense, hybrid)")


def compare_retrievers(chunks: list[Chunk], query: str,
                       k: int = 5) -> dict[str, list[tuple[Chunk, float]]]:
    """Dieselbe Query durch sparse, dense und hybrid, für den direkten Vergleich.

    Zeigt im Unterricht, dass BM25 exakte Begriffe trifft (Teile-Nummern,
    Fehlercodes), Dense Umschreibungen findet und Hybrid beides einsammelt. Die
    Embeddings entstehen nur einmal, weil der HybridRetriever seinen BM25- und
    Dense-Teil schon hält. Verglichen werden die Ränge, nicht die Rohscores
    (BM25 und Cosine liegen auf verschiedenen Skalen).
    """
    hybrid = HybridRetriever(chunks)
    return {
        "sparse": hybrid.bm25.search(query, k),
        "dense": hybrid.dense.search(query, k),
        "hybrid": hybrid.search(query, k),
    }


_cross_encoder = None


def cross_encoder_rerank(query, scored_chunks, k: int = 5,
                         model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    """Ordnet Kandidaten mit einem Cross-Encoder neu (Query und Chunk gemeinsam).

    Erwartet eine Kandidatenliste [(Chunk, score), ...] und gibt die Top-k nach
    Cross-Encoder-Score zurück. Der erste Aufruf zieht das Modell (~90 MB)."""
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder(model)
    cands = [c for c, _ in scored_chunks]
    pairs = [[query, c.for_embedding()] for c in cands]
    ce_scores = _cross_encoder.predict(pairs)
    order = np.argsort(ce_scores)[::-1][:k]
    return [(cands[i], float(ce_scores[i])) for i in order]


def collapse_to_docs(scored_chunks) -> list[tuple[str, float]]:
    """Chunk-Ranking auf Dokumentebene reduzieren (bester Chunk je doc_id).

    Nötig, wenn gegen ein Gold-Set auf Dokumentebene (common.goldset) gemessen
    wird, das Retrieval aber Chunks liefert.
    """
    best: dict[str, float] = {}
    for chunk, score in scored_chunks:
        if chunk.doc_id not in best or score > best[chunk.doc_id]:
            best[chunk.doc_id] = score
    return sorted(best.items(), key=lambda x: -x[1])


def format_context(scored_chunks, max_chars: int | None = None) -> str:
    """Nummerierter Kontextblock für den Prompt, jede Quelle mit [doc_id].

    Genau dieses Format erwartet der Grounding-Prompt der Labs: das Modell soll
    mit der Kennung in eckigen Klammern zitieren.
    """
    blocks = []
    for chunk, _ in scored_chunks:
        text = chunk.text
        if max_chars and len(text) > max_chars:
            text = text[:max_chars].rstrip() + " ..."
        blocks.append(f"[{chunk.doc_id}] {chunk.title}\n{text}")
    return "\n\n".join(blocks)
