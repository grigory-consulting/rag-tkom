
import numpy as np

from common import embeddings as E
from common.corpus import load_corpus
from common.goldset import load_queries, qrels
from common import retrieval as R



import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

docs = load_corpus()


def chunk_absatz(docs, max_chars=550):
    """Strategie A: absatzweise packen (der Default aus common.retrieval)."""
    return R.chunk_corpus(docs, max_chars=max_chars)


def chunk_sliding(docs, size=400, overlap=80):
    """Strategie B: zeichenbasiertes Sliding-Window mit Überlappung."""
    out = []
    for d in docs:
        text = d.text
        step = max(1, size - overlap)
        i = idx = 0
        while i < len(text):
            piece = text[i:i + size].strip()
            if piece:
                out.append(R.Chunk(f"{d.doc_id}#s{idx}", d.doc_id, d.title, piece, d.acl))
                idx += 1
            i += step
    return out


def chunk_recursive(docs, size=400, overlap=60):
    """Strategie C: rekursiver Splitter (langchain), respektiert Trennzeichen."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""])
    out = []
    for d in docs:
        for idx, piece in enumerate(splitter.split_text(d.text)):
            piece = piece.strip()
            if piece:
                out.append(R.Chunk(f"{d.doc_id}#r{idx}", d.doc_id, d.title, piece, d.acl))
    return out


strategien = {
    "absatz":    chunk_absatz(docs),
    "sliding400": chunk_sliding(docs),
    "recursive400": chunk_recursive(docs),
}
for name, ch in strategien.items():
    laengen = [len(c.text) for c in ch]
    print(f"  {name:14s}: {len(ch):3d} Chunks, Median {int(np.median(laengen)):3d} Zeichen")
