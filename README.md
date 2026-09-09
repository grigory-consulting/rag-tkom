# common

Gemeinsame Bausteine für die Hands-on-Labs des RAG-Komplettkurses.

| Modul | Inhalt |
|-------|--------|
| `corpus.py` + `corpus/*.md` | Lader und 10 Dokumente zur Hydraulikpumpe P-12, je mit `doc_id`, `title` und `acl` |
| `embeddings.py` | Embeddings, lokal über sentence-transformers oder über die OpenAI-API, plus `cosine` |
| `llm.py` | Chat-Client, umschaltbar zwischen Ollama und OpenAI |
| `retrieval.py` | Chunking, BM25, Dense, Reciprocal Rank Fusion, Cross-Encoder-Reranking |
| `goldset.py` | 12 Gold-Queries mit abgestufter Relevanz |
| `poison_doc.md` | präpariertes Angriffsdokument für das Prompt-Injection-Lab |

Der Korpus ist synthetisch. Die Pumpe P-12 gibt es nicht, alle Werte sind für den
Kurs erfunden und konsistent gehalten.

## Verwendung

```python
from common.corpus import load_corpus
from common import retrieval as R

docs = load_corpus()
index = R.HybridRetriever(R.chunk_corpus(docs))
print(index.search("Wie hoch ist der Dauerbetriebsdruck?", k=3))
```

Konfiguration über Umgebungsvariablen, siehe Docstrings in `llm.py` und
`embeddings.py`: `LAB_LLM_BACKEND`, `LAB_OLLAMA_MODEL`, `LAB_OPENAI_MODEL`,
`LAB_EMBED_BACKEND`, `LAB_EMBED_MODEL`.

Abhängigkeiten: `numpy`, `rank_bm25`, `sentence-transformers`, optional `openai`
und `python-dotenv`.
