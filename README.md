# RAG-Demo: Hydraulikpumpe P-12

Dieses Repository enthält kompakte Beispiele für Retrieval-Augmented Generation
(RAG). Als Wissensbasis dient eine **vollständig synthetische** Dokumentation der
fiktiven Hydraulikpumpe P-12. Die Beispiele sind für Schulung und Experimente
gedacht; alle technischen Werte sind erfunden.

## Was enthalten ist

| Pfad | Inhalt |
|---|---|
| `common/corpus/` | Markdown-Dokumente mit Metadaten |
| `common/corpus.py` | Laden des Korpus |
| `common/retrieval.py` | Chunking, BM25, Dense Retrieval und Reciprocal Rank Fusion |
| `common/embeddings.py` | Lokale oder OpenAI-basierte Embeddings |
| `common/llm.py` | Einheitliche Schnittstelle für Ollama und OpenAI |
| `rag_antwort.ipynb` | Kleinstes vollständiges RAG-Beispiel |
| `chatapp.py` | RAG-Chat mit Gesprächshistorie |
| `agent.py` | RAG-Agent mit Datei- und Kommando-Tools |
| `chunking.ipynb`, `hybrid.ipynb`, `judge.ipynb` | Vertiefende Übungen |

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install numpy rank-bm25 python-dotenv sentence-transformers openai jupyter
```

Die Datei `.env` wird automatisch geladen und von Git ignoriert.

### OpenAI verwenden

```dotenv
LAB_LLM_BACKEND=openai
LAB_EMBED_BACKEND=openai
OPENAI_API_KEY=dein-api-key
LAB_OPENAI_MODEL=gpt-4o-mini
LAB_OPENAI_EMBED_MODEL=text-embedding-3-small
```

### Lokal mit Ollama arbeiten

```dotenv
LAB_LLM_BACKEND=ollama
LAB_OLLAMA_MODEL=gpt-oss:20b
LAB_EMBED_BACKEND=local
```

Das Ollama-Modell muss vorher lokal verfügbar sein. Das standardmäßige lokale
Embedding-Modell wird beim ersten Start von `sentence-transformers` geladen.

## RAG-Chat starten

```powershell
python .\chatapp.py
```

Die Chat-App sucht zu jeder Frage passende Abschnitte, übergibt nur diesen
Kontext an das Sprachmodell und zeigt die verwendeten Quellen. Die letzten zehn
Dialogrunden bleiben als Gesprächshistorie erhalten, damit Rückfragen verstanden
werden.

Beispielfragen:

- `Wie hoch ist der Dauerbetriebsdruck der P-12?`
- `Welches Hydrauliköl wird empfohlen?`
- `Und welche Sicherheitsmaßnahmen gelten dabei?`

## RAG-Agent starten

Einzelne Aufgabe:

```powershell
python .\agent.py "Ermittle den Dauerbetriebsdruck und speichere die Antwort in antwort.txt."
```

Interaktiver Modus:

```powershell
python .\agent.py
```

Vor jedem Agentenlauf wird automatisch Retrieval ausgeführt. Danach kann das
Modell über ein bewusst minimales JSON-Protokoll folgende Tools aufrufen:

- `retrieve(query, top_k)` – weitere Suche im P-12-Korpus
- `read_file(path)` – Textdatei lesen
- `write_file(path, content)` – Text- oder JSON-Datei schreiben
- `execute(command)` – PowerShell-Befehl oder Programm ausführen

Modellantworten und Tool-Ergebnisse werden im Terminal ausgegeben. Dadurch ist
der Ablauf während einer Schulung direkt nachvollziehbar.

> **Sicherheit:** `agent.py` ist ein lokales Lehrbeispiel und keine Sandbox.
> `execute` besitzt keine Befehlsfreigabeliste. Auch absolute Dateipfade und
> Pfade mit `..` werden nicht blockiert. Verwende den Agenten deshalb nur in
> einer vertrauenswürdigen Testumgebung.

## Minimale RAG-Pipeline

```python
from common import retrieval as R
from common.corpus import load_corpus

dokumente = load_corpus()
retriever = R.HybridRetriever(R.chunk_corpus(dokumente))
treffer = retriever.search("Wie hoch ist der Dauerbetriebsdruck?", k=3)
print(R.format_context(treffer))
```

## Hinweise

Der Korpus in diesem Repository ist ausschließlich synthetisch und dient nur zu
Demonstrationszwecken.
