"""Einfache Chat-Anwendung mit RAG."""

from common import llm
from common.corpus import load_corpus
from common import retrieval as R


SYSTEM_PROMPT = """
Sie sind ein Hydraulik-Experte. Nutzen Sie den Gesprächsverlauf, um Rückfragen
zu verstehen. Beantworten Sie fachliche Fragen ausschließlich anhand des neu
bereitgestellten Kontexts, knapp und konkret. Zitieren Sie verwendete Quellen
mit ihrer Kennung in eckigen Klammern. Wenn die Antwort nicht im Kontext steht,
sagen Sie das klar.
"""


def answer(frage, retriever, history):
    vorherige_fragen = [
        message["content"] for message in history if message["role"] == "user"
    ]
    suchanfrage = " ".join(vorherige_fragen[-2:] + [frage])
    treffer = retriever.search(suchanfrage, k=3)
    kontext = R.format_context(treffer)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *history,
        {
            "role": "user",
            "content": f"Aktuelle Frage:\n{frage}\n\nNeuer Kontext:\n{kontext}",
        },
    ]
    antwort = llm.chat(
        messages,
        temperature=0.0,
        max_tokens=200,
    )
    return antwort, treffer


def main():
    dokumente = load_corpus()
    retriever = R.HybridRetriever(R.chunk_corpus(dokumente))
    history = []

    print("P-12 RAG-Chat. Beenden mit 'exit' oder 'quit'.")

    while True:
        try:
            frage = input("\nFrage: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if frage.lower() in {"exit", "quit"}:
            break
        if not frage:
            continue

        try:
            antwort, treffer = answer(frage, retriever, history)
            print(f"\nAntwort: {antwort}")
            quellen = dict.fromkeys(chunk.doc_id for chunk, _ in treffer)
            print("Quellen:", ", ".join(quellen))
            history.extend([
                {"role": "user", "content": frage},
                {"role": "assistant", "content": antwort},
            ])
            history = history[-20:]
        except Exception as exc:
            print(f"Fehler: {exc}")


if __name__ == "__main__":
    main()
