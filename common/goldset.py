"""Gold-Query-Set für den Hydraulikpumpe-P-12-Korpus.

Zwölf deutsche Fragen mit abgestufter Relevanz (graded relevance, 0-3) auf
Dokumentebene. Die Relevanzurteile sind aus den Korpusdokumenten abgeleitet und
faktisch konsistent mit dem roten Faden der Folien (350 bar Dauerbetriebsdruck,
HLP 46, Wartung alle 500 h, erstes Öl nach 50 h, ...).

Skala:
  3  perfekter Treffer (beantwortet die Frage direkt und vollständig)
  2  stark relevant (enthält die Antwort, aber nicht als Hauptthema)
  1  am Rande relevant (erwähnt das Thema)
  0  irrelevant (nicht in der Relevanztabelle geführt)

Verwendung:
  from common.goldset import load_queries, gold_doc_ids
  queries = load_queries()              # Liste von Query-Objekten
  rel = queries[0].relevance            # {"p12-betriebsdruck": 3, ...}

Die Frage q12 zielt bewusst auf das vertrauliche Service-/Garantiedokument
(acl=vertraulich). Im ACL-Lab (Teil IV) zeigt sie den Unterschied zwischen
"relevant" und "für diesen Nutzer zugänglich".
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Query:
    qid: str
    text: str
    relevance: dict          # doc_id -> grade (1..3)


_QUERIES: list[Query] = [
    Query("q01", "Wie hoch ist der zulässige Dauerbetriebsdruck der Pumpe?",
          {"p12-betriebsdruck": 3, "p12-datenblatt": 2}),
    Query("q02", "Welches Hydrauliköl soll verwendet werden?",
          {"p12-hydraulikoel": 3, "p12-datenblatt": 1}),
    Query("q03", "Wann muss das erste Öl gewechselt werden?",
          {"p12-wartungsintervalle": 3, "p12-inbetriebnahme": 2, "p12-hydraulikoel": 1}),
    Query("q04", "Mit welchem Anzugsmoment werden die Pumpenkopf-Schrauben angezogen?",
          {"p12-anzugsmomente": 3}),
    Query("q05", "Was bedeutet der Fehlercode E02?",
          {"p12-stoerungstabelle": 3}),
    Query("q06", "Welche Schutzausrüstung ist bei der Wartung nötig?",
          {"p12-sicherheit": 3}),
    Query("q07", "Wie wird die Pumpe in Betrieb genommen und entlüftet?",
          {"p12-inbetriebnahme": 3, "p12-sicherheit": 1}),
    Query("q08", "Wie oft muss der Rücklauffilter getauscht werden?",
          {"p12-wartungsintervalle": 3, "p12-hydraulikoel": 2, "p12-ersatzteile": 1}),
    Query("q09", "Welche Teile-Nummer hat der Wellendichtring?",
          {"p12-ersatzteile": 3}),
    Query("q10", "Welcher Spitzendruck ist kurzzeitig zulässig?",
          {"p12-betriebsdruck": 3, "p12-datenblatt": 2}),
    Query("q11", "Was tun bei Kavitation (pfeifendes Geräusch)?",
          {"p12-stoerungstabelle": 3, "p12-hydraulikoel": 1}),
    Query("q12", "Wie lange gilt die Werksgarantie und was kostet der Servicevertrag?",
          {"p12-service-garantie": 3}),
]


def load_queries() -> list[Query]:
    """Alle Gold-Queries als Liste (Kopie der Relevanz-Dicts, damit Labs sie
    gefahrlos verändern können)."""
    return [Query(q.qid, q.text, dict(q.relevance)) for q in _QUERIES]


def gold_doc_ids(qid: str, min_grade: int = 1) -> set[str]:
    """Menge der relevanten doc_ids für eine Query ab Mindestnote min_grade.

    min_grade=2 liefert die Dokumente, die als Beleg (Citation) erwartet werden;
    min_grade=1 die volle Relevanzmenge für Recall.
    """
    for q in _QUERIES:
        if q.qid == qid:
            return {d for d, g in q.relevance.items() if g >= min_grade}
    raise KeyError(qid)


def qrels() -> dict:
    """Relevanzurteile im ranx/BEIR-Format: {qid: {doc_id: grade}}."""
    return {q.qid: dict(q.relevance) for q in _QUERIES}


if __name__ == "__main__":
    qs = load_queries()
    print(f"{len(qs)} Gold-Queries")
    for q in qs:
        rel = ", ".join(f"{d}={g}" for d, g in q.relevance.items())
        print(f"  [{q.qid}] {q.text}\n        -> {rel}")
