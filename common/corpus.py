"""Lader für den synthetischen Hydraulikpumpe-P-12-Korpus.

Die Dokumente liegen als Markdown unter common/corpus/ 

    ---
    doc_id: p12-datenblatt
    title: Datenblatt Hydraulikpumpe P-12
    acl: all            # all | wartung | vertraulich
    ---
    <Markdown-Inhalt>


"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

CORPUS_DIR = Path(os.environ.get(
    "LAB_CORPUS_DIR",
    str(Path(__file__).parent / "corpus"),
))


@dataclass
class Doc:
    doc_id: str
    title: str
    text: str
    acl: str = "all"
    path: str = ""
    meta: dict = field(default_factory=dict)


def _parse(path: Path) -> Doc:
    raw = path.read_text(encoding="utf-8")
    meta: dict = {}
    body = raw
    if raw.startswith("---"):
        _, fm, body = raw.split("---", 2)
        for line in fm.strip().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
    return Doc(
        doc_id=meta.get("doc_id", path.stem),
        title=meta.get("title", path.stem),
        text=body.strip(),
        acl=meta.get("acl", "all"),
        path=str(path),
        meta=meta,
    )


def load_corpus(include_acl=None, corpus_dir: Path | None = None) -> list[Doc]:
    """Lädt alle Dokumente, optional gefiltert auf eine Menge erlaubter ACL-Stufen.

    include_acl=None  -> alle Dokumente
    include_acl={"all", "wartung"} -> nur Dokumente mit dieser Sichtbarkeit
    """
    base = Path(corpus_dir or CORPUS_DIR)
    docs = [_parse(p) for p in sorted(base.glob("*.md"))]
    if include_acl is not None:
        allowed = set(include_acl)
        docs = [d for d in docs if d.acl in allowed]
    return docs


def load_poison(path: str | None = None) -> Doc:
    """Lädt das präparierte Angriffsdokument für das Prompt-Injection-Lab.

    Liegt absichtlich getrennt unter common/ (nicht im sauberen Korpus), damit
    Sie es im Lab kontrolliert in den Index einschleusen können.
    """
    p = Path(path or (Path(__file__).parent / "poison_doc.md"))
    return _parse(p)


if __name__ == "__main__":
    docs = load_corpus()
    print(f"{len(docs)} Dokumente geladen aus {CORPUS_DIR}")
    for d in docs:
        print(f"  [{d.acl:11s}] {d.doc_id:22s} {d.title}  ({len(d.text)} Zeichen)")
