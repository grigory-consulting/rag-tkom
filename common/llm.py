"""Konfigurierbarer LLM-Client für die RAG-Komplettkurs-Labs.

Schaltet per Umgebungsvariable zwischen lokalem Ollama (Default) und OpenAI um,
sodass alle Labs sowohl offline im Schulungsraum als auch gegen eine API laufen.

Umgebungsvariablen (siehe .env.example):
  LAB_LLM_BACKEND   "ollama" (Default) | "openai"
  OLLAMA_HOST       Default http://localhost:11434
  LAB_OLLAMA_MODEL  Default "gpt-oss:20b"
  LAB_OPENAI_MODEL  Default "gpt-4o-mini"
  OPENAI_API_KEY    nur für Backend "openai" nötig

Bewusst ohne schwere Abhängigkeit: Ollama wird über die REST-API mit der
Standardbibliothek angesprochen. Das OpenAI-SDK wird nur importiert, wenn das
Backend "openai" aktiv ist.
"""
from __future__ import annotations

import json
import os
import urllib.request
import urllib.error

try:  # optionales Komfort-Feature: .env automatisch laden, falls vorhanden
    from dotenv import load_dotenv  # type: ignore

    # override=True: die .env gewinnt gegen einen evtl. veralteten Shell-Key
    load_dotenv(override=True)
except Exception:  # python-dotenv ist nicht zwingend
    pass


def _env(name: str, default: str) -> str:
    val = os.environ.get(name)
    return val if val not in (None, "") else default


BACKEND = _env("LAB_LLM_BACKEND", "ollama").lower()
OLLAMA_HOST = _env("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = _env("LAB_OLLAMA_MODEL", "gpt-oss:20b")
OPENAI_MODEL = _env("LAB_OPENAI_MODEL", "gpt-4o-mini")


class LLMError(RuntimeError):
    pass


def _ollama_chat(messages, temperature, max_tokens, model):
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": float(temperature), "num_predict": int(max_tokens)},
    }
    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:  # Ollama läuft nicht / Modell fehlt
        raise LLMError(
            f"Ollama unter {OLLAMA_HOST} nicht erreichbar ({exc}). "
            f"Läuft 'ollama serve' und ist das Modell '{model}' gezogen "
            f"('ollama pull {model}')?"
        ) from exc
    return data["message"]["content"]


_openai_client = None


def _openai_chat(messages, temperature, max_tokens, model):
    global _openai_client
    if _openai_client is None:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise LLMError(
                "Backend 'openai' gewählt, aber das Paket 'openai' ist nicht "
                "installiert (pip install openai)."
            ) from exc
        _openai_client = OpenAI()
    resp = _openai_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=float(temperature),
        max_tokens=int(max_tokens),
    )
    return resp.choices[0].message.content


def chat(messages, temperature: float = 0.0, max_tokens: int = 512,
         backend: str | None = None, model: str | None = None) -> str:
    """Ein Chat-Aufruf gegen das konfigurierte Backend.

    messages: Liste von {"role": "system"|"user"|"assistant", "content": str}
    Gibt den reinen Antworttext zurück.
    """
    backend = (backend or BACKEND).lower()
    if backend == "openai":
        return _openai_chat(messages, temperature, max_tokens, model or OPENAI_MODEL)
    if backend == "ollama":
        return _ollama_chat(messages, temperature, max_tokens, model or OLLAMA_MODEL)
    raise LLMError(f"Unbekanntes Backend '{backend}' (erlaubt: ollama, openai).")


def complete(prompt: str, system: str | None = None, **kwargs) -> str:
    """Bequemer Single-Turn-Aufruf: optionaler System-Prompt + User-Prompt."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return chat(messages, **kwargs)


def active_backend() -> dict:
    """Kleine Selbstauskunft für die erste Lab-Zelle.

    Zeigt nur, was für das aktive Backend gilt: der Ollama-Host taucht gar nicht
    erst auf, wenn gegen die OpenAI-API gefahren wird.
    """
    info = {
        "backend": BACKEND,
        "model": OPENAI_MODEL if BACKEND == "openai" else OLLAMA_MODEL,
    }
    if BACKEND == "ollama":
        info["ollama_host"] = OLLAMA_HOST
    return info


if __name__ == "__main__":
    print("Aktives Backend:", active_backend())
    print("Testantwort:", complete("Antworte mit genau einem Wort: Funktioniert?",
                                    max_tokens=20))
