"""Incremental sentence chunker: turns a stream of LLM tokens into speakable segments.

Feeding TTS one sentence at a time (instead of the full reply) is the single biggest lever on
time-to-first-audio. The first segment is allowed to be a shorter clause so audio can start
even sooner; later segments wait for full sentences to keep prosody natural.
"""

from __future__ import annotations

import re

_ABBREVIATIONS = {
    "mr.",
    "mrs.",
    "ms.",
    "dr.",
    "st.",
    "vs.",
    "etc.",
    "e.g.",
    "i.e.",
    "approx.",
    "no.",
    "fig.",
    "u.s.",
    "a.m.",
    "p.m.",
    "min.",
    "max.",
    "alt.",
    "hdg.",
}
_SENTENCE_END = re.compile(r"[.!?…]+[\"')\]]*(?=\s)")
_CLAUSE_END = re.compile(r"[,;:—–](?=\s)")
_MARKDOWN = re.compile(r"(\*\*|__|`|^#+\s*|^\s*[-*]\s+)", re.MULTILINE)


def clean_for_speech(text: str) -> str:
    text = _MARKDOWN.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


class SentenceChunker:
    """``mode``:

    * ``full``     — no streaming: the entire reply is one segment (baseline to quantify chunking)
    * ``sentence`` — whole sentences only
    * ``clause``   — like sentence, but the first segment may end at a comma/semicolon
    * ``eager``    — like clause, and if no boundary has appeared after ``first_max_words`` words the
                     first segment is cut at a word boundary (lowest time-to-first-audio, slight
                     prosody cost on the first phrase only)
    """

    MODES = ("full", "sentence", "clause", "eager")

    def __init__(
        self,
        first_clause_min_chars: int = 24,
        min_chars: int = 12,
        max_chars: int = 220,
        mode: str = "clause",
        first_max_words: int = 7,
    ):
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        self.mode = mode
        self.first_max_words = first_max_words
        self.first_clause_min_chars = first_clause_min_chars
        self.min_chars = min_chars
        self.max_chars = max_chars
        self._buf = ""
        self._emitted = 0

    def _is_abbrev(self, text: str, end: int) -> bool:
        start = text.rfind(" ", 0, end) + 1
        word = text[start : end + 1].lower()
        if word in _ABBREVIATIONS:
            return True
        # decimals like "3.5" never match because we require whitespace after the period
        return len(word) == 2 and word[0].isalpha() and word[1] == "."  # single initial "J."

    def _cut(self) -> str | None:
        buf = self._buf
        for m in _SENTENCE_END.finditer(buf):
            end = m.end()
            if buf[m.start()] == "." and self._is_abbrev(buf, m.start()):
                continue
            if len(buf[:end].strip()) >= self.min_chars:
                return buf[:end]
        if self._emitted == 0 and self.mode in ("clause", "eager"):
            for m in _CLAUSE_END.finditer(buf):
                if m.end() >= self.first_clause_min_chars:
                    return buf[: m.end()]
        if self._emitted == 0 and self.mode == "eager":
            words = list(re.finditer(r"\S+(?=\s)", buf))
            if len(words) >= self.first_max_words:
                return buf[: words[self.first_max_words - 1].end()]
        if len(buf) >= self.max_chars:
            cut = buf.rfind(" ", 0, self.max_chars)
            return buf[: cut if cut > 0 else self.max_chars]
        return None

    def feed(self, token: str) -> list[str]:
        """Add a token; return zero or more complete segments ready for TTS."""
        self._buf += token
        out: list[str] = []
        if self.mode == "full":
            return out
        while True:
            seg = self._cut()
            if seg is None:
                break
            self._buf = self._buf[len(seg) :]
            seg = clean_for_speech(seg)
            if seg:
                out.append(seg)
                self._emitted += 1
        return out

    def flush(self) -> list[str]:
        seg = clean_for_speech(self._buf)
        self._buf = ""
        if seg:
            self._emitted += 1
            return [seg]
        return []
