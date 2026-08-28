# SPDX-FileCopyrightText: 2026 Pedro Sordo Martínez <amurlaniakea@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Copyright (C) 2026 Pedro Sordo Martínez
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public
# License along with this program. If not, see
# <https://www.gnu.org/licenses/>.

"""Entity-swap detector (lightweight, KI-3).

Extracts a small set of high-signal entities (money amounts, years, ORG-like
capitalized tokens, percentages) and flags a document whose entities disagree
with the trusted profile (entities that never appear in the trusted set, or that
replace a trusted entity). v0.1 uses a pattern extractor, not a full NER.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class EntitySwapResult:
    novel_entities: list[str]
    flagged: bool


_ENTITY_RE = re.compile(
    r"(?P<money>\$\s?\d[\d.,]*)"
    r"|(?P<year>\b(19|20)\d{2}\b)"
    r"|(?P<pct>\b\d{1,3}\s?%)"
    r"|(?P<org>\b[A-Z][A-Za-z0-9]+(?:\s[A-Z][A-Za-z0-9]+){0,2}\b)"
)

# Function/start-of-sentence words that are NOT proper nouns / organizations.
# Excluded so generic prose ("The report shows...", "This year...") does not flag.
_ORG_STOPWORDS = {
    "the",
    "this",
    "that",
    "these",
    "those",
    "our",
    "we",
    "i",
    "you",
    "they",
    "he",
    "she",
    "it",
    "a",
    "an",
    "and",
    "or",
    "but",
    "if",
    "then",
    "else",
    "when",
    "where",
    "why",
    "how",
    "what",
    "who",
    "which",
    "whose",
    "while",
    "although",
    "because",
    "since",
    "after",
    "before",
    "during",
    "between",
    "among",
    "against",
    "within",
    "without",
    "upon",
    "across",
    "through",
    "their",
    "his",
    "her",
    "its",
    "my",
    "your",
    "in",
    "on",
    "at",
    "by",
    "to",
    "of",
    "for",
    "with",
    "from",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "has",
    "have",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "should",
    "could",
    "may",
    "might",
    "must",
    "can",
    "not",
    "no",
    "yes",
    "all",
    "any",
    "each",
    "every",
    "both",
    "such",
    "same",
    "other",
    "one",
    "two",
    "three",
    "first",
    "second",
    "next",
    "last",
    "new",
    "old",
    "more",
    "most",
    "less",
    "few",
    "many",
    "also",
    "only",
    "than",
    "over",
    "under",
}


def _is_real_org(token_span: str) -> bool:
    """Filter a matched ORG span: drop function words and single common nouns.

    A span is treated as a real organization/proper noun only if:
      - it is an ALL-CAPS acronym (e.g. "NASA", "IBM"), or
      - it has 2+ capitalized words (e.g. "Atlas Ventures"), and
      - none of its words is a stopword.
    """
    words = token_span.split()
    if any(w.lower() in _ORG_STOPWORDS for w in words):
        return False
    if len(words) >= 2:
        return True
    # single word: only ALL-CAPS acronyms (2+ letters) count
    return len(words) == 1 and words[0].isupper() and len(words[0]) >= 2


def extract_entities(text: str) -> set[str]:
    found = set()
    for m in _ENTITY_RE.finditer(text):
        for kind, val in m.groupdict().items():
            if not val:
                continue
            if kind == "org":
                if not _is_real_org(val.strip()):
                    continue
                found.add(f"org:{val.strip()}")
            else:
                found.add(f"{kind}:{val.strip()}")
    return found


@dataclass
class EntityProfile:
    """Known entities of the user's domain (injectable). Closes KI-7 without a NER model.

    A poison-by-entity-swap may replace a known entity with a different one even when
    that entity is a single capitalized-or-lowercase word (e.g. a city: ``Madrid`` ->
    ``Beijing``). The v0.1 regex could not catch single-word swaps; ``EntityProfile``
    lets the user list their known entities so any divergence from the trusted corpus
    is flagged regardless of capitalization or word count.
    """

    known: set[str]

    def __post_init__(self) -> None:
        self._norm = {self._norm_key(k) for k in self.known}

    @staticmethod
    def _norm_key(token: str) -> str:
        return token.strip().lower()

    def contains_norm(self, token: str) -> bool:
        return self._norm_key(token) in self._norm


def _looks_like_entity(token: str) -> bool:
    """Heuristic: is ``token`` a plausible named entity (for KI-7 profile-swap check)?

    True if it is capitalized (proper noun-like) and not a function/stopword. Used to
    avoid flagging common lowercase words as swaps. Single ALL-CAPS acronyms and
    Capitalized words count; lowercase words do not (unless in the user's known set,
    handled separately by the caller).
    """
    if not token:
        return False
    if token.lower() in _ORG_STOPWORDS:
        return False
    return token[0].isupper()


def detect_entity_swap(
    doc_text: str,
    profile_texts: list[str],
    entity_profile: EntityProfile | None = None,
) -> EntitySwapResult:
    """Flag entities in ``doc_text`` absent from the trusted ``profile_texts``.

    A poison-by-entity-swap replaces a factual entity (e.g. a year, an amount) with
    a different one; that novel entity will not appear in the trusted profile.

    When ``entity_profile`` is provided (KI-7 mitigation), any token in the document
    that is a known entity of the user's domain but is ABSENT from the trusted corpus
    texts is flagged as a swap — including single-word entities like a city name.
    This closes the KI-7 hole (``Madrid`` -> ``Beijing``) without a heavy NER model.
    """
    profile_entities: set[str] = set()
    for t in profile_texts:
        profile_entities |= extract_entities(t)
    doc_entities = extract_entities(doc_text)
    novel = sorted(doc_entities - profile_entities)

    if entity_profile is not None:
        # KI-7: any entity-candidate token in the doc that is NOT in the trusted corpus
        # AND NOT in the user's known entities is flagged as a possible swap. This catches
        # single-word swaps like Madrid -> Beijing without a heavy NER model: the user
        # declares Madrid/Spain as known; Beijing is a candidate not in corpus/known -> flag.
        profile_entity_norm = {t.split(":", 1)[1].lower() for t in profile_entities}
        doc_tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-]*", doc_text)
        swapped = sorted(
            tok
            for tok in doc_tokens
            if tok.lower() not in profile_entity_norm
            and not entity_profile.contains_norm(tok)
            and _looks_like_entity(tok)
        )
        for s in swapped:
            novel.append(f"profile-swap:{s}")

    return EntitySwapResult(novel_entities=novel, flagged=bool(novel))
