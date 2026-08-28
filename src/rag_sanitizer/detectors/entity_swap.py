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
    r"(?P<money>\$\s?\d[\d.,]*)|"
    r"(?P<year>\b(19|20)\d{2}\b)|"
    r"(?P<pct>\b\d{1,3}\s?%)|"
    r"(?P<org>\b[A-Z][A-Za-z0-9]+(?:\s[A-Z][A-Za-z0-9]+){0,2}\b)"
)


def extract_entities(text: str) -> set[str]:
    found = set()
    for m in _ENTITY_RE.finditer(text):
        for kind, val in m.groupdict().items():
            if val:
                found.add(f"{kind}:{val.strip()}")
    return found


def detect_entity_swap(
    doc_text: str,
    profile_texts: list[str],
) -> EntitySwapResult:
    """Flag entities in ``doc_text`` absent from the trusted ``profile_texts``.

    A poison-by-entity-swap replaces a factual entity (e.g. a year, an amount) with
    a different one; that novel entity will not appear in the trusted profile.
    """
    profile_entities: set[str] = set()
    for t in profile_texts:
        profile_entities |= extract_entities(t)
    doc_entities = extract_entities(doc_text)
    novel = sorted(doc_entities - profile_entities)
    return EntitySwapResult(novel_entities=novel, flagged=bool(novel))
