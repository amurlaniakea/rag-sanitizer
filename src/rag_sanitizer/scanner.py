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
# License along with this program. if not, see
# <https://www.gnu.org/licenses/>.

"""Scanner: orchestrates detectors into a per-document verdict + reasons."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from .corpus import Document, corpus_sha256
from .detectors.entity_swap import EntityProfile, detect_entity_swap
from .detectors.mimicry import detect_mimicry
from .detectors.multimodal import MultimodalDetector, select_multimodal


@dataclass
class DocVerdict:
    id: str
    verdict: str  # CLEAN | SUSPECT | POISON
    reasons: list[str] = field(default_factory=list)


@dataclass
class ScanReport:
    corpus_sha256: str
    verdicts: list[DocVerdict]
    summary: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _combine(
    mimicry_flag: bool,
    entity_flag: bool,
    multimodal_flag: bool,
    multimodal_reason: str | None,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if mimicry_flag:
        reasons.append("semantic mimicry outlier vs trusted profile")
    if entity_flag:
        reasons.append("novel entity absent from trusted profile (possible entity-swap)")
    if multimodal_flag:
        reasons.append("multimodal mismatch (possible visual poison)")
        if multimodal_reason:
            reasons.append(multimodal_reason)
    if not reasons:
        return "CLEAN", reasons
    # POISON if mimicry OR entity-swap (content-level); SUSPECT if only multimodal
    # (conservative, KI-2).
    if mimicry_flag or entity_flag:
        return "POISON", reasons
    return "SUSPECT", reasons


def scan(
    docs: Sequence[Document],
    k: float = 3.0,
    entity_profile: EntityProfile | None = None,
    multimodal_backend: str | MultimodalDetector = "heuristic",
) -> ScanReport:
    """Scan a corpus. ``docs`` with ``trust=='clean'`` form the trusted profile.

    If no document is marked clean, the largest cluster centroid is used as the
    profile (declared unsupervised mode). Deterministic (AC-6): no unseeded RNG.

    - ``entity_profile``: known entities of the user's domain (closes KI-7).
    - ``multimodal_backend``: "heuristic" (default, no deps) or "clip" (CLIP, extra).
    """
    clean = [d for d in docs if d.trust == "clean"]
    if not clean:
        clean = list(docs)
    profile_vectors = [d.embedding for d in clean]
    profile_texts = [d.text for d in clean]

    mm_detector = (
        multimodal_backend
        if isinstance(multimodal_backend, MultimodalDetector)
        else select_multimodal(multimodal_backend)
    )

    verdicts: list[DocVerdict] = []
    scan_warnings: set[str] = set()
    for d in docs:
        mim = detect_mimicry(d.embedding, profile_vectors, k=k)
        ent = detect_entity_swap(d.text, profile_texts, entity_profile=entity_profile)
        mm = mm_detector.detect(d.text, d.image_path)
        verdict, reasons = _combine(mim.flagged, ent.flagged, mm.flagged, mm.reason)
        if mim.low_confidence and mim.note:
            scan_warnings.add(mim.note)
        verdicts.append(DocVerdict(id=d.id, verdict=verdict, reasons=reasons))

    counts = {"CLEAN": 0, "SUSPECT": 0, "POISON": 0}
    for v in verdicts:
        counts[v.verdict] += 1
    return ScanReport(
        corpus_sha256=corpus_sha256(docs),
        verdicts=verdicts,
        summary=counts,
        warnings=sorted(scan_warnings),
    )
