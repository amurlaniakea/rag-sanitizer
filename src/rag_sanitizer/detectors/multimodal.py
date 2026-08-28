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

"""Multimodal (visual poison) detector.

KI-2: v0.1 used a cheap heuristic (image missing / caption mismatch). v0.2 adds a
real CLIP-based detector (text+image similarity) behind ``ClipMultimodalDetector``.
Both are deliberately conservative: they never clear an image as safe, they only
raise a SUSPECT signal with evidence. ``select_multimodal`` picks the backend.

Vis-Poison (2608.20756) shows the poisoned image itself is the payload, without
touching caption/metadata; CLIP catches the case where the image content disagrees
with the document text even when caption/metadata look fine.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class MultimodalResult:
    reason: str | None
    flagged: bool
    score: float | None = None


@runtime_checkable
class MultimodalDetector(Protocol):
    """Multimodal backend contract."""

    def detect(self, doc_text: str, image_path: str | None) -> MultimodalResult: ...


# --- Heuristic backend (v0.1, no heavy deps) ---------------------------------


class HeuristicMultimodalDetector:
    """v0.1 heuristic: image missing/unreadable OR caption/text mismatch (token overlap)."""

    def detect(self, doc_text: str, image_path: str | None) -> MultimodalResult:
        if not image_path:
            return MultimodalResult(reason=None, flagged=False)
        if not os.path.isfile(image_path):
            return MultimodalResult(
                reason=f"declared image missing/unreadable: {image_path}", flagged=True
            )
        caption = _caption_line(doc_text)
        if caption is not None:
            body = _body_without_caption_line(doc_text)
            if not _caption_supported_by_body(caption, body):
                return MultimodalResult(
                    reason="caption references content absent from document body", flagged=True
                )
        return MultimodalResult(reason=None, flagged=False)


def _body_without_caption_line(text: str) -> str:
    kept = []
    for line in text.splitlines():
        low = line.lower().lstrip("-* ").strip()
        if low.startswith("caption:"):
            continue
        kept.append(line)
    return "\n".join(kept)


def _caption_supported_by_body(caption: str, body: str) -> bool:
    cap_tokens = set(re.findall(r"[A-Za-z0-9]+", caption.lower()))
    if not cap_tokens:
        return True
    body_tokens = set(re.findall(r"[A-Za-z0-9]+", body.lower()))
    overlap = len(cap_tokens & body_tokens) / len(cap_tokens)
    return overlap >= 0.5


def _caption_line(text: str) -> str | None:
    for line in text.splitlines():
        low = line.lower().lstrip("-* ").strip()
        if low.startswith("caption:"):
            return line.split(":", 1)[1].strip()
    return None


# --- CLIP backend (v0.2, extra "multimodal") --------------------------------


class ClipMultimodalDetector:
    """CLIP text+image similarity backend. Conservative: only raises SUSPECT with evidence.

    Compares the document text against its declared image in CLIP space. If the image
    is missing, falls back to the structural flag (same as heuristic). If similarity is
    below ``threshold`` (default 0.5 on the sigmoid(logit) score), raises SUSPECT with a
    ``clip_score`` for evidence. Never returns a "clear" verdict.
    """

    def __init__(
        self, threshold: float = 0.5, model_name: str = "openai/clip-vit-base-patch32"
    ) -> None:
        from .clip_embedder import ClipEmbedder

        self._clip = ClipEmbedder(model_name)
        self.threshold = threshold

    def detect(self, doc_text: str, image_path: str | None) -> MultimodalResult:
        if not image_path:
            return MultimodalResult(reason=None, flagged=False)
        if not os.path.isfile(image_path):
            return MultimodalResult(
                reason=f"declared image missing/unreadable: {image_path}", flagged=True
            )
        sim = self._clip.similarity(doc_text, image_path)
        if sim < self.threshold:
            return MultimodalResult(
                reason=(
                    f"CLIP text-image similarity {sim:.3f} < {self.threshold} "
                    "(visual poison suspect)"
                ),
                flagged=True,
                score=sim,
            )
        return MultimodalResult(reason=None, flagged=False, score=sim)


# --- Backend selection ------------------------------------------------------


def select_multimodal(backend: str) -> MultimodalDetector:
    """Return a multimodal detector by name.

    ``backend="heuristic"`` -> no heavy deps (default for fast suite).
    ``backend="clip"`` -> CLIP (requires the ``multimodal`` extra; raises ImportError if absent).
    """
    if backend == "heuristic":
        return HeuristicMultimodalDetector()
    if backend == "clip":
        return ClipMultimodalDetector()
    raise ValueError(f"unknown multimodal backend: {backend!r}")
