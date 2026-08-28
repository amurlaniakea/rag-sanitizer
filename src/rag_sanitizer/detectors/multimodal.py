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

"""Multimodal (visual poison) detector — v0.1 heuristic (KI-2).

Vis-Poison (2608.20756) shows the poisoned image itself is the payload, without
touching caption/metadata. v0.1 uses a cheap heuristic: when a document carries an
image, it flags SUSPECT if the image file is missing/unreadable OR if a ``caption``
field disagrees with the document text (caption/text mismatch). Real embedding-based
detection (CLIP) is feature 002.

This detector is deliberately conservative: it never clears an image as safe, it
only raises a SUSPECT signal when a structural mismatch is present.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass


@dataclass
class MultimodalResult:
    reason: str | None
    flagged: bool


def detect_multimodal(
    doc_text: str,
    image_path: str | None,
) -> MultimodalResult:
    """Heuristic visual-poison check for a single document.

    Returns ``flagged=True`` (SUSPECT) only on a structural mismatch:
      - image declared but file missing/unreadable, or
      - a ``caption:`` line in the text disagrees with the surrounding body.
    Otherwise ``flagged=False`` (not an automatic clear — see KI-2).
    """
    # No image referenced: nothing to flag multimodally.
    if not image_path:
        return MultimodalResult(reason=None, flagged=False)

    if not os.path.isfile(image_path):
        return MultimodalResult(
            reason=f"declared image missing/unreadable: {image_path}", flagged=True
        )

    # Caption mismatch: a "caption:" field whose content does NOT appear in the
    # document body once the caption line itself is removed. We compare against the
    # body excluding the caption line (not against the whole doc_text, which would
    # always contain the substring we just extracted).
    caption = _caption_line(doc_text)
    if caption is not None:
        body_without_caption = _body_without_caption_line(doc_text)
        if not _caption_supported_by_body(caption, body_without_caption):
            return MultimodalResult(
                reason="caption references content absent from document body", flagged=True
            )

    return MultimodalResult(reason=None, flagged=False)


def _body_without_caption_line(text: str) -> str:
    """Return ``text`` with any ``caption:`` line removed (for mismatch comparison)."""
    kept = []
    for line in text.splitlines():
        low = line.lower().lstrip("-* ").strip()
        if low.startswith("caption:"):
            continue
        kept.append(line)
    return "\n".join(kept)


def _caption_supported_by_body(caption: str, body: str) -> bool:
    """True if the caption content is present in the body (excluding the caption line).

    Uses token overlap so a caption like "see chart showing record performance"
    is considered supported only if those words actually appear in the rest of the
    document. A caption that references content absent from the body returns False
    (mismatch -> flag).
    """
    cap_tokens = set(re.findall(r"[A-Za-z0-9]+", caption.lower()))
    if not cap_tokens:
        # Empty caption: nothing to support, treat as supported (no mismatch).
        return True
    body_tokens = set(re.findall(r"[A-Za-z0-9]+", body.lower()))
    # Require a meaningful fraction of caption tokens to be present in the body.
    # A caption that does not relate to the body yields low overlap -> mismatch.
    overlap = len(cap_tokens & body_tokens) / len(cap_tokens)
    return overlap >= 0.5


def _caption_line(text: str) -> str | None:
    for line in text.splitlines():
        low = line.lower().lstrip("-* ").strip()
        if low.startswith("caption:"):
            return line.split(":", 1)[1].strip()
    return None
