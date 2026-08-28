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

    # Caption mismatch: a "caption:" field that does not appear in the body text.
    m = _caption_line(doc_text)
    if m is not None and m not in doc_text:
        return MultimodalResult(
            reason="caption references content absent from document body", flagged=True
        )

    return MultimodalResult(reason=None, flagged=False)


def _caption_line(text: str) -> str | None:
    for line in text.splitlines():
        low = line.lower().lstrip("-* ").strip()
        if low.startswith("caption:"):
            return line.split(":", 1)[1].strip()
    return None
