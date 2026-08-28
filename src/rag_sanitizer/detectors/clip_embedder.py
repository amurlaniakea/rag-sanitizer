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

"""CLIP text+image embedder (lazy import). Behind this, rag-sanitizer compares a
document's text against its declared image to detect visual poison (Vis-Poison,
2608.20756) where the image itself is the payload.

Lazily imports ``transformers`` + ``torch`` (extra ``multimodal``) so the fast suite
and the heuristic detector need no heavyweight model. The text and image are encoded
into a shared embedding space; cosine similarity is computed downstream.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# Pinned reference revision of openai/clip-vit-base-patch32 (HF API, lastModified 2024-02-29).
# Pinning the revision is the concrete mitigation for B615 (bandit): it fixes the exact
# weights served, so a future upstream change on the default branch cannot silently alter
# the model. Pass model_name="org/model@<sha>" to override per-call.
DEFAULT_CLIP_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"


@dataclass
class ClipScore:
    text_image_similarity: float


class ClipEmbedder:
    """Thin wrapper over HuggingFace CLIP (text+image) with lazy import.

    Used by the ``real_embeddings`` test suite / ``--multimodal clip``. Raises a clear
    ``ImportError`` if the ``multimodal`` extra is not installed.
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32") -> None:
        try:
            import torch  # noqa: F401
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "transformers/torch not installed. Install extra: pip install -e '.[multimodal]'"
            ) from exc
        # Resolve a concrete revision: prefer an explicit "@<sha>" suffix on model_name,
        # otherwise fall back to the pinned DEFAULT_CLIP_REVISION.
        if "@" in model_name:
            resolved_name, revision = model_name.split("@", 1)
        else:
            resolved_name, revision = model_name, DEFAULT_CLIP_REVISION
        # B615 (bandit) mitigation: revision is pinned (explicit or DEFAULT_CLIP_REVISION),
        # so the exact weights are fixed and upstream default-branch changes cannot alter them.
        # No #nosec needed: bandit does not flag from_pretrained when revision is set.
        self._model = CLIPModel.from_pretrained(resolved_name, revision=revision)
        self._processor = CLIPProcessor.from_pretrained(resolved_name, revision=revision)
        self.model_name = resolved_name
        self.revision = revision

    def similarity(self, text: str, image_path: str) -> float:
        """Cosine similarity between the document text and its declared image in CLIP space."""
        from PIL import Image
        from transformers import CLIPModel, CLIPProcessor

        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"image not found: {image_path}")
        image = Image.open(image_path).convert("RGB")
        inputs = self._processor(
            text=[text], images=image, return_tensors="pt", padding=True, truncation=True
        )
        # model already loaded; reference types for clarity
        _ = (CLIPModel, CLIPProcessor)
        out = self._model(**inputs)
        # CLIP returns logits_per_text = (cosine_similarity / temperature) for the single
        # (text, image) pair. temperature for clip-vit-base-patch32 is 0.01, so dividing by
        # 100 recovers an approximate cosine similarity in roughly [-1, 1].
        # NOTE: do NOT apply sigmoid here — raw logits (~±20..30) saturate sigmoid to ~1.0
        # for every pair, destroying discriminability (this was the v0.2 bug: every pair
        # scored 1.0000). Temperature-scaled cosine is the correct, discriminative score.
        logits = out.logits_per_text
        sim = float(logits[0, 0].detach()) / 100.0
        return sim
