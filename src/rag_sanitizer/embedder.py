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

"""Embedder interface and a deterministic dummy embedder for the fast suite.

The real embedding backend (sentence-transformers / OpenAI / HF) is injected via
the ``Embedder`` protocol in v0.2. v0.1 uses ``DummyDeterministicEmbedder`` so the
fast test suite needs no network and no model download (KI-1).
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol, runtime_checkable

DIM = 64


@runtime_checkable
class Embedder(Protocol):
    """Embedding backend contract (injectable)."""

    def embed(self, text: str) -> list[float]:
        """Return a fixed-dimension embedding for ``text``."""
        ...


class DummyDeterministicEmbedder:
    """Deterministic, dependency-free embedder used by the fast suite.

    Maps a hashed bag-of-tokens into a fixed-dimension vector. NOT semantically
    meaningful: it exists to validate the scanner logic (outlier detection,
    determinism), not embedding quality (KI-1).
    """

    def __init__(self, dim: int = DIM) -> None:
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        tokens = re.findall(r"\w+", text.lower())
        vec = [0.0] * self.dim
        for tok in tokens:
            h = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(h[:4], "big") % self.dim
            sign = 1.0 if int.from_bytes(h[4:8], "big") % 2 == 0 else -1.0
            vec[idx] += sign
        # L2-normalize for stable distance geometry.
        norm = sum(v * v for v in vec) ** 0.5
        if norm == 0.0:
            return [0.0] * self.dim
        return [v / norm for v in vec]


class SentenceTransformerEmbedder:
    """Real embedding backend (sentence-transformers) implementing the ``Embedder`` Protocol.

    Lazily imports ``sentence_transformers`` so the fast suite (DummyDeterministicEmbedder)
    needs no network or model download. Closed vector dimension depends on the model
    (MiniLM-L6-v2 -> 384). Used by the ``real_embeddings`` test suite (AC-1/AC-2).

    Example
    -------
    >>> emb = SentenceTransformerEmbedder("sentence-transformers/all-MiniLM-L6-v2")
    >>> vec = emb.embed("The Q3 2025 revenue was $4.2M")
    >>> len(vec)
    384
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "sentence-transformers not installed. Install extra: "
                "pip install -e '.[real-embeddings]'"
            ) from exc
        self._model = SentenceTransformer(model_name)
        self.model_name = model_name

    def embed(self, text: str) -> list[float]:
        vec = self._model.encode(text, normalize_embeddings=True, convert_to_numpy=True)
        return vec.tolist()
