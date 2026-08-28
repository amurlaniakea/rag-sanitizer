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

"""Semantic-mimicry detector: flags documents that are outliers from the trust profile.

A poison-by-mimicry document imitates the tone/structure of trusted documents but
steers the content. We model the trusted corpus as a centroid in embedding space
and flag documents that fall outside ``k * sigma`` of the per-dimension spread.
"""

from __future__ import annotations

from dataclasses import dataclass

# Minimum number of trusted-profile documents required for the k*sigma threshold to
# be statistically meaningful. Below this, the per-document spread (sigma) estimated
# from n points is unstable noise, so any mimicry verdict is low-confidence and must
# be surfaced as such rather than reported silently as reliable.
MIN_PROFILE_SIZE = 10


@dataclass
class MimicryResult:
    distance: float
    threshold: float
    flagged: bool
    low_confidence: bool = False
    note: str | None = None


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _centroid(vectors: list[list[float]]) -> list[float]:
    dim = len(vectors[0])
    c = [0.0] * dim
    for v in vectors:
        for i in range(dim):
            c[i] += v[i]
    return [x / len(vectors) for x in c]


def _spread(vectors: list[list[float]], centroid: list[float]) -> float:
    # Mean cosine distance to the centroid (robust enough for fixture geometry).
    dists = [_cosine(v, centroid) for v in vectors]
    # cosine distance = 1 - cosine
    mean_dist = sum(1.0 - d for d in dists) / len(dists)
    return mean_dist


def detect_mimicry(
    doc_vector: list[float],
    profile_vectors: list[list[float]],
    k: float = 3.0,
) -> MimicryResult:
    """Return whether ``doc_vector`` is an outlier from ``profile_vectors``.

    Uses cosine distance to the trusted centroid. Flagged when the distance exceeds
    ``k * spread`` of the trusted set. Order-independent (AC-2).

    When ``len(profile_vectors) < MIN_PROFILE_SIZE`` the k*sigma threshold is built on
    too few points to be reliable; the result is still computed (so callers get a
    deterministic answer) but ``low_confidence`` is set and ``note`` explains why, so
    the scanner can surface it instead of emitting a silently unreliable verdict.
    """
    if not profile_vectors:
        return MimicryResult(0.0, 0.0, False)
    low_confidence = len(profile_vectors) < MIN_PROFILE_SIZE
    note = (
        f"profile too small for reliable k-sigma threshold "
        f"(n={len(profile_vectors)} < {MIN_PROFILE_SIZE})"
        if low_confidence
        else None
    )
    centroid = _centroid(profile_vectors)
    spread = _spread(profile_vectors, centroid)
    threshold = k * spread
    dist = 1.0 - _cosine(doc_vector, centroid)
    # Guard against degenerate zero-spread profiles.
    if spread == 0.0:
        threshold = 0.0
    return MimicryResult(
        distance=dist, threshold=threshold, flagged=dist > threshold,
        low_confidence=low_confidence, note=note,
    )
