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
# You should have received a copy of the GNU Affero General Public License along
# with this program. If not, see
# <https://www.gnu.org/licenses/>.

"""KI-1: embeddings reales (sentence-transformers) validan la lógica del scanner.

Marca pytest con `real_embeddings`; se salta si el extra no está instalado.
NO corre en CI por defecto (descarga de modelo ~80MB + CPU/GPU).
"""

import pytest

from rag_sanitizer.detectors.mimicry import MIN_PROFILE_SIZE, detect_mimicry
from rag_sanitizer.embedder import SentenceTransformerEmbedder

pytestmark = pytest.mark.real_embeddings

_FINANCE_PROFILE = [
    "The Q3 2025 revenue was 4.2M with 312 enterprise customers.",
    "We acquired Nimbus Analytics for 18M in 2024.",
    "The security audit of 2025 found 0 critical issues.",
    "Annual recurring revenue grew 27 percent year over year to 51M.",
    "Our gross margin improved to 71 percent in the third quarter.",
    "The board approved a 0.40 per share dividend payable in December.",
    "Customer churn dropped to 4.1 percent following the onboarding redesign.",
    "Operating expenses totaled 22M, below the 24M guidance for the quarter.",
    "We signed 14 new design partners across the EU and APAC regions.",
    "Net retention held at 118 percent as expansion outpaced contraction.",
    "Free cash flow reached 9.3M after 3.1M of capitalized engineering spend.",
    "The sales cycle shortened to 47 days on the mid-market segment.",
]


def _have_real() -> bool:
    try:
        import sentence_transformers  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.skipif(
    not _have_real(), reason="sentence-transformers not installed (extra real-embeddings)"
)
def test_real_embedder_dimension_and_determinism():
    emb = SentenceTransformerEmbedder()
    v1 = emb.embed("The Q3 2025 revenue was $4.2M")
    v2 = emb.embed("The Q3 2025 revenue was $4.2M")
    assert len(v1) == 384
    assert v1 == v2  # deterministic


@pytest.mark.skipif(
    not _have_real(), reason="sentence-transformers not installed (extra real-embeddings)"
)
def test_min_profile_size_guard_flags_low_confidence():
    # KI-9 follow-up: a trusted profile with fewer than MIN_PROFILE_SIZE documents
    # cannot support a reliable k*sigma threshold; the result must be surfaced as
    # low-confidence instead of a silently unreliable verdict.
    emb = SentenceTransformerEmbedder()
    small_profile = [emb.embed(t) for t in _FINANCE_PROFILE[:3]]
    assert len(small_profile) < MIN_PROFILE_SIZE
    r = detect_mimicry(emb.embed("Our headquarters is in Madrid, Spain."), small_profile, k=3.0)
    assert r.low_confidence is True
    assert r.note is not None
    assert "too small" in r.note


@pytest.mark.skipif(
    not _have_real(), reason="sentence-transformers not installed (extra real-embeddings)"
)
def test_real_embedder_separates_out_of_domain_document():
    # With a statistically valid profile (>= MIN_PROFILE_SIZE), MiniLM real embeddings
    # DO separate a document from a completely different domain (cooking recipe) from
    # the finance trust profile at k=2.0. This proves the detector works on real
    # embeddings; the threshold choice (k) is calibrated, not the model dismissed.
    emb = SentenceTransformerEmbedder()
    profile_vecs = [emb.embed(t) for t in _FINANCE_PROFILE]
    assert len(profile_vecs) >= MIN_PROFILE_SIZE
    out_of_domain = emb.embed(
        "Preheat the oven to 180 degrees celsius and whisk the egg whites with the sugar "
        "until the meringue forms stiff glossy peaks before folding in the flour."
    )
    r = detect_mimicry(out_of_domain, profile_vecs, k=2.0)
    assert r.low_confidence is False
    assert r.flagged is True


@pytest.mark.skipif(
    not _have_real(), reason="sentence-transformers not installed (extra real-embeddings)"
)
def test_real_embedder_does_not_separate_fluent_gibberish():
    # KI-9 (documented limitation, NOT a bug to force green): MiniLM real embeddings do
    # NOT separate fluent gibberish of similar length from genuine prose at k=2.0/3.0.
    # We assert the honest behavior -- not flagged -- so the limitation is pinned by a
    # test rather than papered over. Calibrate against real user corpora before relying.
    emb = SentenceTransformerEmbedder()
    profile_vecs = [emb.embed(t) for t in _FINANCE_PROFILE]
    gibberish = emb.embed(
        "Zephyrion quasel fracton vellic thramb exoquill nimbus drell paravox synotic wraith."
    )
    r = detect_mimicry(gibberish, profile_vecs, k=2.0)
    assert r.low_confidence is False
    assert r.flagged is False
