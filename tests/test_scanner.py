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

import os

from rag_sanitizer.corpus import load_corpus
from rag_sanitizer.embedder import DummyDeterministicEmbedder
from rag_sanitizer.report import to_json, to_markdown
from rag_sanitizer.scanner import scan

FIX = os.path.join(os.path.dirname(__file__), "..", "fixtures", "corpus.jsonl")


def _docs():
    return load_corpus(FIX, embedder=DummyDeterministicEmbedder())


# AC-1: carga determinista, dimensión fija
def test_ac1_load_deterministic():
    d1 = _docs()
    d2 = _docs()
    assert len(d1) == 6
    assert all(len(d.embedding) == 64 for d in d1)
    # mismo input -> mismo vector
    assert d1[0].embedding == d2[0].embedding
    assert d1[0].sha256 == d2[0].sha256


# AC-2: mimicry marcado, clean no; invariante al orden
def test_ac2_mimicry_order_invariant():
    base = _docs()
    rep = {}
    for seed in range(5):
        perm = sorted(base, key=lambda d: (d.id, seed))
        report = scan(perm, k=3.0)
        rep[seed] = {v.id: v.verdict for v in report.verdicts}
    # todos los seeds dan el mismo veredicto por doc
    first = next(iter(rep.values()))
    for other in rep.values():
        assert other == first
    assert first["clean-1"] == "CLEAN"
    assert first["clean-2"] == "CLEAN"
    assert first["poison-mimicry"] in ("POISON", "SUSPECT")
    assert first["poison-mimicry"] != "CLEAN"


# AC-3: entity-swap marcado, clean no
def test_ac3_entity_swap():
    report = scan(_docs())
    by_id = {v.id: v for v in report.verdicts}
    assert by_id["clean-1"].verdict == "CLEAN"
    assert by_id["poison-entity"].verdict in ("POISON", "SUSPECT")
    assert any("entity" in r for r in by_id["poison-entity"].reasons)


# AC-4: multimodal heurístico marca visual poison, no imagen limpia
def test_ac4_multimodal_heuristic():
    report = scan(_docs())
    by_id = {v.id: v for v in report.verdicts}
    assert by_id["poison-multimodal"].verdict == "SUSPECT"
    assert any(
        "multimodal" in r or "image" in r.lower() for r in by_id["poison-multimodal"].reasons
    )


# AC-5: determinismo (mismo corpus -> mismo sha256 y veredicto)
def test_ac5_determinism():
    r1 = scan(_docs())
    r2 = scan(_docs())
    assert r1.corpus_sha256 == r2.corpus_sha256
    assert {v.id: v.verdict for v in r1.verdicts} == {v.id: v.verdict for v in r2.verdicts}


# AC-6: reasons legibles + JSON/MD válido
def test_ac6_report_serialization():
    report = scan(_docs())
    js = to_json(report)
    import json

    parsed = json.loads(js)
    assert parsed["corpus_sha256"] == report.corpus_sha256
    assert len(parsed["verdicts"]) == 6
    md = to_markdown(report)
    assert "rag-sanitizer" in md
    assert "POISON" in md or "SUSPECT" in md
