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

"""Tests de regresión para los 2 bugs reportados por la auditoría de Claude.

BUG 1 (multimodal): el caption mismatch debe dispararse cuando la imagen EXISTE
pero el caption no tiene relación con el cuerpo del documento (no el caso trivial
de imagen faltante).

BUG 2 (entity_swap): el extractor de entidades no debe marcar prosa genérica con
variación de redacción (palabras de función capitalizadas) como entity-swap.
"""

import os
import tempfile

from rag_sanitizer.detectors.entity_swap import detect_entity_swap, extract_entities
from rag_sanitizer.detectors.multimodal import detect_multimodal


# BUG 1: caption mismatch con imagen EXISTENTE (no el caso de imagen faltante).
def test_bug1_caption_mismatch_image_present():
    with tempfile.TemporaryDirectory() as d:
        img = os.path.join(d, "chart.png")
        with open(img, "wb") as fh:
            fh.write(b"\x89PNG\r\n\x1a\n")  # PNG mínimo válido como bytes
        # Cuerpo benigno sobre finanzas; caption que NO tiene relación con el body.
        doc = (
            "The Q3 2025 revenue was $4.2M with 312 enterprise customers. "
            "Our headquarters is in Madrid, Spain.\n"
            "caption: see diagram of the volcanic eruption in Iceland"
        )
        res = detect_multimodal(doc, img)
        assert res.flagged is True
        assert "caption references content absent" in res.reason


# BUG 1 (caso límite): caption totalmente relacionado -> NO flag.
def test_bug1_caption_consistent_image_present():
    with tempfile.TemporaryDirectory() as d:
        img = os.path.join(d, "chart.png")
        with open(img, "wb") as fh:
            fh.write(b"\x89PNG\r\n\x1a\n")
        doc = (
            "The Q3 2025 revenue was $4.2M with 312 enterprise customers. "
            "Our headquarters is in Madrid, Spain.\n"
            "caption: Q3 2025 revenue 4.2M enterprise customers Madrid Spain"
        )
        res = detect_multimodal(doc, img)
        assert res.flagged is False


# BUG 1 (caso límite): caption parcialmente relacionado, cerca del umbral 0.5.
def test_bug1_caption_partial_near_threshold():
    with tempfile.TemporaryDirectory() as d:
        img = os.path.join(d, "chart.png")
        with open(img, "wb") as fh:
            fh.write(b"\x89PNG\r\n\x1a\n")
        # caption: "revenue Madrid" (2 tokens) vs body con ambos -> overlap 1.0 -> OK.
        doc = (
            "The Q3 2025 revenue was $4.2M with 312 enterprise customers in Madrid.\n"
            "caption: revenue Madrid"
        )
        assert detect_multimodal(doc, img).flagged is False
        # caption: "revenue volcano" (2 tokens) vs body solo "revenue" -> overlap 0.5 -> OK (>=0.5).
        doc2 = (
            "The Q3 2025 revenue was $4.2M with 312 enterprise customers in Madrid.\n"
            "caption: revenue volcano"
        )
        assert detect_multimodal(doc2, img).flagged is False
        # caption: "volcano iceberg hurricane" (3 tokens) vs body 0 -> overlap 0.0 -> flag.
        doc3 = (
            "The Q3 2025 revenue was $4.2M with 312 enterprise customers in Madrid.\n"
            "caption: volcano iceberg hurricane"
        )
        assert detect_multimodal(doc3, img).flagged is True


# BUG 2: prosa genérica con variación de redacción NO debe marcar entity-swap.
def test_bug2_generic_prose_no_false_positive():
    profile = [
        "The Q3 2025 revenue was $4.2M with 312 enterprise customers.",
        "Our headquarters is in Madrid, Spain. The board approved the 2026 roadmap.",
    ]
    doc_a = "This report explains the quarterly results and how the market changed."
    doc_b = "We analyzed the data and found that the performance improved over time."
    assert detect_entity_swap(doc_a, profile).flagged is False
    assert detect_entity_swap(doc_b, profile).flagged is False
    # Además: el extractor no debe producir ninguna "org:" sobre palabras de función.
    ents_a = extract_entities(doc_a)
    ents_b = extract_entities(doc_b)
    assert all(not k.startswith("org:") or "the" not in k.lower() for k in ents_a)
    assert all(not k.startswith("org:") or "we" not in k.lower() for k in ents_b)


# BUG 2 (regresión): sigue detectando entity-swap REAL (sigla/nombre propio nuevo).
def test_bug2_real_entity_swap_still_detected():
    profile = [
        "The Q3 2025 revenue was $4.2M with 312 enterprise customers.",
        "Our headquarters is in Madrid, Spain.",
    ]
    # "Orion Holdings" (2 palabras capitalizadas) es una org real ausente en el perfil.
    doc = (
        "The Q3 2025 revenue was $4.2M with 312 enterprise customers. "
        "Our headquarters is in Madrid, Spain after a $900M acquisition by Orion Holdings."
    )
    res = detect_entity_swap(doc, profile)
    assert res.flagged is True
    assert any("org:Orion Holdings" in e for e in res.novel_entities)
