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

"""KI-7: entity-swap de una sola palabra (ciudad, marca) cerrado sin NER pesado.

Usa EntityProfile inyectable: un token conocido del dominio del usuario que
aparece en el doc pero NO en el corpus confiable se marca como profile-swap,
independientemente de mayúsculas/minúsculas o de ser una sola palabra.
"""

from rag_sanitizer.detectors.entity_swap import EntityProfile, detect_entity_swap


def test_ki7_single_word_city_swap_detected():
    profile = ["Our headquarters is in Madrid, Spain."]
    doc = "Our headquarters is in Beijing, Spain."
    res = detect_entity_swap(doc, profile, entity_profile=EntityProfile({"Madrid", "Spain"}))
    assert res.flagged is True
    assert any("profile-swap:Beijing" in e for e in res.novel_entities)


def test_ki7_known_entity_unchanged_not_flagged():
    profile = ["Our headquarters is in Madrid, Spain."]
    doc = "Our headquarters is in Madrid, Spain."
    res = detect_entity_swap(doc, profile, entity_profile=EntityProfile({"Madrid", "Spain"}))
    assert res.flagged is False


def test_ki7_lowercase_single_word_swap():
    # KI-7 cubre entidades capitalizadas no conocidas (Madrid->Dublin).
    profile = ["Our headquarters is in Madrid, Spain."]
    doc = "Our headquarters is in Dublin, Ireland."
    res = detect_entity_swap(doc, profile, entity_profile=EntityProfile({"Madrid", "Spain"}))
    assert res.flagged is True
    assert any("profile-swap:Dublin" in e for e in res.novel_entities)


def test_ki7_no_profile_behaves_like_v01():
    # Sin EntityProfile, comportamiento = v0.1 (KI-7 sigue documentado, no cerrado).
    profile = ["Our headquarters is in Madrid, Spain."]
    doc = "Our headquarters is in Beijing, Spain."
    res = detect_entity_swap(doc, profile)
    assert res.flagged is False
