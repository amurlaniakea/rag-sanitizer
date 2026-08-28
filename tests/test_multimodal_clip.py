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

"""KI-2: CLIP real (texto+imagen) detecta visual poison. Marca `real_embeddings`.

Se salta si el extra `multimodal` (transformers+torch) no está instalado. Genera
una imagen sintética en disco (texto vs imagen discordantes) para probar el backend.
"""

import os
import tempfile

import pytest

from rag_sanitizer.detectors.multimodal import ClipMultimodalDetector, HeuristicMultimodalDetector

pytestmark = pytest.mark.real_embeddings


def _have_clip() -> bool:
    try:
        import torch  # noqa: F401
        from transformers import CLIPModel  # noqa: F401

        return True
    except ImportError:
        return False


def _make_png(path: str, color: tuple[int, int, int]) -> None:
    from PIL import Image

    Image.new("RGB", (32, 32), color).save(path, "PNG")


@pytest.mark.skipif(not _have_clip(), reason="transformers/torch not installed (extra multimodal)")
def test_clip_flags_text_image_mismatch():
    with tempfile.TemporaryDirectory() as d:
        # Imagen roja; texto habla de "volcanic eruption" -> baja similitud CLIP.
        img = os.path.join(d, "red.png")
        _make_png(img, (255, 0, 0))
        doc = "The volcanic eruption in Iceland released ash across the north atlantic."
        # Uses the recalibrated default threshold (0.25); mismatch scores ~0.20, below it.
        det = ClipMultimodalDetector()
        res = det.detect(doc, img)
        assert res.flagged is True
        assert res.score is not None


@pytest.mark.skipif(not _have_clip(), reason="transformers/torch not installed (extra multimodal)")
def test_clip_does_not_flag_text_image_match():
    # Regression for the threshold-recalibration bug: a legitimate text-image match must
    # NOT be flagged. With the default threshold 0.25, matches score ~0.27-0.32 (measured),
    # above the threshold. This is the positive case the mismatch-only test missed.
    with tempfile.TemporaryDirectory() as d:
        img = os.path.join(d, "red.png")
        _make_png(img, (200, 30, 30))
        doc = "a red square on a white background"
        det = ClipMultimodalDetector()
        res = det.detect(doc, img)
        assert res.flagged is False
        assert res.score is not None


@pytest.mark.skipif(not _have_clip(), reason="transformers/torch not installed (extra multimodal)")
def test_heuristic_fallback_still_works():
    # Regresión: el heurístico v0.1 sigue disponible y marca imagen faltante.
    with tempfile.TemporaryDirectory() as d:
        doc = "caption: see chart of the volcanic eruption in Iceland"
        res = HeuristicMultimodalDetector().detect(doc, os.path.join(d, "missing.png"))
        assert res.flagged is True
