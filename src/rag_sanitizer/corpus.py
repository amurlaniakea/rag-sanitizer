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

"""Corpus loading: turn a directory / JSONL of documents into ``Document`` objects."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from dataclasses import dataclass, field

from .embedder import DummyDeterministicEmbedder, Embedder


@dataclass
class Document:
    id: str
    text: str
    image_path: str | None = None
    embedding: list[float] = field(default_factory=list)
    trust: str = "unknown"  # "clean" | "poison" | "unknown" (ground truth for tests)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


def _read_text(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _load_jsonl(path: str) -> list[dict]:
    out: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_corpus(
    source: str,
    embedder: Embedder | None = None,
    trust_field: str = "trust",
) -> list[Document]:
    """Load a corpus from a directory, a ``.jsonl`` file, or a single text file.

    - directory: every ``*.txt``/``*.md`` becomes a Document (id = filename).
    - ``.jsonl``: each line ``{id, text, [image_path], [trust]}``.
    - single ``.txt``/``*.md``: one Document.
    """
    emb = embedder or DummyDeterministicEmbedder()
    docs: list[Document] = []

    if os.path.isdir(source):
        for name in sorted(os.listdir(source)):
            if name.endswith((".txt", ".md")):
                text = _read_text(os.path.join(source, name))
                docs.append(Document(id=name, text=text, embedding=emb.embed(text)))
        return docs

    if source.endswith(".jsonl"):
        for i, row in enumerate(_load_jsonl(source)):
            text = row["text"]
            docs.append(
                Document(
                    id=row.get("id", str(i)),
                    text=text,
                    image_path=row.get("image_path"),
                    embedding=emb.embed(text),
                    trust=row.get(trust_field, "unknown"),
                )
            )
        return docs

    if source.endswith((".txt", ".md")):
        text = _read_text(source)
        docs.append(Document(id=os.path.basename(source), text=text, embedding=emb.embed(text)))
        return docs

    raise ValueError(f"Unsupported corpus source: {source}")


def corpus_sha256(docs: Iterable[Document]) -> str:
    h = hashlib.sha256()
    for d in docs:
        h.update(d.id.encode("utf-8"))
        h.update(d.text.encode("utf-8"))
    return h.hexdigest()
