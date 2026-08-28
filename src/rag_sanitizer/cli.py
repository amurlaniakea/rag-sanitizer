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

"""rag-sanitizer CLI (Typer)."""

from __future__ import annotations

import json

import typer

from .corpus import load_corpus
from .detectors.entity_swap import EntityProfile
from .embedder import DummyDeterministicEmbedder, SentenceTransformerEmbedder
from .report import print_report, write_report
from .scanner import scan

app = typer.Typer(help="Scan a RAG corpus for poison documents BEFORE ingestion.")


@app.command()
def scan_cmd(
    corpus: str = typer.Argument(..., help="Corpus dir, .jsonl, or .txt/.md file"),
    out: str | None = typer.Option(
        None, "--out", help="Write report to .json or .md (default: stdout)"
    ),
    k: float = typer.Option(3.0, "--k", help="Mimicry outlier multiplier (k*sigma)"),
    embedder: str = typer.Option(
        "dummy", "--embedder", help="Embedding backend: dummy (fast) | real (sentence-transformers)"
    ),
    multimodal: str = typer.Option(
        "heuristic", "--multimodal", help="Multimodal backend: heuristic (no deps) | clip (CLIP)"
    ),
    entity_profile: str | None = typer.Option(
        None,
        "--entity-profile",
        help='Path to JSON file with {"known": ["entity", ...]} (closes KI-7)',
    ),
) -> None:
    """Scan CORPUS and emit a verdict per document (CLEAN/SUSPECT/POISON)."""
    if embedder == "dummy":
        emb = DummyDeterministicEmbedder()
    elif embedder == "real":
        emb = SentenceTransformerEmbedder()
    else:
        typer.echo(f"unknown --embedder: {embedder}", err=True)
        raise typer.Exit(code=2)

    docs = load_corpus(corpus, embedder=emb)

    profile: EntityProfile | None = None
    if entity_profile:
        with open(entity_profile, encoding="utf-8") as fh:
            data = json.load(fh)
        profile = EntityProfile(known=set(data.get("known", [])))

    report = scan(docs, k=k, entity_profile=profile, multimodal_backend=multimodal)
    if out:
        write_report(report, out)
        typer.echo(f"Report written to {out}")
    else:
        print_report(report, typer.get_text_stream("stdout"))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
