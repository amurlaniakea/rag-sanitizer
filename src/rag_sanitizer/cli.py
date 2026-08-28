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

"""rag-sanitizer CLI (Typer)."""

from __future__ import annotations

import typer

from .corpus import load_corpus
from .embedder import DummyDeterministicEmbedder
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
) -> None:
    """Scan CORPUS and emit a verdict per document (CLEAN/SUSPECT/POISON)."""
    docs = load_corpus(corpus, embedder=DummyDeterministicEmbedder())
    report = scan(docs, k=k)
    if out:
        write_report(report, out)
        typer.echo(f"Report written to {out}")
    else:
        print_report(report, typer.get_text_stream("stdout"))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
