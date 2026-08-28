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

"""ScanReport serialization to JSON and Markdown."""

from __future__ import annotations

import json
from typing import IO

from .scanner import ScanReport


def to_json(report: ScanReport) -> str:
    payload = {
        "corpus_sha256": report.corpus_sha256,
        "summary": report.summary,
        "verdicts": [
            {"id": v.id, "verdict": v.verdict, "reasons": v.reasons} for v in report.verdicts
        ],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def to_markdown(report: ScanReport) -> str:
    lines = ["# rag-sanitizer — Scan Report", ""]
    lines.append(f"- corpus_sha256: `{report.corpus_sha256}`")
    lines.append(f"- summary: {report.summary}")
    lines.append("")
    for v in report.verdicts:
        lines.append(f"## {v.id} — **{v.verdict}**")
        if v.reasons:
            for r in v.reasons:
                lines.append(f"  - {r}")
        else:
            lines.append("  - (no signals)")
        lines.append("")
    return "\n".join(lines)


def write_report(report: ScanReport, out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as fh:
        if out_path.endswith(".json"):
            fh.write(to_json(report))
        else:
            fh.write(to_markdown(report))


def print_report(report: ScanReport, fh: IO[str]) -> None:
    fh.write(to_markdown(report))
