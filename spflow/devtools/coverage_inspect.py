"""Inspect coverage HTML reports for missed lines.

This module provides a small CLI that:
1) Lists per-file coverage from ``htmlcov/index.html``.
2) Shows missed line chunks with configurable context from either:
   - a per-file coverage HTML page (``htmlcov/z_*.html``), or
   - a source path (``spflow/...py``) resolved via the index.

When available, it prefers the ``.coverage`` data file for computing missed
lines. If coverage data isn't available, it falls back to parsing the per-file
HTML report.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable, Iterator, Sequence


@dataclass(frozen=True, slots=True)
class CoverageIndexRow:
    """A single row from coverage.py's HTML index page."""

    file_path: str
    statements: int
    missing: int
    excluded: int
    coverage_percent: int
    html_href: str

    @property
    def covered(self) -> int:
        """Return the number of statements covered."""
        return max(0, self.statements - self.missing)


def _normalize_rel_path(path: Path, repo_root: Path) -> str:
    """Normalize a path to a repo-root relative POSIX string when possible."""
    try:
        rel = path.resolve().relative_to(repo_root.resolve())
        return rel.as_posix()
    except Exception:
        return path.as_posix()


class _CoverageIndexParser(HTMLParser):
    """Parse coverage.py ``index.html`` into rows.

    The coverage index table has rows like:
      <tr class="region">
        <td><a href="z_....html">spflow/foo.py</a></td>
        <td>...</td> ...
      </tr>
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._in_index_table = False
        self._in_tbody = False
        self._in_row = False
        self._current_cells: list[str] = []
        self._current_href: str | None = None
        self._in_first_cell_link = False
        self._link_text_parts: list[str] = []
        self.rows: list[CoverageIndexRow] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        if tag == "table" and attrs_dict.get("class") == "index":
            self._in_index_table = True
            return

        if not self._in_index_table:
            return

        if tag == "tbody":
            self._in_tbody = True
            return

        if not self._in_tbody:
            return

        if tag == "tr":
            cls = attrs_dict.get("class", "") or ""
            # coverage.py uses class="region" for file rows.
            if "region" in cls.split():
                self._in_row = True
                self._current_cells = []
                self._current_href = None
                self._in_first_cell_link = False
                self._link_text_parts = []
            return

        if not self._in_row:
            return

        if tag == "a" and self._current_href is None:
            href = attrs_dict.get("href")
            if href is not None:
                self._current_href = href
                self._in_first_cell_link = True
            return

    def handle_endtag(self, tag: str) -> None:
        if tag == "table" and self._in_index_table:
            self._in_index_table = False
            return

        if not self._in_index_table:
            return

        if tag == "tbody" and self._in_tbody:
            self._in_tbody = False
            return

        if not self._in_tbody:
            return

        if tag == "a" and self._in_first_cell_link:
            link_text = "".join(self._link_text_parts).strip()
            if link_text:
                self._current_cells.append(link_text)
            self._in_first_cell_link = False
            self._link_text_parts = []
            return

        if tag == "tr" and self._in_row:
            self._in_row = False
            self._finalize_row()
            return

    def handle_data(self, data: str) -> None:
        if not self._in_index_table or not self._in_tbody or not self._in_row:
            return

        text = data.strip()
        if not text:
            return

        if self._in_first_cell_link:
            self._link_text_parts.append(text)
            return

        self._current_cells.append(text)

    def _finalize_row(self) -> None:
        if self._current_href is None:
            return
        if len(self._current_cells) < 5:
            return
        file_path = self._current_cells[0]
        try:
            statements = int(self._current_cells[1])
            missing = int(self._current_cells[2])
            excluded = int(self._current_cells[3])
            coverage_text = self._current_cells[4].strip().rstrip("%")
            coverage_percent = int(float(coverage_text))
        except ValueError:
            return

        self.rows.append(
            CoverageIndexRow(
                file_path=file_path,
                statements=statements,
                missing=missing,
                excluded=excluded,
                coverage_percent=coverage_percent,
                html_href=self._current_href,
            )
        )


def parse_coverage_index(index_html: str) -> list[CoverageIndexRow]:
    """Parse the HTML coverage index page content into rows."""
    parser = _CoverageIndexParser()
    parser.feed(index_html)
    parser.close()
    return parser.rows


class _CoverageFileParser(HTMLParser):
    """Parse coverage.py per-file HTML to get missed lines and optional text."""

    def __init__(self, include_partial: bool) -> None:
        super().__init__(convert_charrefs=True)
        self._include_partial = include_partial

        self._in_line_p = False
        self._line_is_missed = False
        self._line_is_partial = False
        self._in_line_number = False
        self._current_line_no: int | None = None
        self._in_text_span = False
        self._current_text_parts: list[str] = []

        self.missed_lines: set[int] = set()
        self.partial_lines: set[int] = set()
        self.line_text: dict[int, str] = {}
        self.source_path: str | None = None

        self._in_header_b = False
        self._header_b_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        if tag == "b" and self.source_path is None:
            self._in_header_b = True
            self._header_b_parts = []
            return

        if tag != "p":
            if self._in_line_p and tag == "span" and attrs_dict.get("class") == "t":
                self._in_text_span = True
                self._current_text_parts = []
            elif self._in_line_p and tag == "a" and attrs_dict.get("id", "").startswith("t"):
                self._in_line_number = True
            return

        cls = attrs_dict.get("class", "") or ""
        classes = set(cls.split())
        self._in_line_p = True
        self._line_is_missed = "mis" in classes
        self._line_is_partial = "par" in classes
        self._current_line_no = None
        self._in_line_number = False
        self._in_text_span = False
        self._current_text_parts = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "b" and self._in_header_b:
            self._in_header_b = False
            if self.source_path is None:
                text = "".join(self._header_b_parts).strip()
                if text:
                    self.source_path = text
            return

        if not self._in_line_p:
            return

        if tag == "a" and self._in_line_number:
            self._in_line_number = False
            return

        if tag == "span" and self._in_text_span:
            self._in_text_span = False
            if self._current_line_no is not None and self._current_text_parts:
                txt = "".join(self._current_text_parts).replace("\xa0", " ")
                self.line_text[self._current_line_no] = txt.rstrip("\n")
            return

        if tag == "p":
            if self._current_line_no is not None:
                if self._line_is_missed:
                    self.missed_lines.add(self._current_line_no)
                if self._line_is_partial:
                    self.partial_lines.add(self._current_line_no)
            self._in_line_p = False
            self._line_is_missed = False
            self._line_is_partial = False
            self._current_line_no = None
            self._current_text_parts = []
            return

    def handle_data(self, data: str) -> None:
        if self._in_header_b:
            self._header_b_parts.append(data)
            return

        if not self._in_line_p:
            return

        if self._in_line_number:
            text = data.strip()
            if text.isdigit():
                self._current_line_no = int(text)
            return

        if self._in_text_span:
            self._current_text_parts.append(data)


@dataclass(frozen=True, slots=True)
class CoverageFileParseResult:
    """Result of parsing a per-file coverage HTML page."""

    source_path: str | None
    missed_lines: set[int]
    partial_lines: set[int]
    line_text: dict[int, str]


def parse_coverage_file_html(file_html: str, include_partial: bool = False) -> CoverageFileParseResult:
    """Parse coverage.py per-file HTML page content."""
    parser = _CoverageFileParser(include_partial=include_partial)
    parser.feed(file_html)
    parser.close()
    return CoverageFileParseResult(
        source_path=parser.source_path,
        missed_lines=parser.missed_lines,
        partial_lines=parser.partial_lines,
        line_text=parser.line_text,
    )


def _try_import_coverage() -> object | None:
    try:
        import coverage  # type: ignore[import-not-found]

        return coverage
    except Exception:
        return None


def _find_best_measured_file(measured: Sequence[str], abs_source: str) -> str | None:
    if abs_source in measured:
        return abs_source
    # Prefer longest suffix match (most specific).
    target_suffix = abs_source.replace("\\", "/")
    candidates: list[str] = []
    for m in measured:
        m_norm = m.replace("\\", "/")
        if m_norm.endswith(target_suffix):
            candidates.append(m)
            continue
        # Also match by repo-relative path when abs_source is within the repo.
        if target_suffix.endswith(m_norm):
            candidates.append(m)
    if not candidates:
        # Try matching by relative suffix (e.g. spflow/foo.py).
        suffix_parts = Path(abs_source).as_posix()
        for m in measured:
            if m.replace("\\", "/").endswith(suffix_parts):
                candidates.append(m)
    if not candidates:
        return None
    return max(candidates, key=lambda p: len(p))


def get_missing_lines_from_coverage(source_path: Path, coverage_data_path: Path) -> set[int] | None:
    """Return missing executable line numbers from a ``.coverage`` data file.

    Returns ``None`` if the coverage library isn't available or data can't be read.
    """
    coverage_mod = _try_import_coverage()
    if coverage_mod is None:
        return None
    if not coverage_data_path.exists():
        return None

    cov = coverage_mod.Coverage(data_file=str(coverage_data_path))
    cov.load()

    abs_source = str(source_path.resolve())
    try:
        _, _, _, missing, _ = cov.analysis2(abs_source)
        return set(int(n) for n in missing)
    except Exception:
        pass

    try:
        data = cov.get_data()
        measured_files = sorted(data.measured_files())
        best = _find_best_measured_file(measured_files, abs_source)
        if best is None:
            return None
        _, _, _, missing, _ = cov.analysis2(best)
        return set(int(n) for n in missing)
    except Exception:
        return None


def group_contiguous_lines(lines: Iterable[int]) -> list[tuple[int, int]]:
    """Group line numbers into contiguous inclusive ranges."""
    sorted_lines = sorted(set(int(n) for n in lines))
    if not sorted_lines:
        return []
    ranges: list[tuple[int, int]] = []
    start = prev = sorted_lines[0]
    for n in sorted_lines[1:]:
        if n == prev + 1:
            prev = n
            continue
        ranges.append((start, prev))
        start = prev = n
    ranges.append((start, prev))
    return ranges


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _read_source_lines(source_path: Path) -> list[str] | None:
    try:
        text = _read_text_file(source_path)
    except Exception:
        return None
    return text.splitlines()


def render_chunks_text(
    *,
    display_name: str,
    source_lines: Sequence[str] | None,
    html_line_text: dict[int, str],
    missed_lines: set[int],
    context: int,
    max_chunks: int,
) -> str:
    """Render missed line chunks in a readable text format."""
    chunks = group_contiguous_lines(missed_lines)[:max_chunks]
    if not chunks:
        return f"{display_name} — missing 0 lines\n"

    total_missing = len(missed_lines)
    out_lines: list[str] = [f"{display_name} — missing {total_missing} lines"]

    def get_line_text(line_no: int) -> str:
        if source_lines is not None and 1 <= line_no <= len(source_lines):
            return source_lines[line_no - 1]
        return html_line_text.get(line_no, "").rstrip("\n")

    max_line_no = 0
    if source_lines is not None:
        max_line_no = len(source_lines)
    if html_line_text:
        max_line_no = max(max_line_no, max(html_line_text.keys()))

    width = max(4, len(str(max_line_no if max_line_no > 0 else max(missed_lines))))
    for start, end in chunks:
        out_lines.append(f"--- missed {start}-{end} (context {context}) ---")
        lo = max(1, start - context)
        hi = min(max_line_no if max_line_no > 0 else end + context, end + context)
        for ln in range(lo, hi + 1):
            prefix = "!" if ln in missed_lines else " "
            text = get_line_text(ln)
            out_lines.append(f"{prefix}{ln:{width}d}: {text}")
    out_lines.append("")
    return "\n".join(out_lines)


def render_chunks_markdown(
    *,
    display_name: str,
    source_lines: Sequence[str] | None,
    html_line_text: dict[int, str],
    missed_lines: set[int],
    context: int,
    max_chunks: int,
) -> str:
    """Render missed line chunks in markdown code blocks."""
    chunks = group_contiguous_lines(missed_lines)[:max_chunks]
    if not chunks:
        return f"**{display_name}** — missing 0 lines\n"

    total_missing = len(missed_lines)
    out_lines: list[str] = [f"**{display_name}** — missing {total_missing} lines"]

    def get_line_text(line_no: int) -> str:
        if source_lines is not None and 1 <= line_no <= len(source_lines):
            return source_lines[line_no - 1]
        return html_line_text.get(line_no, "").rstrip("\n")

    max_line_no = 0
    if source_lines is not None:
        max_line_no = len(source_lines)
    if html_line_text:
        max_line_no = max(max_line_no, max(html_line_text.keys()))
    width = max(4, len(str(max_line_no if max_line_no > 0 else max(missed_lines))))

    for start, end in chunks:
        out_lines.append(f"\n_Missed {start}-{end} (context {context})_")
        out_lines.append("```")
        lo = max(1, start - context)
        hi = min(max_line_no if max_line_no > 0 else end + context, end + context)
        for ln in range(lo, hi + 1):
            prefix = "!" if ln in missed_lines else " "
            out_lines.append(f"{prefix}{ln:{width}d}: {get_line_text(ln)}")
        out_lines.append("```")
    out_lines.append("")
    return "\n".join(out_lines)


def _load_index_rows(index_path: Path) -> list[CoverageIndexRow]:
    if not index_path.exists():
        raise FileNotFoundError(f"Coverage index not found: {index_path}")
    return parse_coverage_index(_read_text_file(index_path))


def _resolve_source_to_href(rows: Sequence[CoverageIndexRow], source: str) -> str | None:
    # Exact match against the file_path displayed in index.html.
    for row in rows:
        if row.file_path == source:
            return row.html_href
    # Normalize slashes and try suffix matching.
    source_norm = source.replace("\\", "/")
    for row in rows:
        if row.file_path.replace("\\", "/").endswith(source_norm):
            return row.html_href
    return None


def _resolve_href_to_source(rows: Sequence[CoverageIndexRow], href: str) -> str | None:
    for row in rows:
        if row.html_href == href:
            return row.file_path
    return None


def cmd_list(args: argparse.Namespace) -> int:
    htmlcov_dir = Path(args.htmlcov_dir)
    index_path = Path(args.index) if args.index else htmlcov_dir / "index.html"
    rows = _load_index_rows(index_path)

    if args.path_prefix:
        rows = [r for r in rows if r.file_path.startswith(args.path_prefix)]

    rows = [r for r in rows if r.missing >= args.min_missing]

    sort_key = args.sort
    if sort_key == "coverage":
        rows.sort(key=lambda r: (r.coverage_percent, -r.missing, -r.statements, r.file_path))
    elif sort_key == "missing":
        rows.sort(key=lambda r: (-r.missing, r.coverage_percent, -r.statements, r.file_path))
    elif sort_key == "ratio":
        rows.sort(
            key=lambda r: (r.coverage_percent, -r.missing, -(r.missing / max(1, r.statements)), r.file_path)
        )
    else:
        raise ValueError(f"Unknown sort: {sort_key}")

    rows = rows[: args.limit]

    headers = ["cov%", "miss", "stmt", "file", "html"]
    table: list[list[str]] = [headers]
    for r in rows:
        table.append(
            [
                f"{r.coverage_percent:>3d}",
                str(r.missing),
                str(r.statements),
                r.file_path,
                str((htmlcov_dir / r.html_href).as_posix()),
            ]
        )

    col_widths = [max(len(row[i]) for row in table) for i in range(len(headers))]
    lines: list[str] = []
    for idx, row in enumerate(table):
        parts = []
        for i, cell in enumerate(row):
            if i in (0, 1, 2):
                parts.append(cell.rjust(col_widths[i]))
            else:
                parts.append(cell.ljust(col_widths[i]))
        lines.append("  ".join(parts))
        if idx == 0:
            lines.append("  ".join("-" * w for w in col_widths))
    sys.stdout.write("\n".join(lines) + "\n")
    return 0


def _resolve_target_to_paths(
    *,
    target: str,
    htmlcov_dir: Path,
    index_path: Path,
    repo_root: Path,
) -> tuple[Path, str]:
    """Return (per_file_html_path, display_source_path)."""
    target_path = Path(target)
    rows = _load_index_rows(index_path)

    if target_path.suffix.lower() == ".html":
        html_path = target_path if target_path.is_absolute() else (repo_root / target_path)
        if not html_path.exists():
            # Also try relative to htmlcov_dir.
            html_path2 = htmlcov_dir / target_path.name
            if html_path2.exists():
                html_path = html_path2
            else:
                raise FileNotFoundError(f"Coverage HTML file not found: {target}")
        href = os.path.relpath(html_path, start=htmlcov_dir).replace("\\", "/")
        src = _resolve_href_to_source(rows, href) or target_path.name
        return html_path, src

    # Source path: resolve to a per-file HTML page via index.
    rel_source = _normalize_rel_path(target_path, repo_root)
    href = _resolve_source_to_href(rows, rel_source) or _resolve_source_to_href(rows, target)
    if href is None:
        raise FileNotFoundError(f"Could not resolve source path in index: {target}")
    return htmlcov_dir / href, rel_source


def cmd_show(args: argparse.Namespace) -> int:
    repo_root = Path.cwd()
    htmlcov_dir = Path(args.htmlcov_dir)
    index_path = Path(args.index) if args.index else htmlcov_dir / "index.html"
    coverage_data_path = Path(args.coverage_data)

    html_path, display_source = _resolve_target_to_paths(
        target=args.target,
        htmlcov_dir=htmlcov_dir,
        index_path=index_path,
        repo_root=repo_root,
    )
    file_html = _read_text_file(html_path)
    parsed = parse_coverage_file_html(file_html, include_partial=args.include_partial)

    # Determine source path for reading the file from disk.
    source_path = Path(display_source)
    if not source_path.exists() and parsed.source_path is not None:
        source_path = Path(parsed.source_path)

    missed_lines: set[int] = set()
    missing_from_cov = get_missing_lines_from_coverage(source_path, coverage_data_path)
    if missing_from_cov is not None:
        missed_lines = missing_from_cov
    else:
        missed_lines = set(parsed.missed_lines)
        if args.include_partial:
            missed_lines |= set(parsed.partial_lines)

    source_lines = _read_source_lines(source_path)
    display_name = display_source
    if parsed.source_path is not None:
        display_name = parsed.source_path

    if args.format == "markdown":
        rendered = render_chunks_markdown(
            display_name=display_name,
            source_lines=source_lines,
            html_line_text=parsed.line_text,
            missed_lines=missed_lines,
            context=args.context,
            max_chunks=args.max_chunks,
        )
    else:
        rendered = render_chunks_text(
            display_name=display_name,
            source_lines=source_lines,
            html_line_text=parsed.line_text,
            missed_lines=missed_lines,
            context=args.context,
            max_chunks=args.max_chunks,
        )
    sys.stdout.write(rendered)
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="coverage_inspect",
        description="Inspect coverage.py HTML reports for low coverage and missed lines.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    list_p = sub.add_parser("list", help="List coverage HTML files from htmlcov/index.html.")
    list_p.add_argument("--htmlcov-dir", default="htmlcov", help="Path to htmlcov directory.")
    list_p.add_argument(
        "--index",
        default=None,
        help="Path to coverage index HTML (default: <htmlcov-dir>/index.html).",
    )
    list_p.add_argument("--limit", type=int, default=50, help="Maximum rows to show.")
    list_p.add_argument("--min-missing", type=int, default=1, help="Minimum missing statements to include.")
    list_p.add_argument("--path-prefix", default=None, help="Only include files starting with this prefix.")
    list_p.add_argument(
        "--sort",
        choices=("coverage", "missing", "ratio"),
        default="coverage",
        help="Sort order for the list output.",
    )
    list_p.set_defaults(_handler=cmd_list)

    show_p = sub.add_parser("show", help="Show missed line chunks for a file.")
    show_p.add_argument(
        "target", help="A source path (spflow/foo.py) or per-file HTML page (htmlcov/z_*.html)."
    )
    show_p.add_argument("--htmlcov-dir", default="htmlcov", help="Path to htmlcov directory.")
    show_p.add_argument(
        "--index",
        default=None,
        help="Path to coverage index HTML (default: <htmlcov-dir>/index.html).",
    )
    show_p.add_argument("--coverage-data", default=".coverage", help="Path to .coverage file.")
    show_p.add_argument(
        "--context", type=int, default=3, help="Context lines before and after missed chunks."
    )
    show_p.add_argument("--max-chunks", type=int, default=50, help="Maximum missed chunks to show.")
    show_p.add_argument(
        "--include-partial",
        action="store_true",
        help="Include partially-covered lines (branch/partial coverage) when parsing HTML.",
    )
    show_p.add_argument("--format", choices=("text", "markdown"), default="text", help="Output format.")
    show_p.set_defaults(_handler=cmd_show)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        handler = args._handler  # type: ignore[attr-defined]
        return int(handler(args))
    except (FileNotFoundError, ValueError) as e:
        parser.error(str(e))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
