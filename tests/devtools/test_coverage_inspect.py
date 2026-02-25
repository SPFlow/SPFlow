from __future__ import annotations

import argparse
import runpy
from pathlib import Path

import pytest

import spflow.devtools.coverage_inspect as coverage_inspect
from spflow.devtools.coverage_inspect import (
    get_missing_lines_from_coverage,
    group_contiguous_lines,
    parse_coverage_file_html,
    parse_coverage_index,
    render_chunks_markdown,
    render_chunks_text,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_parse_coverage_index_fixture() -> None:
    html = (FIXTURES_DIR / "coverage_index_min.html").read_text(encoding="utf-8")
    rows = parse_coverage_index(html)
    assert len(rows) == 2
    assert rows[0].file_path == "spflow/exceptions.py"
    assert rows[0].statements == 13
    assert rows[0].missing == 2
    assert rows[0].coverage_percent == 85
    assert rows[0].html_href == "z_aaa_spflow_exceptions_py.html"


def test_parse_coverage_file_html_fixture() -> None:
    html = (FIXTURES_DIR / "coverage_file_min.html").read_text(encoding="utf-8")
    res = parse_coverage_file_html(html)
    assert res.source_path == "spflow/exceptions.py"
    assert res.missed_lines == {2, 3}
    assert res.line_text[2] == "missed2"


def test_parse_coverage_file_html_include_partial_and_header_once() -> None:
    html = """
    <html><body>
      <h1><b>spflow/a.py</b><b>ignored-second-header</b></h1>
      <p class="par run"><span class="n"><a id="t2">2</a></span><span class="t">p\xa0line</span></p>
      <p class="mis"><span class="n"><a id="t3">3</a></span><span class="t">missed</span></p>
    </body></html>
    """
    res = parse_coverage_file_html(html, include_partial=True)
    assert res.source_path == "spflow/a.py"
    assert res.partial_lines == {2}
    assert res.missed_lines == {3}
    assert res.line_text[2] == "p line"


def test_parse_coverage_index_ignores_non_region_and_invalid_cells() -> None:
    html = """
    <html><body>
      <table class="index"><tbody>
        <tr><td><a href="x.html">ignore/not-region.py</a></td><td>1</td><td>0</td><td>0</td><td>100%</td></tr>
        <tr class="region"><td><a href="bad.html">bad.py</a></td><td>abc</td><td>1</td><td>0</td><td>50%</td></tr>
        <tr class="region"><td><a href="short.html">short.py</a></td><td>1</td></tr>
        <tr class="region"><td><a href="ok.html">ok.py</a></td><td>10</td><td>2</td><td>0</td><td>80%</td></tr>
      </tbody></table>
    </body></html>
    """
    rows = parse_coverage_index(html)
    assert [r.file_path for r in rows] == ["ok.py"]
    assert rows[0].covered == 8


def test_parse_coverage_index_handles_non_tbody_tags_and_missing_href() -> None:
    html = """
    <html><body>
      <table class="index">
        <thead>
          <tr class="region"><td><a href="head.html">ignore.py</a></td><td>1</td><td>0</td><td>0</td><td>100%</td></tr>
        </thead>
        <tbody>
          <tr class="region"><td>no-link.py</td><td>3</td><td>1</td><td>0</td><td>66%</td></tr>
        </tbody>
      </table>
    </body></html>
    """
    # Parser intentionally trusts tbody region rows with linked file pages only.
    assert parse_coverage_index(html) == []


def test_group_contiguous_lines() -> None:
    assert group_contiguous_lines([]) == []
    assert group_contiguous_lines([3, 2, 10, 11, 12]) == [(2, 3), (10, 12)]
    assert group_contiguous_lines([1, 1, "2"]) == [(1, 2)]


def test_render_chunks_text_context(tmp_path: Path) -> None:
    src = tmp_path / "x.py"
    src.write_text("a\nb\nc\nd\ne\n", encoding="utf-8")
    output = render_chunks_text(
        display_name="x.py",
        source_lines=src.read_text(encoding="utf-8").splitlines(),
        html_line_text={},
        missed_lines={3},
        context=1,
        max_chunks=50,
    )
    assert "missed 3-3" in output
    assert "!   3: c" in output
    assert "    2: b" in output
    assert "    4: d" in output


def test_render_chunks_text_and_markdown_zero_and_html_fallback() -> None:
    text_empty = render_chunks_text(
        display_name="x.py",
        source_lines=None,
        html_line_text={},
        missed_lines=set(),
        context=1,
        max_chunks=10,
    )
    md_empty = render_chunks_markdown(
        display_name="x.py",
        source_lines=None,
        html_line_text={},
        missed_lines=set(),
        context=1,
        max_chunks=10,
    )
    assert "missing 0 lines" in text_empty
    assert "missing 0 lines" in md_empty

    md = render_chunks_markdown(
        display_name="x.py",
        source_lines=None,
        html_line_text={7: "from html"},
        missed_lines={7},
        context=1,
        max_chunks=10,
    )
    assert "_Missed 7-7 (context 1)_" in md
    assert "!   7: from html" in md


def test_get_missing_lines_from_coverage(tmp_path: Path) -> None:
    coverage = pytest.importorskip("coverage")

    mod = tmp_path / "m.py"
    # Keep one branch unexecuted so coverage must report a missed line.
    mod.write_text(
        "\n".join(
            [
                "def f(x: bool) -> int:",
                "    if x:",
                "        return 1",
                "    return 0",
                "",
                "if __name__ == '__main__':",
                "    f(True)",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    data_file = tmp_path / ".coverage"
    cov = coverage.Coverage(data_file=str(data_file))
    cov.start()
    runpy.run_path(str(mod), run_name="__main__")
    cov.stop()
    cov.save()

    missing = get_missing_lines_from_coverage(mod, data_file)
    assert missing is not None
    # Regression check: line mapping must point to the unvisited else branch.
    assert 4 in missing


def test_get_missing_lines_from_coverage_when_module_or_data_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "a.py"
    src.write_text("x = 1\n", encoding="utf-8")

    # Optional dependency path: callers should degrade to HTML-only mode cleanly.
    monkeypatch.setattr(coverage_inspect, "_try_import_coverage", lambda: None)
    assert get_missing_lines_from_coverage(src, tmp_path / ".coverage") is None


def test_try_import_coverage_returns_none_when_import_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    orig_import = builtins.__import__

    def _failing_import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: ANN001, ANN202
        if name == "coverage":
            raise RuntimeError("boom")
        return orig_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _failing_import)
    assert coverage_inspect._try_import_coverage() is None


def test_get_missing_lines_from_coverage_uses_measured_file_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "pkg" / "a.py"
    src.parent.mkdir(parents=True)
    src.write_text("x = 1\n", encoding="utf-8")
    data_file = tmp_path / ".coverage"
    data_file.write_text("stub", encoding="utf-8")
    source_abs = str(src.resolve())
    # Coverage sometimes records execution from another workspace root.
    measured_best = f"/other/root/{src.as_posix()}"

    class FakeData:
        def measured_files(self) -> list[str]:
            return ["/short/a.py", measured_best]

    class FakeCov:
        def __init__(self, *, data_file: str) -> None:
            assert data_file

        def load(self) -> None:
            return None

        def analysis2(self, path: str):  # noqa: ANN202
            if path == source_abs:
                raise RuntimeError("force fallback")
            if path == measured_best:
                return path, None, None, [5, 8], None
            raise RuntimeError("unexpected path")

        def get_data(self) -> FakeData:
            return FakeData()

    class FakeCoverageModule:
        Coverage = FakeCov

    monkeypatch.setattr(coverage_inspect, "_try_import_coverage", lambda: FakeCoverageModule())
    missing = get_missing_lines_from_coverage(src, data_file)
    assert missing == {5, 8}


def test_get_missing_lines_from_coverage_missing_data_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "a.py"
    src.write_text("x=1\n", encoding="utf-8")
    # Missing .coverage should not crash CLI flows; they fall back to HTML parsing.
    monkeypatch.setattr(coverage_inspect, "_try_import_coverage", lambda: object())
    assert get_missing_lines_from_coverage(src, tmp_path / ".coverage") is None


def test_get_missing_lines_from_coverage_returns_none_when_fallback_has_no_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "pkg" / "a.py"
    src.parent.mkdir(parents=True)
    src.write_text("x = 1\n", encoding="utf-8")
    data_file = tmp_path / ".coverage"
    data_file.write_text("stub", encoding="utf-8")
    source_abs = str(src.resolve())

    class FakeData:
        def measured_files(self) -> list[str]:
            return ["/other/path/b.py"]

    class FakeCov:
        def __init__(self, *, data_file: str) -> None:
            assert data_file

        def load(self) -> None:
            return None

        def analysis2(self, path: str):  # noqa: ANN202
            if path == source_abs:
                raise RuntimeError("force fallback")
            raise RuntimeError("unexpected")

        def get_data(self) -> FakeData:
            return FakeData()

    class FakeCoverageModule:
        Coverage = FakeCov

    monkeypatch.setattr(coverage_inspect, "_try_import_coverage", lambda: FakeCoverageModule())
    assert get_missing_lines_from_coverage(src, data_file) is None


def test_get_missing_lines_from_coverage_returns_none_on_fallback_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "a.py"
    src.write_text("x = 1\n", encoding="utf-8")
    data_file = tmp_path / ".coverage"
    data_file.write_text("stub", encoding="utf-8")
    source_abs = str(src.resolve())

    class FakeCov:
        def __init__(self, *, data_file: str) -> None:
            assert data_file

        def load(self) -> None:
            return None

        def analysis2(self, path: str):  # noqa: ANN202
            if path == source_abs:
                raise RuntimeError("force fallback")
            raise RuntimeError("unexpected")

        def get_data(self):  # noqa: ANN202
            raise RuntimeError("no data")

    class FakeCoverageModule:
        Coverage = FakeCov

    monkeypatch.setattr(coverage_inspect, "_try_import_coverage", lambda: FakeCoverageModule())
    assert get_missing_lines_from_coverage(src, data_file) is None


def test_find_best_measured_file_and_resolvers() -> None:
    abs_source = "/repo/spflow/x.py"
    measured = ["/tmp/spflow/y.py", "/different/prefix/repo/spflow/x.py"]
    best = coverage_inspect._find_best_measured_file(measured, abs_source)
    assert best == "/different/prefix/repo/spflow/x.py"
    assert coverage_inspect._find_best_measured_file([], abs_source) is None

    rows = [
        coverage_inspect.CoverageIndexRow("spflow/a.py", 1, 0, 0, 100, "a.html"),
        coverage_inspect.CoverageIndexRow("pkg/spflow/b.py", 2, 1, 0, 50, "b.html"),
    ]
    assert coverage_inspect._resolve_source_to_href(rows, "spflow/a.py") == "a.html"
    assert coverage_inspect._resolve_source_to_href(rows, "spflow/b.py") == "b.html"
    assert coverage_inspect._resolve_source_to_href(rows, "missing.py") is None
    assert coverage_inspect._resolve_href_to_source(rows, "b.html") == "pkg/spflow/b.py"
    assert coverage_inspect._resolve_href_to_source(rows, "missing.html") is None


def test_find_best_measured_file_exact_and_suffix_variants() -> None:
    abs_source = "/repo/spflow/x.py"
    measured = ["/repo/spflow/x.py", "/different/prefix/repo/spflow/x.py"]
    assert coverage_inspect._find_best_measured_file(measured, abs_source) == "/repo/spflow/x.py"

    # Exercise fallback when measured paths are shorter than target paths.
    target_with_prefix = "/prefix/repo/spflow/x.py"
    measured_short = ["repo/spflow/x.py"]
    assert coverage_inspect._find_best_measured_file(measured_short, target_with_prefix) == "repo/spflow/x.py"

    # Relative source paths should still resolve via suffix matching.
    rel_source = "spflow/y.py"
    measured_rel = ["/tmp/build/spflow/y.py"]
    assert coverage_inspect._find_best_measured_file(measured_rel, rel_source) == "/tmp/build/spflow/y.py"


def test_read_source_lines_and_normalize_rel_path(tmp_path: Path) -> None:
    existing = tmp_path / "a.py"
    existing.write_text("one\ntwo\n", encoding="utf-8")
    assert coverage_inspect._read_source_lines(existing) == ["one", "two"]
    assert coverage_inspect._read_source_lines(tmp_path / "missing.py") is None
    assert coverage_inspect._normalize_rel_path(existing, tmp_path) == "a.py"


def test_cmd_list_output_sort_and_filters(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    htmlcov = tmp_path / "htmlcov"
    htmlcov.mkdir()
    index = htmlcov / "index.html"
    index.write_text((FIXTURES_DIR / "coverage_index_min.html").read_text(encoding="utf-8"), encoding="utf-8")

    args = argparse.Namespace(
        htmlcov_dir=str(htmlcov),
        index=None,
        limit=10,
        min_missing=1,
        path_prefix="spflow/",
        sort="missing",
    )
    rc = coverage_inspect.cmd_list(args)
    out = capsys.readouterr().out
    assert rc == 0
    assert "spflow/exceptions.py" in out
    assert "spflow/__init__.py" not in out

    args.sort = "ratio"
    assert coverage_inspect.cmd_list(args) == 0


def test_cmd_list_unknown_sort_raises(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    htmlcov = tmp_path / "htmlcov"
    htmlcov.mkdir()
    index = htmlcov / "index.html"
    index.write_text((FIXTURES_DIR / "coverage_index_min.html").read_text(encoding="utf-8"), encoding="utf-8")
    args = argparse.Namespace(
        htmlcov_dir=str(htmlcov),
        index=None,
        limit=10,
        min_missing=1,
        path_prefix=None,
        sort="bad",
    )
    with pytest.raises(ValueError):
        coverage_inspect.cmd_list(args)


def test_load_index_rows_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        coverage_inspect._load_index_rows(tmp_path / "missing.html")


def test_resolve_target_to_paths_html_and_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path
    htmlcov = repo / "htmlcov"
    htmlcov.mkdir()
    index = htmlcov / "index.html"
    index.write_text((FIXTURES_DIR / "coverage_index_min.html").read_text(encoding="utf-8"), encoding="utf-8")
    page = htmlcov / "z_aaa_spflow_exceptions_py.html"
    page.write_text("x", encoding="utf-8")

    html_path, src = coverage_inspect._resolve_target_to_paths(
        target=str(page),
        htmlcov_dir=htmlcov,
        index_path=index,
        repo_root=repo,
    )
    assert html_path == page
    assert src == "spflow/exceptions.py"

    rel_html_path, rel_src = coverage_inspect._resolve_target_to_paths(
        target="z_aaa_spflow_exceptions_py.html",
        htmlcov_dir=htmlcov,
        index_path=index,
        repo_root=repo,
    )
    assert rel_html_path == page
    assert rel_src == "spflow/exceptions.py"

    src_page, src_name = coverage_inspect._resolve_target_to_paths(
        target="spflow/exceptions.py",
        htmlcov_dir=htmlcov,
        index_path=index,
        repo_root=repo,
    )
    assert src_page == page
    assert src_name == "spflow/exceptions.py"

    with pytest.raises(FileNotFoundError):
        coverage_inspect._resolve_target_to_paths(
            target="missing.html",
            htmlcov_dir=htmlcov,
            index_path=index,
            repo_root=repo,
        )

    with pytest.raises(FileNotFoundError):
        coverage_inspect._resolve_target_to_paths(
            target="spflow/unknown.py",
            htmlcov_dir=htmlcov,
            index_path=index,
            repo_root=repo,
        )

    monkeypatch.chdir(repo)


def test_cmd_show_uses_coverage_data_and_markdown_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = tmp_path
    monkeypatch.chdir(repo)

    source = repo / "spflow" / "exceptions.py"
    source.parent.mkdir(parents=True)
    source.write_text("line1\nline2\nline3\nline4\n", encoding="utf-8")

    htmlcov = repo / "htmlcov"
    htmlcov.mkdir()
    index = htmlcov / "index.html"
    index.write_text((FIXTURES_DIR / "coverage_index_min.html").read_text(encoding="utf-8"), encoding="utf-8")
    file_html = htmlcov / "z_aaa_spflow_exceptions_py.html"
    file_html.write_text(
        (FIXTURES_DIR / "coverage_file_min.html").read_text(encoding="utf-8"), encoding="utf-8"
    )

    monkeypatch.setattr(coverage_inspect, "get_missing_lines_from_coverage", lambda *_args, **_kwargs: {4})
    args = argparse.Namespace(
        target="spflow/exceptions.py",
        htmlcov_dir=str(htmlcov),
        index=None,
        coverage_data=str(repo / ".coverage"),
        context=1,
        max_chunks=5,
        include_partial=False,
        format="markdown",
    )
    assert coverage_inspect.cmd_show(args) == 0
    out = capsys.readouterr().out
    assert "**spflow/exceptions.py**" in out
    assert "!   4: line4" in out


def test_cmd_show_falls_back_to_html_with_partial_lines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = tmp_path
    monkeypatch.chdir(repo)
    source = repo / "spflow" / "x.py"
    source.parent.mkdir(parents=True)
    source.write_text("a\nb\nc\n", encoding="utf-8")

    htmlcov = repo / "htmlcov"
    htmlcov.mkdir()
    index = htmlcov / "index.html"
    index.write_text(
        """
        <table class="index"><tbody>
          <tr class="region"><td><a href="z_x.html">spflow/x.py</a></td><td>3</td><td>1</td><td>0</td><td>66%</td></tr>
        </tbody></table>
        """,
        encoding="utf-8",
    )
    (htmlcov / "z_x.html").write_text(
        """
        <h1><b>spflow/x.py</b></h1>
        <p class="run"><span class="n"><a id="t1">1</a></span><span class="t">a</span></p>
        <p class="par"><span class="n"><a id="t2">2</a></span><span class="t">b</span></p>
        <p class="mis"><span class="n"><a id="t3">3</a></span><span class="t">c</span></p>
        """,
        encoding="utf-8",
    )

    # Force fallback path so include_partial is sourced from parsed HTML classes.
    monkeypatch.setattr(coverage_inspect, "get_missing_lines_from_coverage", lambda *_args, **_kwargs: None)
    args = argparse.Namespace(
        target="spflow/x.py",
        htmlcov_dir=str(htmlcov),
        index=None,
        coverage_data=str(repo / ".coverage"),
        context=0,
        max_chunks=5,
        include_partial=True,
        format="text",
    )
    assert coverage_inspect.cmd_show(args) == 0
    out = capsys.readouterr().out
    assert "missing 2 lines" in out
    assert "!   2: b" in out
    assert "!   3: c" in out


def test_render_chunks_text_uses_html_line_text_when_source_line_missing() -> None:
    output = render_chunks_text(
        display_name="x.py",
        source_lines=["only-line"],
        html_line_text={2: "fallback"},
        missed_lines={2},
        context=0,
        max_chunks=5,
    )
    assert "!   2: fallback" in output


def test_cmd_show_uses_parsed_source_path_when_display_source_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = tmp_path
    monkeypatch.chdir(repo)
    htmlcov = repo / "htmlcov"
    htmlcov.mkdir()
    index = htmlcov / "index.html"
    index.write_text(
        """
        <table class="index"><tbody>
          <tr class="region"><td><a href="z_x.html">spflow/missing.py</a></td><td>2</td><td>1</td><td>0</td><td>50%</td></tr>
        </tbody></table>
        """,
        encoding="utf-8",
    )
    real_source = repo / "spflow" / "real.py"
    real_source.parent.mkdir(parents=True)
    real_source.write_text("a\nb\n", encoding="utf-8")
    (htmlcov / "z_x.html").write_text(
        f"""
        <h1><b>{real_source.as_posix()}</b></h1>
        <p class="run"><span class="n"><a id="t1">1</a></span><span class="t">a</span></p>
        <p class="mis"><span class="n"><a id="t2">2</a></span><span class="t">b</span></p>
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(coverage_inspect, "get_missing_lines_from_coverage", lambda *_args, **_kwargs: None)
    args = argparse.Namespace(
        target="spflow/missing.py",
        htmlcov_dir=str(htmlcov),
        index=None,
        coverage_data=str(repo / ".coverage"),
        context=0,
        max_chunks=5,
        include_partial=False,
        format="text",
    )
    assert coverage_inspect.cmd_show(args) == 0
    out = capsys.readouterr().out
    assert real_source.as_posix() in out


def test_build_arg_parser_and_main_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    parser = coverage_inspect.build_arg_parser()
    parsed = parser.parse_args(["list"])
    assert parsed.cmd == "list"

    htmlcov = tmp_path / "htmlcov"
    htmlcov.mkdir()
    (htmlcov / "index.html").write_text(
        (FIXTURES_DIR / "coverage_index_min.html").read_text(encoding="utf-8"), encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    assert coverage_inspect.main(["list", "--limit", "1"]) == 0
    out = capsys.readouterr().out
    assert "cov%" in out

    # argparse exits for missing required target; main should preserve that UX.
    with pytest.raises(SystemExit):
        coverage_inspect.main(["show", "missing.py"])


def test_main_returns_2_when_parser_error_does_not_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    parser = coverage_inspect.build_arg_parser()

    def _no_raise_error(_msg: str) -> None:
        return None

    # Defensive branch: if parser.error is customized not to exit, main still returns 2.
    monkeypatch.setattr(parser, "error", _no_raise_error)
    monkeypatch.setattr(coverage_inspect, "build_arg_parser", lambda: parser)

    def _raise_value_error(_args: argparse.Namespace) -> int:
        raise ValueError("boom")

    args = argparse.Namespace(_handler=_raise_value_error)
    monkeypatch.setattr(parser, "parse_args", lambda _argv=None: args)
    assert coverage_inspect.main([]) == 2


def test_module_entrypoint_raises_system_exit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    htmlcov = tmp_path / "htmlcov"
    htmlcov.mkdir()
    (htmlcov / "index.html").write_text(
        (FIXTURES_DIR / "coverage_index_min.html").read_text(encoding="utf-8"), encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["coverage_inspect", "list", "--limit", "1"])
    monkeypatch.delitem(sys.modules, "spflow.devtools.coverage_inspect", raising=False)
    with pytest.raises(SystemExit):
        runpy.run_module("spflow.devtools.coverage_inspect", run_name="__main__")
