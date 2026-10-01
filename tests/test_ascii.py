"""The ASCII fallback exists because Windows consoles still default to legacy
code pages: printing a ❌ used to raise UnicodeEncodeError and kill the gate
after it had already decided there was a regression."""

from __future__ import annotations

import io
import sys

import pytest

from llm_eval_gate.cli import EXIT_PASS, EXIT_REGRESSION, _write_stdout, main
from llm_eval_gate.report import to_ascii

BASELINE = "tests/fixtures/baseline.json"
REGRESSED = "tests/fixtures/candidate_regressed.json"
OK = "tests/fixtures/candidate_ok.json"


def test_to_ascii_is_pure_ascii():
    text = to_ascii("❌ arc_easy/acc 0.75 → 0.69 (Δ -8%, σ≈2.1, p≤0.05) · n=1")
    assert text.isascii()
    assert "[FAIL]" in text
    assert "->" in text
    assert "delta" in text
    assert "sigma" in text
    assert "<=" in text


def test_to_ascii_leaves_ascii_untouched():
    assert to_ascii("plain 0.123 ok") == "plain 0.123 ok"


def test_to_ascii_covers_every_marker_the_renderers_emit():
    from llm_eval_gate.compare import compare_runs
    from llm_eval_gate.loader import load_results
    from llm_eval_gate.report import RENDERERS
    from llm_eval_gate.config import GateConfig

    report = compare_runs(load_results(BASELINE), load_results(REGRESSED), GateConfig())
    for name, render in RENDERERS.items():
        assert to_ascii(render(report)).isascii(), f"{name} renderer leaked non-ASCII"


@pytest.mark.parametrize(
    "path,expected",
    [(OK, EXIT_PASS), (REGRESSED, EXIT_REGRESSION)],
)
def test_ascii_flag_runs_on_a_real_console(path, expected, capsys):
    assert main([BASELINE, path, "--ascii"]) == expected
    out = capsys.readouterr().out
    assert out.isascii()
    if expected == EXIT_REGRESSION:
        assert "[FAIL]" in out
    else:
        assert "[PASS]" in out


def test_write_stdout_degrades_instead_of_raising(monkeypatch):
    """Simulate a cp936 console: the emoji cannot be encoded, the report must
    still come out (as ASCII) rather than exploding."""
    stream = io.TextIOWrapper(io.BytesIO(), encoding="gbk", newline="")
    monkeypatch.setattr(sys, "stdout", stream)

    _write_stdout("✅ gate passed · ❌ arc_easy/acc", force_ascii=False)

    stream.flush()
    written = stream.buffer.getvalue().decode("gbk")
    assert "PASS" in written
    assert "FAIL" in written
