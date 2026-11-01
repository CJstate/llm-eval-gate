"""Tests for the text / Markdown / JSON / GitHub renderers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_eval_gate import GateConfig, compare_runs, load_results
from llm_eval_gate.report import (
    render_github,
    render_json,
    render_markdown,
    render_text,
    report_to_dict,
)

FIXTURES = Path(__file__).parent / "fixtures"


def report(candidate="candidate_regressed.json", **config_kwargs):
    return compare_runs(
        load_results(FIXTURES / "baseline.json"),
        load_results(FIXTURES / candidate),
        GateConfig(**config_kwargs),
    )


def test_report_to_dict_is_serialisable_and_summarises_counts():
    payload = report_to_dict(report())

    assert payload["schema"] == "evalguard/v1"
    assert payload["passed"] is False
    assert payload["exit_code"] == 1
    assert payload["summary"]["regressions"] == 2
    assert len(payload["regressions"]) == 2
    assert payload["baseline"]["git_hash"].startswith("1111")
    assert payload["candidate"]["git_hash"].startswith("2222")
    # It must survive a JSON round-trip unchanged.
    assert json.loads(json.dumps(payload)) == payload


def test_render_json_is_valid_json():
    parsed = json.loads(render_json(report()))
    assert parsed["summary"]["regressions"] == 2


def test_render_text_states_failure_and_lists_regressions():
    text = render_text(report())
    assert "FAIL" in text
    assert "arc_easy" in text
    assert "byte_perplexity" in text
    assert "Regressions:" in text
    assert "| Task" in text
    assert "| Baseline" in text
    assert "| Candidate" in text
    assert "| Noise Band" in text
    assert "| Status" in text
    assert "outside the noise band" in text
    assert "Check the model checkpoint" in text


def test_render_text_colors_regressions_only_when_requested():
    plain = render_text(report())
    colored = render_text(report(), color=True)
    assert "\033[" not in plain
    assert "\033[31m" in colored
    assert "REGRESSION" in colored


def test_render_text_marks_pass_inside_the_noise_band():
    from llm_eval_gate import GateConfig

    noisy = compare_runs(
        load_results(FIXTURES / "baseline.json"),
        load_results(FIXTURES / "candidate_noisy.json"),
        GateConfig(max_rel_regression=0.02),
    )
    assert "PASS (noise)" in render_text(noisy)


def test_render_text_states_pass():
    text = render_text(report("candidate_ok.json"))
    assert text.startswith("evalguard: PASS")
    assert "improvement" in text


def test_render_markdown_contains_a_table_and_regression_section():
    markdown = render_markdown(report())
    assert "## EvalGuard — ❌ **FAIL**" in markdown
    assert "### Regressions" in markdown
    assert "| Task | Metric | Baseline | Candidate | Δ | Detail |" in markdown
    assert "`arc_easy`" in markdown
    assert "<details>" in markdown


def test_render_markdown_passes_cleanly():
    markdown = render_markdown(report("candidate_ok.json"))
    assert "✅ **PASS**" in markdown
    assert "### Regressions" not in markdown


def test_render_markdown_flags_missing_tasks():
    markdown = render_markdown(report("candidate_missing_task.json"))
    assert "hellaswag" in markdown
    assert "missing from the candidate run" in markdown


def test_render_github_emits_error_annotations():
    output = render_github(report())
    assert "::error title=EvalGuard regression::" in output
    assert "arc_easy/acc" in output


def test_render_github_emits_missing_task_annotations_when_gated():
    strict = report("candidate_missing_task.json", fail_on_missing_task=True)
    output = render_github(strict)
    assert "::error title=Missing eval task::" in output


def test_render_github_escapes_newlines():
    output = render_github(report())
    for line in output.splitlines():
        assert "\r" not in line


@pytest.mark.parametrize(
    "renderer", [render_text, render_markdown, render_json, render_github]
)
def test_renderers_handle_a_clean_pass(renderer):
    assert renderer(report("candidate_ok.json"))
