"""Tests for harness results loading and metric-key normalisation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_eval_gate.loader import ResultsFormatError, load_results, parse_metric_key

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("acc,none", ("acc", None)),
        ("acc", ("acc", None)),
        ("acc,-none", ("acc", None)),
        ("acc,null", ("acc", None)),
        ("exact_match,strict-match", ("exact_match", "strict-match")),
        ("f1,marian", ("f1", "marian")),
        (" acc , none ", ("acc", None)),
    ],
)
def test_parse_metric_key(raw, expected):
    assert parse_metric_key(raw) == expected


def test_load_results_attaches_stderr_and_metadata():
    run = load_results(FIXTURES / "baseline.json")

    assert run.model_name == "hf"
    assert run.git_hash == "1111111111111111111111111111111111111111"
    assert run.harness_version == "0.4.5"
    assert run.task_names == ["arc_easy", "hellaswag", "wikitext"]

    arc_acc = run.tasks["arc_easy"]["acc"]
    assert arc_acc.value == pytest.approx(0.75)
    assert arc_acc.stderr == pytest.approx(0.02)
    assert arc_acc.has_stderr

    # byte_perplexity has no stderr sibling in the fixture.
    assert run.tasks["wikitext"]["byte_perplexity"].stderr is None


def test_load_results_ignores_non_numeric_entries():
    run = load_results(FIXTURES / "baseline.json")
    assert "alias" not in run.tasks["arc_easy"]


def test_load_results_keeps_filter_suffixes_distinct(tmp_path):
    payload = {
        "results": {
            "task": {
                "exact_match,strict-match": 0.5,
                "exact_match_stderr,strict-match": 0.1,
                "exact_match,flexible-extract": 0.6,
                "alias": "task",
            }
        }
    }
    path = tmp_path / "filters.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    run = load_results(path)
    metrics = run.tasks["task"]

    assert metrics["exact_match[strict-match]"].value == pytest.approx(0.5)
    assert metrics["exact_match[strict-match]"].stderr == pytest.approx(0.1)
    assert metrics["exact_match[flexible-extract]"].value == pytest.approx(0.6)
    assert metrics["exact_match[flexible-extract]"].stderr is None


def test_load_results_skips_nan_and_infinity(tmp_path):
    payload = {
        "results": {
            "task": {
                "acc,none": 0.5,
                "acc_stderr,none": float("nan"),
                "loss,none": float("inf"),
            }
        }
    }
    path = tmp_path / "weird.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    run = load_results(path)
    assert run.tasks["task"]["acc"].stderr is None
    assert "loss" not in run.tasks["task"]


def test_load_results_rejects_missing_results_object():
    with pytest.raises(ResultsFormatError, match="results"):
        load_results(FIXTURES / "not_results.json")


def test_load_results_rejects_invalid_json():
    with pytest.raises(ResultsFormatError, match="not valid JSON"):
        load_results(FIXTURES / "broken.json")


def test_load_results_reports_missing_file():
    with pytest.raises(ResultsFormatError, match="cannot read"):
        load_results(FIXTURES / "does-not-exist.json")


def test_load_results_rejects_results_without_numbers(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"results": {"task": {"alias": "task"}}}), encoding="utf-8")
    with pytest.raises(ResultsFormatError, match="no numeric metrics"):
        load_results(path)
