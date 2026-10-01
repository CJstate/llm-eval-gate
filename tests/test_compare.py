"""Tests for the regression-detection engine."""

from __future__ import annotations

import pytest

from llm_eval_gate import GateConfig, compare_runs, is_lower_is_better, load_results
from llm_eval_gate.compare import (
    STATUS_IMPROVEMENT,
    STATUS_MISSING,
    STATUS_REGRESSION,
)
from llm_eval_gate.loader import MetricResult, RunResults

FIXTURES = __import__("pathlib").Path(__file__).parent / "fixtures"


def load(name):
    return load_results(FIXTURES / name)


@pytest.fixture
def baseline():
    return load("baseline.json")


def regressed_names(report):
    return {(m.task, m.metric) for m in report.regressions}


# -- direction detection ---------------------------------------------------


@pytest.mark.parametrize(
    ("metric", "expected"),
    [
        ("acc", False),
        ("acc_norm", False),
        ("exact_match", False),
        ("f1", False),
        ("mcc", False),
        ("word_perplexity", True),
        ("byte_perplexity", True),
        ("bits_per_byte", True),
        ("ppl", True),
        ("loss", True),
        ("wer", True),
        ("char_perplexity", True),
    ],
)
def test_is_lower_is_better(metric, expected):
    assert is_lower_is_better(metric) is expected


# -- the core decision -----------------------------------------------------


def test_detects_acc_and_perplexity_regressions(baseline):
    report = compare_runs(baseline, load("candidate_regressed.json"))

    assert regressed_names(report) == {("arc_easy", "acc"), ("wikitext", "byte_perplexity")}
    assert report.failed
    assert report.exit_code == 1


def test_regression_above_threshold_is_not_reported_as_improvement(baseline):
    report = compare_runs(baseline, load("candidate_regressed.json"))
    improvements = {(m.task, m.metric) for m in report.improvements}
    assert ("arc_easy", "acc") not in improvements


def test_significance_is_derived_from_reported_standard_errors(baseline):
    report = compare_runs(baseline, load("candidate_regressed.json"))
    by_key = {(m.task, m.metric): m for m in report.metrics}

    # -0.06 drop against a pooled stderr of 0.0283 is beyond 2 sigma.
    assert by_key[("arc_easy", "acc")].significant is True
    # -0.005 against a pooled stderr of 0.0141 is not.
    assert by_key[("hellaswag", "acc")].significant is False
    # byte_perplexity has no stderr anywhere, so significance is unknowable.
    assert by_key[("wikitext", "byte_perplexity")].significant is None


def test_small_changes_pass_even_in_the_same_direction(baseline):
    report = compare_runs(baseline, load("candidate_regressed.json"))
    by_key = {(m.task, m.metric): m for m in report.metrics}
    assert by_key[("hellaswag", "acc")].status == "pass"
    assert by_key[("wikitext", "word_perplexity")].status == "pass"


def test_improvements_are_reported(baseline):
    report = compare_runs(baseline, load("candidate_ok.json"))

    assert report.regressions == ()
    assert not report.failed
    assert report.exit_code == 0
    assert {(m.task, m.metric) for m in report.improvements} == {("arc_easy", "acc")}
    assert report.improvements[0].status == STATUS_IMPROVEMENT


# -- noise band ------------------------------------------------------------


def test_noise_band_suppresses_a_large_but_uncertain_drop(baseline):
    report = compare_runs(baseline, load("candidate_noisy.json"))

    assert report.regressions == ()
    assert not report.failed

    arc = next(m for m in report.metrics if (m.task, m.metric) == ("arc_easy", "acc"))
    assert arc.status == "pass"
    assert arc.significant is False
    assert "noise" in arc.reason


def test_require_significance_false_turns_noise_into_a_failure(baseline):
    report = compare_runs(
        baseline,
        load("candidate_noisy.json"),
        GateConfig(require_significance=False),
    )

    assert regressed_names(report) == {("arc_easy", "acc")}
    assert report.failed


def test_larger_sigma_widens_the_noise_band(baseline, monkeypatch):
    strict = compare_runs(baseline, load("candidate_regressed.json"), GateConfig(sigma=1.0))
    lenient = compare_runs(baseline, load("candidate_regressed.json"), GateConfig(sigma=5.0))

    # 0.06 clears a 1-sigma band but not a 5-sigma band.
    assert ("arc_easy", "acc") in regressed_names(strict)
    assert ("arc_easy", "acc") not in regressed_names(lenient)


# -- configuration surfaces ------------------------------------------------


def test_threshold_is_configurable(baseline):
    loose = compare_runs(
        baseline, load("candidate_regressed.json"), GateConfig(max_rel_regression=0.20)
    )
    assert regressed_names(loose) == set()


def test_direction_override_flips_the_verdict(baseline):
    report = compare_runs(
        baseline,
        load("candidate_regressed.json"),
        GateConfig(lower_is_better=frozenset({"acc"})),
    )
    assert ("arc_easy", "acc") not in regressed_names(report)
    assert ("arc_easy", "acc") in {(m.task, m.metric) for m in report.improvements}


def test_ignore_metrics_skips_them(baseline):
    report = compare_runs(
        baseline, load("candidate_regressed.json"), GateConfig(ignore_metrics=frozenset({"acc"}))
    )
    assert regressed_names(report) == {("wikitext", "byte_perplexity")}
    assert all(m.metric != "acc" for m in report.metrics)


def test_include_tasks_restricts_the_comparison(baseline):
    report = compare_runs(
        baseline,
        load("candidate_regressed.json"),
        GateConfig(include_tasks=frozenset({"arc_easy"})),
    )
    assert {t.task for t in report.tasks} == {"arc_easy"}


def test_per_metric_override_wins_over_the_global_threshold(baseline):
    from llm_eval_gate.config import MetricOverride

    report = compare_runs(
        baseline,
        load("candidate_regressed.json"),
        GateConfig(overrides={"arc_easy.acc": MetricOverride(max_rel_regression=0.01)}),
    )
    assert ("arc_easy", "acc") in regressed_names(report)


# -- missing / new tasks ---------------------------------------------------


def test_missing_task_is_tolerated_by_default(baseline):
    report = compare_runs(baseline, load("candidate_missing_task.json"))

    assert report.missing_tasks == ("hellaswag",)
    assert not report.failed
    assert report.exit_code == 0


def test_missing_task_fails_when_requested(baseline):
    report = compare_runs(
        baseline, load("candidate_missing_task.json"), GateConfig(fail_on_missing_task=True)
    )
    assert report.failed


def test_missing_task_marks_all_its_metrics():
    base = RunResults(path="b", tasks={"t": {"acc": MetricResult(0.5)}})
    cand = RunResults(path="c", tasks={"other": {"acc": MetricResult(0.5)}})

    report = compare_runs(base, cand)
    assert report.missing_tasks == ("t",)
    assert report.new_tasks == ("other",)
    missing = next(t for t in report.tasks if t.task == "t")
    assert missing.status == STATUS_MISSING


def test_new_task_fails_when_requested():
    base = RunResults(path="b", tasks={"t": {"acc": MetricResult(0.5)}})
    cand = RunResults(
        path="c",
        tasks={"t": {"acc": MetricResult(0.5)}, "brand_new": {"acc": MetricResult(0.1)}},
    )
    assert not compare_runs(base, cand).failed
    assert compare_runs(base, cand, GateConfig(fail_on_new_task=True)).failed


# -- degenerate baselines --------------------------------------------------


def test_zero_baseline_uses_the_absolute_rule_when_configured():
    base = RunResults(path="b", tasks={"t": {"ppl": MetricResult(0.0)}})
    cand = RunResults(path="c", tasks={"t": {"ppl": MetricResult(0.5)}})

    # Without an absolute limit a zero baseline cannot be judged relatively.
    assert not compare_runs(base, cand).failed

    gated = compare_runs(base, cand, GateConfig(max_abs_regression=0.1))
    assert gated.failed
    assert gated.regressions[0].status == STATUS_REGRESSION


def test_identical_runs_are_a_clean_pass(baseline):
    report = compare_runs(baseline, baseline)
    assert not report.failed
    assert report.regressions == ()
    assert report.improvements == ()
