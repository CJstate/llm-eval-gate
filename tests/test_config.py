"""Tests for gate configuration and TOML loading."""

from __future__ import annotations

import pytest

from llm_eval_gate.config import ConfigError, GateConfig, MetricOverride, load_config


def test_defaults_are_documented_values():
    config = GateConfig()
    assert config.max_rel_regression == 0.05
    assert config.sigma == 2.0
    assert config.require_significance is True
    assert config.fail_on_missing_task is False
    assert config.fail_on_missing_metric is True
    assert config.fail_on_new_task is False
    assert config.max_abs_regression is None


def test_from_mapping_reads_gate_table():
    config = GateConfig.from_mapping(
        {
            "gate": {
                "max_rel_regression": 0.02,
                "sigma": 3,
                "require_significance": False,
                "fail_on_missing_task": True,
                "ignore_metrics": ["samples", "alias"],
                "lower_is_better": ["ppl"],
            }
        }
    )
    assert config.max_rel_regression == pytest.approx(0.02)
    assert config.sigma == pytest.approx(3.0)
    assert config.require_significance is False
    assert config.fail_on_missing_task is True
    assert config.ignore_metrics == frozenset({"samples", "alias"})
    assert config.lower_is_better == frozenset({"ppl"})


def test_from_mapping_accepts_a_bare_table():
    config = GateConfig.from_mapping({"max_rel_regression": 0.1})
    assert config.max_rel_regression == pytest.approx(0.1)


def test_override_precedence_is_metric_then_task_then_global():
    config = GateConfig.from_mapping(
        {
            "gate": {"max_rel_regression": 0.10},
            "overrides": {
                "arc_easy": {"max_rel_regression": 0.05},
                "arc_easy.acc": {"max_rel_regression": 0.01, "lower_is_better": False},
            },
        }
    )
    assert config.threshold_for("arc_easy", "acc") == pytest.approx(0.01)
    assert config.threshold_for("arc_easy", "acc_norm") == pytest.approx(0.05)
    assert config.threshold_for("hellaswag", "acc") == pytest.approx(0.10)
    assert config.direction_for("arc_easy", "acc") is False


def test_explicit_direction_lists_win_over_heuristics():
    config = GateConfig(higher_is_better=frozenset({"ppl"}))
    assert config.direction_for("t", "ppl") is False
    assert config.direction_for("t", "acc") is None  # falls back to auto-detection


def test_is_tracked_honours_filters():
    config = GateConfig(
        ignore_metrics=frozenset({"samples"}),
        include_tasks=frozenset({"arc_easy"}),
        include_metrics=frozenset({"acc"}),
    )
    assert config.is_tracked("arc_easy", "acc")
    assert not config.is_tracked("arc_easy", "samples")
    assert not config.is_tracked("hellaswag", "acc")
    assert not config.is_tracked("arc_easy", "f1")


@pytest.mark.parametrize(
    "payload",
    [
        {"gate": {"max_rel_regression": "a lot"}},
        {"gate": {"require_significance": "yes"}},
        {"gate": {"ignore_metrics": "acc"}},
        {"gate": {"ignore_metrics": [1, 2]}},
        {"overrides": {"arc_easy": "nope"}},
        {"overrides": {"arc_easy": {"sigma": "wide"}}},
    ],
)
def test_invalid_configuration_is_rejected(payload):
    with pytest.raises(ConfigError):
        GateConfig.from_mapping(payload)


def test_load_config_from_toml(tmp_path):
    path = tmp_path / "gate.toml"
    path.write_text(
        """
[gate]
max_rel_regression = 0.02
require_significance = false

[overrides."wikitext.word_perplexity"]
max_rel_regression = 0.01
""",
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.max_rel_regression == pytest.approx(0.02)
    assert config.require_significance is False
    assert config.threshold_for("wikitext", "word_perplexity") == pytest.approx(0.01)
    assert config.override_for("wikitext", "word_perplexity").max_rel_regression == pytest.approx(0.01)


def test_load_config_reports_invalid_toml(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text("this is not = = toml", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid TOML"):
        load_config(path)


def test_load_config_reports_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="cannot read"):
        load_config(tmp_path / "nope.toml")


def test_metric_override_defaults_are_all_unset():
    override = MetricOverride()
    assert override.max_rel_regression is None
    assert override.max_abs_regression is None
    assert override.sigma is None
    assert override.require_significance is None
    assert override.lower_is_better is None
