"""End-to-end tests for the CLI contract (exit codes and flag plumbing)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from llm_eval_gate.cli import EXIT_ERROR, EXIT_PASS, EXIT_REGRESSION, build_parser, main

FIXTURES = Path(__file__).parent / "fixtures"

BASELINE = str(FIXTURES / "baseline.json")
REGRESSED = str(FIXTURES / "candidate_regressed.json")
OK = str(FIXTURES / "candidate_ok.json")
NOISY = str(FIXTURES / "candidate_noisy.json")
MISSING_TASK = str(FIXTURES / "candidate_missing_task.json")
BROKEN = str(FIXTURES / "broken.json")


def test_regression_exits_one(capsys):
    assert main([BASELINE, REGRESSED]) == EXIT_REGRESSION
    assert "FAIL" in capsys.readouterr().out


def test_improvement_exits_zero(capsys):
    assert main([BASELINE, OK]) == EXIT_PASS
    assert "PASS" in capsys.readouterr().out


def test_invalid_input_exits_two(capsys):
    assert main([BASELINE, BROKEN]) == EXIT_ERROR
    assert "error:" in capsys.readouterr().err


def test_missing_file_exits_two(capsys):
    assert main([BASELINE, str(FIXTURES / "nope.json")]) == EXIT_ERROR
    assert "error:" in capsys.readouterr().err


def test_json_format_is_machine_readable(capsys):
    main([BASELINE, REGRESSED, "--format", "json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] is False
    assert payload["summary"]["regressions"] == 2


def test_json_output_flag_emits_only_clean_json(capsys):
    code = main([BASELINE, REGRESSED, "--json-output"])
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert code == EXIT_REGRESSION
    assert payload["schema"] == "evalguard/v1"
    assert "\033[" not in out


def test_json_output_can_write_an_action_output_file(tmp_path, capsys):
    destination = tmp_path / "evalguard.json"
    code = main([BASELINE, REGRESSED, "--json-output", str(destination), "--quiet"])
    assert code == EXIT_REGRESSION
    assert capsys.readouterr().out == ""
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["summary"]["regressions"] == 2


def test_color_always_adds_ansi_and_never_removes_it(capsys):
    assert main([BASELINE, REGRESSED, "--color", "always"]) == EXIT_REGRESSION
    assert "\033[31m" in capsys.readouterr().out
    assert main([BASELINE, REGRESSED, "--color", "never"]) == EXIT_REGRESSION
    assert "\033[" not in capsys.readouterr().out


def test_markdown_format(capsys):
    main([BASELINE, REGRESSED, "--format", "markdown"])
    assert "### Regressions" in capsys.readouterr().out


def test_quiet_suppresses_stdout(capsys):
    code = main([BASELINE, REGRESSED, "--quiet"])
    assert code == EXIT_REGRESSION
    assert capsys.readouterr().out == ""


def test_output_writes_the_report_to_a_file(tmp_path, capsys):
    destination = tmp_path / "report.md"
    main([BASELINE, REGRESSED, "--format", "markdown", "--output", str(destination), "--quiet"])
    assert "### Regressions" in destination.read_text(encoding="utf-8")


def test_no_significance_flag_turns_noise_into_a_failure():
    assert main([BASELINE, NOISY, "--quiet"]) == EXIT_PASS
    assert main([BASELINE, NOISY, "--no-significance", "--quiet"]) == EXIT_REGRESSION


def test_max_rel_regression_flag_relaxes_the_gate():
    assert main([BASELINE, REGRESSED, "--quiet"]) == EXIT_REGRESSION
    assert main([BASELINE, REGRESSED, "--max-rel-regression", "0.5", "--quiet"]) == EXIT_PASS


def test_fail_on_missing_task_flag():
    assert main([BASELINE, MISSING_TASK, "--quiet"]) == EXIT_PASS
    assert main([BASELINE, MISSING_TASK, "--fail-on-missing-task", "--quiet"]) == EXIT_REGRESSION


def test_ignore_flag_skips_a_metric():
    assert main([BASELINE, REGRESSED, "--ignore", "acc", "--ignore", "byte_perplexity", "--quiet"]) == EXIT_PASS


def test_tasks_flag_restricts_the_comparison():
    assert main([BASELINE, REGRESSED, "--tasks", "arc_easy", "--quiet"]) == EXIT_REGRESSION
    assert main([BASELINE, REGRESSED, "--tasks", "hellaswag", "--quiet"]) == EXIT_PASS


def test_direction_override_flag():
    # Flipping acc to "lower is better" clears that regression, but the
    # perplexity blow-up must still fail the gate on its own.
    assert main([BASELINE, REGRESSED, "--lower-is-better", "acc", "--quiet"]) == EXIT_REGRESSION
    assert (
        main(
            [BASELINE, REGRESSED, "--lower-is-better", "acc", "--ignore", "byte_perplexity", "--quiet"]
        )
        == EXIT_PASS
    )


def test_config_file_is_applied_and_cli_overrides_it(tmp_path):
    config = tmp_path / "gate.toml"
    config.write_text("[gate]\nmax_rel_regression = 0.5\n", encoding="utf-8")

    assert main([BASELINE, REGRESSED, "--config", str(config), "--quiet"]) == EXIT_PASS
    # An explicit CLI threshold must win over the file.
    assert (
        main([BASELINE, REGRESSED, "--config", str(config), "--max-rel-regression", "0.01", "--quiet"])
        == EXIT_REGRESSION
    )


def test_invalid_config_file_exits_two(tmp_path, capsys):
    config = tmp_path / "bad.toml"
    config.write_text("[gate]\nmax_rel_regression = \"x\"\n", encoding="utf-8")
    assert main([BASELINE, REGRESSED, "--config", str(config)]) == EXIT_ERROR
    assert "error:" in capsys.readouterr().err


def test_parser_accepts_version_and_help():
    parser = build_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["--version"])
    assert excinfo.value.code == 0


def test_version_matches_package_version():
    import llm_eval_gate

    assert llm_eval_gate.__version__ == "0.1.0"


def test_module_entry_point_is_wired():
    # `python -m llm_eval_gate` must resolve to the same callable as the console script.
    from llm_eval_gate.__main__ import main as module_main

    assert module_main is main


def test_pythonpath_bootstrap_covers_src_layout():
    import llm_eval_gate as package

    assert Path(package.__file__).parent.name == "llm_eval_gate"
    assert "src" in str(package.__file__)
    assert sys.version_info >= (3, 11)
