"""Command line interface.

Exit codes are part of the contract, so CI can rely on them:

* ``0`` — the gate passed
* ``1`` — a regression (or another configured failure) was detected
* ``2`` — usage or input error (unreadable/invalid results or config)
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from .compare import ComparisonReport, compare_runs
from .config import ConfigError, GateConfig, load_config
from .loader import ResultsFormatError, load_results
from .report import RENDERERS, to_ascii

__all__ = ["build_parser", "build_config", "main"]

EXIT_PASS = 0
EXIT_REGRESSION = 1
EXIT_ERROR = 2

PROG = "llm-eval-gate"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "Fail CI when an lm-evaluation-harness run regresses against a baseline. "
            "Compares two harness results JSON files, applies relative thresholds and a "
            "stderr-aware noise band, then reports per-metric verdicts."
        ),
        epilog=(
            "Exit codes: 0 = pass, 1 = regression, 2 = usage/input error. "
            "Example: llm-eval-gate baseline.json candidate.json --format markdown"
        ),
    )
    parser.add_argument("baseline", help="path to the baseline results JSON")
    parser.add_argument("candidate", help="path to the candidate results JSON")

    thresholds = parser.add_argument_group("thresholds")
    thresholds.add_argument(
        "--max-rel-regression",
        type=float,
        metavar="FRACTION",
        help="relative drop that fails the gate (default: 0.05)",
    )
    thresholds.add_argument(
        "--max-abs-regression",
        type=float,
        metavar="VALUE",
        help="absolute drop that fails the gate; also used when a baseline is ~0",
    )
    thresholds.add_argument(
        "--sigma",
        type=float,
        metavar="N",
        help="standard-deviation multiplier for the noise band (default: 2)",
    )
    thresholds.add_argument(
        "--no-significance",
        action="store_true",
        help="fail on any change past the threshold, ignoring standard errors",
    )
    thresholds.add_argument(
        "--ignore",
        action="append",
        default=[],
        metavar="METRIC",
        help="metric to skip entirely (repeatable)",
    )
    thresholds.add_argument(
        "--tasks",
        action="append",
        default=[],
        metavar="TASK",
        help="only compare these tasks (repeatable)",
    )
    thresholds.add_argument(
        "--metrics",
        action="append",
        default=[],
        metavar="METRIC",
        help="only compare these metrics (repeatable)",
    )
    thresholds.add_argument(
        "--lower-is-better",
        action="append",
        default=[],
        metavar="METRIC",
        help="force direction for a metric (repeatable)",
    )
    thresholds.add_argument(
        "--higher-is-better",
        action="append",
        default=[],
        metavar="METRIC",
        help="force direction for a metric (repeatable)",
    )

    failures = parser.add_argument_group("failure conditions")
    failures.add_argument(
        "--fail-on-missing-task",
        action="store_true",
        help="fail when a baseline task is absent from the candidate run",
    )
    failures.add_argument(
        "--allow-missing-metric",
        action="store_true",
        help="do not fail when a metric disappears inside a task",
    )
    failures.add_argument(
        "--fail-on-new-task",
        action="store_true",
        help="fail when the candidate run contains a task the baseline lacked",
    )

    output = parser.add_argument_group("output")
    output.add_argument(
        "--config",
        metavar="PATH",
        help="TOML configuration file; CLI flags override its values",
    )
    output.add_argument(
        "--format",
        choices=sorted(RENDERERS),
        default="text",
        help="report format (default: text)",
    )
    output.add_argument(
        "--output",
        metavar="PATH",
        help="also write the rendered report to this file",
    )
    output.add_argument("--quiet", action="store_true", help="suppress the stdout report")
    output.add_argument(
        "--ascii",
        action="store_true",
        help="force ASCII output markers (for consoles that cannot render Unicode)",
    )
    output.add_argument("--version", action="version", version=f"{PROG} {_version()}")
    return parser


def _version() -> str:
    from . import __version__

    return __version__


def build_config(args: argparse.Namespace) -> GateConfig:
    """Layer CLI flags on top of an optional TOML config file."""
    config = load_config(args.config) if args.config else GateConfig()

    changes: dict[str, object] = {}
    if args.max_rel_regression is not None:
        changes["max_rel_regression"] = args.max_rel_regression
    if args.max_abs_regression is not None:
        changes["max_abs_regression"] = args.max_abs_regression
    if args.sigma is not None:
        changes["sigma"] = args.sigma
    if args.no_significance:
        changes["require_significance"] = False
    if args.fail_on_missing_task:
        changes["fail_on_missing_task"] = True
    if args.allow_missing_metric:
        changes["fail_on_missing_metric"] = False
    if args.fail_on_new_task:
        changes["fail_on_new_task"] = True
    if args.ignore:
        changes["ignore_metrics"] = config.ignore_metrics | frozenset(args.ignore)
    if args.tasks:
        changes["include_tasks"] = frozenset(args.tasks)
    if args.metrics:
        changes["include_metrics"] = frozenset(args.metrics)
    if args.lower_is_better:
        changes["lower_is_better"] = config.lower_is_better | frozenset(args.lower_is_better)
    if args.higher_is_better:
        changes["higher_is_better"] = config.higher_is_better | frozenset(args.higher_is_better)

    if not changes:
        return config
    return replace(config, **changes)


def _write_stdout(text: str, force_ascii: bool) -> None:
    """Print a report without ever dying on a legacy console encoding.

    Windows terminals still ship with code pages such as cp936/cp1252, where
    printing our status markers raises ``UnicodeEncodeError`` -- the gate would
    crash *after* computing the verdict but *before* telling anyone what it
    found. Degrade to ASCII markers when the stream cannot represent them, and
    keep a last-resort replacement pass in case even that fails.
    """
    if force_ascii:
        text = to_ascii(text)

    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    if not force_ascii:
        try:
            text.encode(encoding)
        except (UnicodeEncodeError, LookupError):
            text = to_ascii(text)

    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode(encoding, "replace").decode(encoding, "replace"))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = build_config(args)
        baseline = load_results(args.baseline)
        candidate = load_results(args.candidate)
    except (ResultsFormatError, ConfigError) as exc:
        print(f"{PROG}: error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except OSError as exc:  # pragma: no cover - defensive
        print(f"{PROG}: error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    report: ComparisonReport = compare_runs(baseline, candidate, config)
    rendered = RENDERERS[args.format](report)

    if args.output:
        try:
            Path(args.output).write_text(rendered + "\n", encoding="utf-8")
        except OSError as exc:
            print(f"{PROG}: error: cannot write {args.output}: {exc}", file=sys.stderr)
            return EXIT_ERROR

    if not args.quiet:
        _write_stdout(rendered, force_ascii=args.ascii)

    return report.exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
