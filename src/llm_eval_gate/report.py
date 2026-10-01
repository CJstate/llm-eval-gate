"""Renderers: text for terminals, Markdown for PR comments, JSON for machines.

Every renderer is a pure function of a :class:`~llm_eval_gate.compare.ComparisonReport`,
which keeps the gate decision and its presentation testable in isolation.
"""

from __future__ import annotations

import json
from typing import Any

from .compare import (
    STATUS_IMPROVEMENT,
    STATUS_MISSING,
    STATUS_NEW,
    STATUS_PASS,
    STATUS_REGRESSION,
    ComparisonReport,
    MetricComparison,
)

__all__ = [
    "RENDERERS",
    "ASCII_MARKERS",
    "render_github",
    "render_json",
    "render_markdown",
    "render_text",
    "report_to_dict",
    "to_ascii",
]

#: Substitutions used when the output stream cannot encode the pretty markers.
#: Windows consoles still default to legacy code pages (cp936, cp1252, ...),
#: where printing ``❌`` raises ``UnicodeEncodeError`` and kills the process
#: before it can report the regression it had just found.
ASCII_MARKERS = {
    "❌": "[FAIL]",
    "✅": "[PASS]",
    "➖": "[ok]",
    "⚠️": "[WARN]",
    "⚠": "[WARN]",
    "🆕": "[NEW]",
    "—": "-",
    "–": "-",
    "·": "|",
    "→": "->",
    "Δ": "delta",
    "σ": "sigma",
    "≈": "~",
    "≤": "<=",
    "≥": ">=",
}


def to_ascii(text: str) -> str:
    """Degrade a rendered report to pure ASCII.

    Used when the target stream's encoding cannot represent the Unicode
    markers, so a regression report stays readable (and its exit code stays
    meaningful) on a legacy console or a log aggregator.
    """
    for pretty, plain in ASCII_MARKERS.items():
        text = text.replace(pretty, plain)
    return text.encode("ascii", "replace").decode("ascii")

_STATUS_ICON = {
    STATUS_REGRESSION: "❌",
    STATUS_IMPROVEMENT: "✅",
    STATUS_PASS: "➖",
    STATUS_MISSING: "⚠️",
    STATUS_NEW: "🆕",
}


def _fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}g}"


def _fmt_delta(comparison: MetricComparison) -> str:
    if comparison.rel_change is None:
        if comparison.delta is None:
            return "—"
        return f"{comparison.delta:+.3g}"
    return f"{comparison.rel_change * 100:+.2f}%"


def _stderr_note(comparison: MetricComparison) -> str:
    if comparison.significant is False:
        return "within noise"
    return comparison.reason


def report_to_dict(report: ComparisonReport) -> dict[str, Any]:
    """A stable, JSON-serialisable view of the report."""

    def metric_dict(m: MetricComparison) -> dict[str, Any]:
        return {
            "task": m.task,
            "metric": m.metric,
            "status": m.status,
            "baseline": m.baseline,
            "candidate": m.candidate,
            "baseline_stderr": m.baseline_stderr,
            "candidate_stderr": m.candidate_stderr,
            "delta": m.delta,
            "rel_change": m.rel_change,
            "worsened_by": m.worsened_by,
            "significant": m.significant,
            "threshold": m.threshold,
            "noise_band": m.noise_band,
            "lower_is_better": m.lower_is_better,
            "reason": m.reason,
        }

    return {
        "schema": "llm-eval-gate/v1",
        "passed": not report.failed,
        "exit_code": report.exit_code,
        "baseline": {
            "path": report.baseline.path,
            "model_name": report.baseline.model_name,
            "git_hash": report.baseline.git_hash,
            "harness_version": report.baseline.harness_version,
            "tasks": report.baseline.task_names,
        },
        "candidate": {
            "path": report.candidate.path,
            "model_name": report.candidate.model_name,
            "git_hash": report.candidate.git_hash,
            "harness_version": report.candidate.harness_version,
            "tasks": report.candidate.task_names,
        },
        "config": {
            "max_rel_regression": report.config.max_rel_regression,
            "max_abs_regression": report.config.max_abs_regression,
            "sigma": report.config.sigma,
            "require_significance": report.config.require_significance,
            "fail_on_missing_task": report.config.fail_on_missing_task,
            "fail_on_missing_metric": report.config.fail_on_missing_metric,
            "fail_on_new_task": report.config.fail_on_new_task,
        },
        "summary": {
            "tasks": len(report.tasks),
            "metrics": len(report.metrics),
            "regressions": len(report.regressions),
            "improvements": len(report.improvements),
            "missing_tasks": list(report.missing_tasks),
            "new_tasks": list(report.new_tasks),
        },
        "tasks": [
            {
                "task": task.task,
                "status": task.status,
                "metrics": [metric_dict(m) for m in task.metrics],
            }
            for task in report.tasks
        ],
        "regressions": [metric_dict(m) for m in report.regressions],
    }


def _headline(report: ComparisonReport) -> str:
    if report.failed:
        failing = len(report.regressions)
        detail = f"{failing} regression{'s' if failing != 1 else ''}"
        if report.config.fail_on_missing_task and report.missing_tasks:
            detail += f", {len(report.missing_tasks)} missing task(s)"
        if report.config.fail_on_new_task and report.new_tasks:
            detail += f", {len(report.new_tasks)} new task(s)"
        return f"FAIL — {detail}"
    return "PASS — no regressions beyond threshold"


def render_text(report: ComparisonReport) -> str:
    """Human-readable terminal output."""
    lines: list[str] = []
    lines.append(f"llm-eval-gate: {_headline(report)}")
    lines.append("")

    baseline_name = report.baseline.model_name or report.baseline.path
    candidate_name = report.candidate.model_name or report.candidate.path
    lines.append(f"  baseline : {baseline_name}")
    lines.append(f"  candidate: {candidate_name}")
    if report.baseline.git_hash or report.candidate.git_hash:
        lines.append(
            f"  git      : {(report.baseline.git_hash or '?')[:8]} -> "
            f"{(report.candidate.git_hash or '?')[:8]}"
        )
    lines.append("")

    failing = {id(m) for m in report.regressions}
    interesting = [
        m
        for m in report.metrics
        if m.status in (STATUS_REGRESSION, STATUS_IMPROVEMENT, STATUS_MISSING, STATUS_NEW)
    ]
    if not interesting:
        lines.append("  All compared metrics are within tolerance.")
    else:
        width_task = max(len(m.task) for m in interesting)
        width_metric = max(len(m.metric) for m in interesting)
        for m in interesting:
            icon = _STATUS_ICON.get(m.status, " ")
            base = _fmt(m.baseline)
            cand = _fmt(m.candidate)
            lines.append(
                f"  {icon} {m.task:<{width_task}}  {m.metric:<{width_metric}}  "
                f"{base:>10} -> {cand:>10}  {_fmt_delta(m):>9}  {_stderr_note(m)}"
            )

    for task in report.tasks:
        if task.status == STATUS_MISSING:
            lines.append(f"  ⚠️  task missing from candidate: {task.task}")
        elif task.status == STATUS_NEW and config_allows_new(report):
            lines.append(f"  🆕 task only present in candidate: {task.task}")

    lines.append("")
    lines.append(
        f"  {len(report.metrics)} metrics compared · "
        f"{len(report.regressions)} regression(s) · "
        f"{len(report.improvements)} improvement(s)"
    )
    if failing:
        lines.append("")
        lines.append("  Regressions:")
        for m in report.regressions:
            lines.append(f"    - {m.task}/{m.metric}: {m.reason}")
    return "\n".join(lines)


def config_allows_new(report: ComparisonReport) -> bool:
    return report.config.fail_on_new_task


def render_markdown(report: ComparisonReport) -> str:
    """Markdown suitable for a pull-request comment."""
    lines: list[str] = []
    headline = "❌ **FAIL**" if report.failed else "✅ **PASS**"
    lines.append(f"## LLM eval gate — {headline}")
    lines.append("")
    lines.append(
        f"`{(report.baseline.git_hash or 'baseline')[:8]}` → "
        f"`{(report.candidate.git_hash or 'candidate')[:8]}` · "
        f"{len(report.metrics)} metrics · {len(report.regressions)} regression(s) · "
        f"{len(report.improvements)} improvement(s)"
    )
    lines.append("")

    if report.regressions:
        lines.append("### Regressions")
        lines.append("")
        lines.append("| Task | Metric | Baseline | Candidate | Δ | Detail |")
        lines.append("| --- | --- | ---: | ---: | ---: | --- |")
        for m in report.regressions:
            lines.append(
                f"| `{m.task}` | `{m.metric}` | {_fmt(m.baseline)} | {_fmt(m.candidate)} | "
                f"{_fmt_delta(m)} | {_stderr_note(m)} |"
            )
        lines.append("")

    lines.append("<details><summary>All compared metrics</summary>")
    lines.append("")
    lines.append("| Task | Metric | Baseline | Candidate | Δ | Status |")
    lines.append("| --- | --- | ---: | ---: | ---: | --- |")
    for m in report.metrics:
        icon = _STATUS_ICON.get(m.status, "")
        lines.append(
            f"| `{m.task}` | `{m.metric}` | {_fmt(m.baseline)} | {_fmt(m.candidate)} | "
            f"{_fmt_delta(m)} | {icon} {m.status} |"
        )
    lines.append("")
    lines.append("</details>")
    lines.append("")

    missing = report.missing_tasks
    if missing:
        lines.append(
            "> ⚠️ Tasks missing from the candidate run: "
            + ", ".join(f"`{t}`" for t in missing)
        )
        lines.append("")

    lines.append(
        f"<sub>Gate: max relative regression {report.config.max_rel_regression:.1%}, "
        f"σ={report.config.sigma:g}, "
        f"require significance: {'yes' if report.config.require_significance else 'no'}</sub>"
    )
    return "\n".join(lines)


def render_json(report: ComparisonReport) -> str:
    """Machine-readable output for downstream tooling."""
    return json.dumps(report_to_dict(report), indent=2, sort_keys=False)


def render_github(report: ComparisonReport) -> str:
    """GitHub Actions workflow commands (annotations) plus a terse summary."""
    lines: list[str] = []
    for m in report.regressions:
        message = (
            f"{m.task}/{m.metric}: {_fmt(m.baseline)} -> {_fmt(m.candidate)} "
            f"({_fmt_delta(m)}) — {_stderr_note(m)}"
        )
        lines.append(f"::error title=LLM eval regression::{_escape(message)}")
    if report.config.fail_on_missing_task:
        for task in report.missing_tasks:
            lines.append(
                f"::error title=Missing eval task::{_escape(f'{task} is missing from the candidate run')}"
            )
    if report.config.fail_on_new_task:
        for task in report.new_tasks:
            lines.append(
                f"::error title=Unexpected new eval task::{_escape(f'{task} only exists in the candidate run')}"
            )
    for m in report.improvements:
        lines.append(
            f"::notice title=LLM eval improvement::{_escape(f'{m.task}/{m.metric}: {_fmt_delta(m)}')}"
        )
    lines.append(_headline(report))
    return "\n".join(lines)


def _escape(message: str) -> str:
    return message.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


RENDERERS = {
    "text": render_text,
    "markdown": render_markdown,
    "md": render_markdown,
    "json": render_json,
    "github": render_github,
}
