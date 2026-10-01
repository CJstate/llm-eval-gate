"""Loading and normalising ``lm-evaluation-harness`` result dumps.

The harness writes a JSON document shaped roughly like::

    {
      "results": {
        "arc_easy": {"acc,none": 0.75, "acc_stderr,none": 0.02, "alias": "arc_easy"},
        "wikitext": {"word_perplexity,none": 12.5, "word_perplexity_stderr,none": 0.3}
      },
      "configs": {"model": "hf", "model_args": "pretrained=..."},
      "git_hash": "deadbeef",
      "lm_eval_version": "0.4.5"
    }

Metric names carry a filter suffix (``acc,none``, ``exact_match,strict-match``)
and standard errors live in sibling keys (``acc_stderr,none``).  This module
flattens that into::

    RunResults.tasks["arc_easy"]["acc"] == MetricResult(value=0.75, stderr=0.02)

so that comparison code never has to think about the harness's key encoding.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

__all__ = [
    "MetricResult",
    "ResultsFormatError",
    "RunResults",
    "load_results",
    "parse_metric_key",
]

STDERR_SUFFIX = "_stderr"

#: Filter suffixes that carry no information and are therefore dropped from
#: metric labels (``acc,none`` is just ``acc``).
_NULL_FILTERS = frozenset({"", "none", "-none", "null"})


class ResultsFormatError(ValueError):
    """Raised when a file is not a recognisable harness results dump."""


@dataclass(frozen=True)
class MetricResult:
    """A single metric value, with its standard error when the harness reported one."""

    value: float
    stderr: float | None = None

    @property
    def has_stderr(self) -> bool:
        return self.stderr is not None


@dataclass(frozen=True)
class RunResults:
    """One evaluated model / task set, normalised for comparison."""

    path: str
    tasks: Mapping[str, Mapping[str, MetricResult]] = field(default_factory=dict)
    model_name: str | None = None
    git_hash: str | None = None
    harness_version: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def task_names(self) -> list[str]:
        return sorted(self.tasks)

    def metrics_for(self, task: str) -> dict[str, MetricResult]:
        return dict(self.tasks.get(task, {}))


def parse_metric_key(raw: str) -> tuple[str, str | None]:
    """Split a harness metric key into ``(metric_name, filter_suffix)``.

    >>> parse_metric_key("acc,none")
    ('acc', None)
    >>> parse_metric_key("exact_match,strict-match")
    ('exact_match', 'strict-match')
    >>> parse_metric_key("acc_stderr")
    ('acc_stderr', None)
    """
    name, _sep, raw_filter = raw.partition(",")
    name = name.strip()
    filter_name = raw_filter.strip()
    return name, (None if filter_name in _NULL_FILTERS else filter_name)


def _label(metric: str, filter_name: str | None) -> str:
    return metric if filter_name is None else f"{metric}[{filter_name}]"


def _as_float(value: Any) -> float | None:
    """Return a finite float, or ``None`` for anything non-numeric/NaN/Inf."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _first_str(*candidates: Any) -> str | None:
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def _parse_task(metrics: Mapping[str, Any]) -> dict[str, MetricResult]:
    values: dict[str, float] = {}
    stderrs: dict[str, float] = {}

    for raw_key, raw_value in metrics.items():
        number = _as_float(raw_value)
        if number is None:
            continue
        metric, filter_name = parse_metric_key(str(raw_key))
        if not metric:
            continue

        if metric.endswith(STDERR_SUFFIX):
            base = metric[: -len(STDERR_SUFFIX)]
            if not base:
                continue
            stderrs[_label(base, filter_name)] = number
        else:
            values[_label(metric, filter_name)] = number

    # A stderr without a matching value carries no comparable information.
    return {
        label: MetricResult(value=value, stderr=stderrs.get(label))
        for label, value in values.items()
    }


def load_results(path: str | Path) -> RunResults:
    """Read a harness results JSON file into :class:`RunResults`.

    Raises :class:`ResultsFormatError` with an actionable message when the file
    is missing, malformed, or has no numeric metrics at all.
    """
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise ResultsFormatError(f"cannot read results file {source}: {exc}") from exc

    try:
        document = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ResultsFormatError(f"{source} is not valid JSON: {exc}") from exc

    if not isinstance(document, dict):
        raise ResultsFormatError(
            f"{source}: expected a JSON object at the top level, got {type(document).__name__}"
        )

    raw_results = document.get("results")
    if not isinstance(raw_results, dict):
        raise ResultsFormatError(
            f"{source}: expected a top-level 'results' object mapping task name to metrics "
            f"(found {type(raw_results).__name__}). Is this an lm-evaluation-harness "
            f"results file?"
        )

    tasks: dict[str, dict[str, MetricResult]] = {}
    for task_name, metrics in raw_results.items():
        if not isinstance(metrics, dict):
            continue
        parsed = _parse_task(metrics)
        if parsed:
            tasks[str(task_name)] = parsed

    if not tasks:
        raise ResultsFormatError(f"{source}: 'results' contained no numeric metrics to compare")

    configs = document.get("configs")
    configs = configs if isinstance(configs, dict) else {}

    return RunResults(
        path=str(source),
        tasks=tasks,
        model_name=_first_str(
            document.get("model_name"),
            configs.get("model_name"),
            configs.get("model"),
        ),
        git_hash=_first_str(document.get("git_hash")),
        harness_version=_first_str(
            document.get("lm_eval_version"),
            document.get("harness_version"),
            configs.get("lm_eval_version"),
        ),
        metadata={
            "date": document.get("date"),
            "configs": configs,
        },
    )
