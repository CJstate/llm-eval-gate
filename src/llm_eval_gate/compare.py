"""The comparison engine: turn two runs into a pass/fail gate decision.

Design notes
------------
*Metrics on different scales.*  ``acc`` lives in 0-1, ``word_perplexity`` in
0-∞.  Comparing raw differences would either drown perplexity regressions or
flood the report with noise, so the primary decision variable is the change
*relative to the baseline magnitude*.

*Direction.*  Higher ``acc`` is good; higher ``perplexity`` is bad.  Direction is
auto-detected from the metric name (see :func:`is_lower_is_better`) and can be
overridden per metric in configuration.

*Noise.*  When both runs report standard errors, a change within
``sigma * sqrt(se_baseline**2 + se_candidate**2)`` is not statistically
distinguishable from run-to-run noise, and by default cannot fail the gate.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Iterable

from .config import GateConfig
from .loader import MetricResult, RunResults

__all__ = [
    "ComparisonReport",
    "MetricComparison",
    "TaskComparison",
    "compare_runs",
    "is_lower_is_better",
]

#: Metric name tokens that mean "smaller is better".
_LOWER_TOKENS = frozenset({"ppl", "perplexity", "loss", "bpw", "wer", "cer", "ter"})

#: Substrings that mean "smaller is better" even when tokenisation splits them.
_LOWER_SUBSTRINGS = ("perplexity", "bits_per_byte", "error_rate")

_TOKEN_RE = re.compile(r"[^a-z0-9]+")

#: Statuses a metric comparison can carry.
STATUS_REGRESSION = "regression"
STATUS_IMPROVEMENT = "improvement"
STATUS_PASS = "pass"
STATUS_MISSING = "missing"
STATUS_NEW = "new"
STATUS_SKIPPED = "skipped"


def is_lower_is_better(metric: str) -> bool:
    """Heuristically decide whether a metric improves as its value decreases.

    >>> is_lower_is_better("acc")
    False
    >>> is_lower_is_better("word_perplexity")
    True
    >>> is_lower_is_better("bits_per_byte")
    True
    """
    lowered = metric.lower()
    if any(fragment in lowered for fragment in _LOWER_SUBSTRINGS):
        return True
    return any(token in _LOWER_TOKENS for token in _TOKEN_RE.split(lowered) if token)


@dataclass(frozen=True)
class MetricComparison:
    """What happened to one metric between baseline and candidate."""

    task: str
    metric: str
    status: str
    baseline: float | None = None
    candidate: float | None = None
    baseline_stderr: float | None = None
    candidate_stderr: float | None = None
    delta: float | None = None
    rel_change: float | None = None
    worsened_by: float | None = None
    significant: bool | None = None
    threshold: float | None = None
    noise_band: float | None = None
    lower_is_better: bool | None = None
    reason: str = ""

    @property
    def regressed(self) -> bool:
        return self.status == STATUS_REGRESSION

    @property
    def improved(self) -> bool:
        return self.status == STATUS_IMPROVEMENT

    @property
    def failed(self) -> bool:
        return self.regressed


@dataclass(frozen=True)
class TaskComparison:
    """All metric comparisons for a single task."""

    task: str
    status: str
    metrics: tuple[MetricComparison, ...] = ()

    @property
    def regressions(self) -> tuple[MetricComparison, ...]:
        return tuple(m for m in self.metrics if m.regressed)

    @property
    def improved(self) -> tuple[MetricComparison, ...]:
        return tuple(m for m in self.metrics if m.improved)


@dataclass(frozen=True)
class ComparisonReport:
    """The full gate result."""

    baseline: RunResults
    candidate: RunResults
    tasks: tuple[TaskComparison, ...] = ()
    config: GateConfig = field(default_factory=GateConfig)

    @property
    def metrics(self) -> tuple[MetricComparison, ...]:
        return tuple(m for task in self.tasks for m in task.metrics)

    @property
    def regressions(self) -> tuple[MetricComparison, ...]:
        return tuple(m for m in self.metrics if m.regressed)

    @property
    def improvements(self) -> tuple[MetricComparison, ...]:
        return tuple(m for m in self.metrics if m.improved)

    @property
    def missing_tasks(self) -> tuple[str, ...]:
        return tuple(t.task for t in self.tasks if t.status == STATUS_MISSING)

    @property
    def new_tasks(self) -> tuple[str, ...]:
        return tuple(t.task for t in self.tasks if t.status == STATUS_NEW)

    @property
    def failed(self) -> bool:
        if self.regressions:
            return True
        if self.config.fail_on_missing_task and self.missing_tasks:
            return True
        if self.config.fail_on_new_task and self.new_tasks:
            return True
        return False

    @property
    def exit_code(self) -> int:
        """``0`` when the gate passes, ``1`` when it fails."""
        return 1 if self.failed else 0


def _compare_metric(
    task: str,
    metric: str,
    baseline: MetricResult | None,
    candidate: MetricResult | None,
    config: GateConfig,
) -> MetricComparison:
    lower_is_better = config.direction_for(task, metric)
    if lower_is_better is None:
        lower_is_better = is_lower_is_better(metric)

    if baseline is None and candidate is None:
        return MetricComparison(
            task=task, metric=metric, status=STATUS_SKIPPED, reason="absent from both runs"
        )
    if baseline is None:
        return MetricComparison(
            task=task,
            metric=metric,
            status=STATUS_NEW,
            candidate=candidate.value,
            candidate_stderr=candidate.stderr,
            lower_is_better=lower_is_better,
            reason="metric only present in the candidate run",
        )
    if candidate is None:
        return MetricComparison(
            task=task,
            metric=metric,
            status=STATUS_MISSING,
            baseline=baseline.value,
            baseline_stderr=baseline.stderr,
            lower_is_better=lower_is_better,
            reason="metric missing from the candidate run",
        )

    delta = candidate.value - baseline.value
    magnitude = abs(baseline.value)
    if magnitude <= config.zero_epsilon:
        rel_change: float | None = None
        worsened_by: float | None = None
    else:
        rel_change = delta / magnitude
        # Positive means "moved in the bad direction".
        worsened_by = rel_change if lower_is_better else -rel_change

    noise_band: float | None = None
    significant: bool | None = None
    if baseline.stderr is not None and candidate.stderr is not None:
        pooled = math.hypot(baseline.stderr, candidate.stderr)
        noise_band = config.sigma_for(task, metric) * pooled
        significant = bool(delta == 0) or abs(delta) > noise_band

    threshold = config.threshold_for(task, metric)
    status = STATUS_PASS
    reason = ""

    if worsened_by is None:
        # Baseline is ~0, so relative change is undefined; fall back to an
        # absolute rule when the caller configured one.
        absolute_limit = config.max_abs_for(task, metric)
        if absolute_limit is not None:
            bad_delta = delta if lower_is_better else -delta
            if bad_delta > absolute_limit:
                status = STATUS_REGRESSION
                reason = f"absolute change {bad_delta:+.4g} exceeds {absolute_limit:g} (baseline ≈ 0)"
    elif worsened_by > threshold:
        if config.require_significance_for(task, metric) and significant is False:
            reason = (
                f"change is within the noise band "
                f"(|Δ|={abs(delta):.4g} ≤ {noise_band:.4g})"
            )
        else:
            status = STATUS_REGRESSION
            reason = f"{worsened_by * 100:.2f}% worse than baseline (limit {threshold * 100:.2f}%)"
    elif worsened_by < -threshold:
        status = STATUS_IMPROVEMENT
        reason = f"{-worsened_by * 100:.2f}% better than baseline"

    return MetricComparison(
        task=task,
        metric=metric,
        status=status,
        baseline=baseline.value,
        candidate=candidate.value,
        baseline_stderr=baseline.stderr,
        candidate_stderr=candidate.stderr,
        delta=delta,
        rel_change=rel_change,
        worsened_by=worsened_by,
        significant=significant,
        threshold=threshold,
        noise_band=noise_band,
        lower_is_better=lower_is_better,
        reason=reason,
    )


def _compare_task(
    task: str,
    baseline_metrics: dict[str, MetricResult],
    candidate_metrics: dict[str, MetricResult],
    config: GateConfig,
) -> TaskComparison:
    metric_names = sorted(set(baseline_metrics) | set(candidate_metrics))
    comparisons: list[MetricComparison] = []

    for metric in metric_names:
        if not config.is_tracked(task, metric):
            continue
        comparison = _compare_metric(
            task,
            metric,
            baseline_metrics.get(metric),
            candidate_metrics.get(metric),
            config,
        )
        if comparison.status == STATUS_SKIPPED:
            continue
        comparisons.append(comparison)

    if any(m.regressed for m in comparisons):
        status = STATUS_REGRESSION
    elif any(m.status == STATUS_MISSING for m in comparisons) and config.fail_on_missing_metric:
        status = STATUS_MISSING
    elif any(m.status == STATUS_NEW for m in comparisons):
        status = STATUS_NEW
    else:
        status = STATUS_PASS

    return TaskComparison(task=task, status=status, metrics=tuple(comparisons))


def compare_runs(
    baseline: RunResults,
    candidate: RunResults,
    config: GateConfig | None = None,
) -> ComparisonReport:
    """Compare two normalised runs and produce a gate report."""
    config = config or GateConfig()

    task_names: Iterable[str] = sorted(set(baseline.tasks) | set(candidate.tasks))
    tasks: list[TaskComparison] = []

    for task in task_names:
        if config.include_tasks is not None and task not in config.include_tasks:
            continue
        baseline_metrics = baseline.tasks.get(task)
        candidate_metrics = candidate.tasks.get(task)

        if baseline_metrics is None:
            tasks.append(
                TaskComparison(
                    task=task,
                    status=STATUS_NEW,
                    metrics=tuple(
                        MetricComparison(
                            task=task,
                            metric=metric,
                            status=STATUS_NEW,
                            candidate=result.value,
                            reason="task only present in the candidate run",
                        )
                        for metric, result in sorted(candidate_metrics.items())
                    ),
                )
            )
            continue
        if candidate_metrics is None:
            tasks.append(
                TaskComparison(
                    task=task,
                    status=STATUS_MISSING,
                    metrics=tuple(
                        MetricComparison(
                            task=task,
                            metric=metric,
                            status=STATUS_MISSING,
                            baseline=result.value,
                            reason="task missing from the candidate run",
                        )
                        for metric, result in sorted(baseline_metrics.items())
                    ),
                )
            )
            continue

        tasks.append(_compare_task(task, dict(baseline_metrics), dict(candidate_metrics), config))

    return ComparisonReport(
        baseline=baseline,
        candidate=candidate,
        tasks=tuple(tasks),
        config=config,
    )
