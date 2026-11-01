"""EvalGuard — catch LLM evaluation regressions in CI.

Compare two ``lm-evaluation-harness`` result files, apply relative thresholds
with an stderr-aware noise band, and fail the build when a metric regresses::

    from llm_eval_gate import GateConfig, compare_runs, load_results

    baseline = load_results("baseline.json")
    candidate = load_results("candidate.json")
    report = compare_runs(baseline, candidate, GateConfig(max_rel_regression=0.02))

    if report.failed:
        for metric in report.regressions:
            print(metric.task, metric.metric, metric.reason)
        raise SystemExit(report.exit_code)
"""

from __future__ import annotations

from .compare import (
    ComparisonReport,
    MetricComparison,
    TaskComparison,
    compare_runs,
    is_lower_is_better,
)
from .config import ConfigError, GateConfig, MetricOverride, load_config
from .loader import MetricResult, ResultsFormatError, RunResults, load_results
from .report import (
    render_github,
    render_json,
    render_markdown,
    render_text,
    report_to_dict,
)

__version__ = "0.1.0"

__all__ = [
    "ComparisonReport",
    "ConfigError",
    "GateConfig",
    "MetricComparison",
    "MetricOverride",
    "MetricResult",
    "ResultsFormatError",
    "RunResults",
    "TaskComparison",
    "compare_runs",
    "is_lower_is_better",
    "load_config",
    "load_results",
    "render_github",
    "render_json",
    "render_markdown",
    "render_text",
    "report_to_dict",
    "__version__",
]
