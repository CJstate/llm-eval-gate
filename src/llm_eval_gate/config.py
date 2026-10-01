"""Gate configuration: thresholds, per-metric overrides, and TOML loading.

A gate decides whether a metric's change counts as a *regression*.  Three knobs
matter:

``max_rel_regression``
    How far a metric may move (as a fraction of its baseline) before it counts.
    Relative movement is used because harness metrics live on wildly different
    scales (``acc`` is 0-1, ``word_perplexity`` is 0-∞).

``sigma``
    Statistical tolerance.  When both runs report a standard error, a change
    smaller than ``sigma * sqrt(se_baseline**2 + se_candidate**2)`` is treated as
    noise and cannot fail the gate.  Defaults to 2 (≈95% confidence).

``require_significance``
    When true (the default), a change that exceeds the relative threshold but is
    *not* distinguishable from noise does not fail the build.

Everything can be overridden globally, per task, or per ``task.metric`` pair.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

__all__ = ["ConfigError", "GateConfig", "MetricOverride", "load_config"]


class ConfigError(ValueError):
    """Raised when a gate configuration is malformed."""


@dataclass(frozen=True)
class MetricOverride:
    """Narrower rules for one task or one ``task.metric`` pair."""

    max_rel_regression: float | None = None
    max_abs_regression: float | None = None
    sigma: float | None = None
    require_significance: bool | None = None
    lower_is_better: bool | None = None


@dataclass(frozen=True)
class GateConfig:
    """Thresholds controlling when a comparison fails."""

    max_rel_regression: float = 0.05
    max_abs_regression: float | None = None
    sigma: float = 2.0
    require_significance: bool = True
    fail_on_missing_task: bool = False
    fail_on_missing_metric: bool = True
    fail_on_new_task: bool = False
    zero_epsilon: float = 1e-12
    ignore_metrics: frozenset[str] = frozenset()
    include_tasks: frozenset[str] | None = None
    include_metrics: frozenset[str] | None = None
    lower_is_better: frozenset[str] = frozenset()
    higher_is_better: frozenset[str] = frozenset()
    overrides: Mapping[str, MetricOverride] = field(default_factory=dict)

    # -- lookups ---------------------------------------------------------

    def override_for(self, task: str, metric: str) -> MetricOverride:
        """Most specific override wins: ``task.metric`` then ``task``."""
        return (
            self.overrides.get(f"{task}.{metric}")
            or self.overrides.get(task)
            or MetricOverride()
        )

    def threshold_for(self, task: str, metric: str) -> float:
        override = self.override_for(task, metric)
        return self.max_rel_regression if override.max_rel_regression is None else override.max_rel_regression

    def max_abs_for(self, task: str, metric: str) -> float | None:
        override = self.override_for(task, metric)
        return self.max_abs_regression if override.max_abs_regression is None else override.max_abs_regression

    def sigma_for(self, task: str, metric: str) -> float:
        override = self.override_for(task, metric)
        return self.sigma if override.sigma is None else override.sigma

    def require_significance_for(self, task: str, metric: str) -> bool:
        override = self.override_for(task, metric)
        return self.require_significance if override.require_significance is None else override.require_significance

    def direction_for(self, task: str, metric: str) -> bool | None:
        """``True`` when lower is better, ``False`` when higher is better, ``None`` to auto-detect."""
        override = self.override_for(task, metric)
        if override.lower_is_better is not None:
            return override.lower_is_better
        if metric in self.higher_is_better:
            return False
        if metric in self.lower_is_better:
            return True
        return None

    def is_tracked(self, task: str, metric: str) -> bool:
        if self.include_tasks is not None and task not in self.include_tasks:
            return False
        if self.include_metrics is not None and metric not in self.include_metrics:
            return False
        if metric in self.ignore_metrics:
            return False
        return True

    # -- construction ----------------------------------------------------

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], *, source: str = "<config>") -> GateConfig:
        if not isinstance(data, Mapping):
            raise ConfigError(f"{source}: expected a TOML table, got {type(data).__name__}")

        gate = data.get("gate", data)
        if not isinstance(gate, Mapping):
            raise ConfigError(f"{source}: [gate] must be a table")

        kwargs: dict[str, Any] = {}
        for key in ("max_rel_regression", "max_abs_regression", "sigma", "zero_epsilon"):
            if key in gate:
                kwargs[key] = _number(gate[key], key, source)
        for key in (
            "require_significance",
            "fail_on_missing_task",
            "fail_on_missing_metric",
            "fail_on_new_task",
        ):
            if key in gate:
                kwargs[key] = _boolean(gate[key], key, source)
        for key in (
            "ignore_metrics",
            "include_tasks",
            "include_metrics",
            "lower_is_better",
            "higher_is_better",
        ):
            if key in gate:
                kwargs[key] = _string_set(gate[key], key, source)

        raw_overrides = data.get("overrides", {})
        if raw_overrides is None:
            raw_overrides = {}
        if not isinstance(raw_overrides, Mapping):
            raise ConfigError(f"{source}: [overrides] must be a table")
        overrides: dict[str, MetricOverride] = {}
        for name, spec in raw_overrides.items():
            if not isinstance(spec, Mapping):
                raise ConfigError(f"{source}: override {name!r} must be a table")
            overrides[str(name)] = MetricOverride(
                max_rel_regression=_optional_number(spec, "max_rel_regression", source),
                max_abs_regression=_optional_number(spec, "max_abs_regression", source),
                sigma=_optional_number(spec, "sigma", source),
                require_significance=_optional_bool(spec, "require_significance", source),
                lower_is_better=_optional_bool(spec, "lower_is_better", source),
            )

        return cls(overrides=overrides, **kwargs)


def load_config(path: str | Path) -> GateConfig:
    """Read a TOML gate configuration from ``path``."""
    source = Path(path)
    try:
        with source.open("rb") as handle:
            data = tomllib.load(handle)
    except OSError as exc:
        raise ConfigError(f"cannot read config {source}: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{source}: invalid TOML: {exc}") from exc
    return GateConfig.from_mapping(data, source=str(source))


def _number(value: Any, key: str, source: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{source}: {key} must be a number, got {value!r}")
    return float(value)


def _optional_number(spec: Mapping[str, Any], key: str, source: str) -> float | None:
    if key not in spec:
        return None
    return _number(spec[key], key, source)


def _boolean(value: Any, key: str, source: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{source}: {key} must be a boolean, got {value!r}")
    return value


def _optional_bool(spec: Mapping[str, Any], key: str, source: str) -> bool | None:
    if key not in spec:
        return None
    return _boolean(spec[key], key, source)


def _string_set(value: Any, key: str, source: str) -> frozenset[str]:
    if isinstance(value, str):
        raise ConfigError(f"{source}: {key} must be a list of strings, got a bare string")
    if not isinstance(value, (list, tuple, set, frozenset)):
        raise ConfigError(f"{source}: {key} must be a list of strings")
    items = []
    for item in value:
        if not isinstance(item, str):
            raise ConfigError(f"{source}: {key} must contain only strings, got {item!r}")
        items.append(item)
    return frozenset(items)
