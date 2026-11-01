# EvalGuard

**A zero-dependency evaluation gate that fails CI for real LLM regressions, not statistical noise.**

[![CI](https://github.com/CJstate/EvalGuard/actions/workflows/ci.yml/badge.svg)](https://github.com/CJstate/EvalGuard/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Zero dependencies](https://img.shields.io/badge/dependencies-0-brightgreen.svg)](pyproject.toml)

EvalGuard compares two [`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness)
results files and answers one question: **did this change make the model worse, or did the
numbers just wobble?**

## Try it in 10 seconds without running any evaluations.

The demo compares a candidate checkpoint against a baseline. MMLU moved, but stayed inside its
noise band; GSM8K genuinely regressed:

```text
+---------------------------------+----------+-----------+-------------+--------------+
| Task                            | Baseline | Candidate | Noise Band  | Status       |
+---------------------------------+----------+-----------+-------------+--------------+
| gsm8k/exact_match[strict-match] | 0.7814   | 0.6123    | +/- 0.02433 | REGRESSION   |
| mmlu/acc                        | 0.6518   | 0.619     | +/- 0.05643 | PASS (noise) |
+---------------------------------+----------+-----------+-------------+--------------+
```

```console
$ bash demo/run_demo.sh
```

One command, no API key, no evaluation jobs. It exits successfully **because** EvalGuard correctly
rejected the GSM8K regression, and prints an actionable diagnosis:

```text
gsm8k/exact_match[strict-match] dropped from 0.7814 to 0.6123 (21.64%), outside the
noise band +/- 0.02433 and the 5.00% limit. Check the model checkpoint, tokenizer/chat
template, decoding parameters, and evaluation sample counts; rerun on the same harness
version before merging.
```

MMLU's 0.0328-point move exceeds the relative threshold, but its reported standard errors give a
band of ±0.05643, so EvalGuard does not fail the build on noise. The margin, not just the raw delta,
is what makes this gate useful.

## The problem

Every serious LLM pipeline has a step like this:

```yaml
- run: lm_eval --model hf --tasks arc_easy,hellaswag,gsm8k --output_path new.json
- run: <compare new.json to baseline.json>
```

The comparison is usually a hand-rolled diff or a spreadsheet, and it fails in both directions:

- **False alarms.** `acc` moves 0.75 → 0.74 on a 200-sample task. The harness already reports
  `acc_stderr = 0.03`; the difference is one sample. A naive `if new < old: fail` blocks an
  unrelated refactor, and people learn to ignore the gate.
- **Missed regressions.** `word_perplexity` moves 12.5 → 13.4 with a small stderr. Every metric
  lives on a different scale, so a single "allow 2%" rule is either too loose for perplexity or
  too tight for accuracy.

Both come from comparing raw numbers while ignoring the uncertainty the harness already computed.

## How the gate decides

For every metric of every task, EvalGuard computes movement in the metric's bad direction,
compares it with a relative threshold, and checks it against a pooled standard-error noise band:

```text
band        = sigma * sqrt(se_baseline^2 + se_candidate^2)
significant = |candidate - baseline| > band
```

| Condition | Verdict |
| --- | --- |
| Worsened beyond `max_rel_regression`, and significant | **REGRESSION → exit 1** |
| Worsened beyond `max_rel_regression`, but inside the noise band | **PASS (noise)** |
| Improved beyond the threshold | improvement (never fails) |
| Baseline is ~0 (relative change undefined) | falls back to `max_abs_regression` |

## Install

Zero runtime dependencies, standard library only.

```console
$ python -m pip install evalguard
```

From source:

```console
$ git clone https://github.com/CJstate/EvalGuard
$ cd EvalGuard
$ python -m pip install -e ".[dev]"
```

## Quickstart

```console
$ evalguard baseline.json candidate.json
```

Exit code `1` fails the CI step. A passing run — including a metric that dropped but stayed
inside its noise band — exits `0`.

### Output formats

```console
$ evalguard baseline.json candidate.json --format markdown --output report.md
$ evalguard baseline.json candidate.json --format json
$ evalguard baseline.json candidate.json --format github   # ::error:: PR annotations
$ evalguard baseline.json candidate.json --json-output     # clean JSON on stdout
$ evalguard baseline.json candidate.json --json-output guard.json
```

`--json-output PATH` writes the stable `evalguard/v1` JSON while preserving normal terminal or
GitHub output. `--json-output` without a path emits only JSON on stdout, with no ANSI codes.
`--color auto|always|never` controls dependency-free ANSI coloring; `auto` colors only a TTY and
honors `NO_COLOR`.

## The noise band, concretely

Both rows below are a `0.75 → 0.69` drop on `arc_easy/acc` (−8%, well past the 5% limit).
Only the reported standard error differs:

| Baseline SE | Candidate SE | Pooled band (σ=2) | Verdict |
| ---: | ---: | ---: | --- |
| 0.02 | 0.02 | 0.057 | regression — the 0.06 drop is bigger than the band |
| 0.02 | 0.25 | 0.502 | pass — a 0.06 drop on a 0.50 band is noise |

This is covered by `tests/fixtures/candidate_noisy.json` and the compare tests.

## Configuration

CLI flags cover simple cases; TOML is easier to review and share:

```toml
[gate]
max_rel_regression = 0.05   # relative movement in the bad direction that fails
sigma = 2.0                 # noise-band multiplier; higher = more forgiving
require_significance = true # never fail a change that is inside the band
fail_on_missing_metric = true

# Most specific key wins: "task.metric" > "task" > [gate] defaults.
[overrides."arc_easy.acc"]
max_rel_regression = 0.02

[overrides."gsm8k.exact_match"]
sigma = 3.0                 # math benchmarks are high-variance

[overrides."wikitext.word_perplexity"]
max_rel_regression = 0.01
```

```console
$ evalguard baseline.json candidate.json --config gate.toml
```

See [`gate.example.toml`](gate.example.toml) for every field. CLI flags override the file, so one
shared policy works locally and in CI.

### Direction detection

Perplexity-style metrics must go **down**; accuracy-style metrics must go **up**. EvalGuard infers
this from metric names (`perplexity`, `bits_per_byte`, `error_rate`, and `ppl / loss / bpw / wer /
cer / ter` are lower-is-better). Override when needed:

```console
$ evalguard b.json c.json --lower-is-better latency_ms --higher-is-better exact_match
```

## Use as a GitHub Action

This repository is itself a composite action — no build, no Node, no Docker.

```yaml
name: eval regression gate
on: [pull_request]

jobs:
  gate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run candidate evals
        run: |
          lm_eval --model hf --model_args pretrained=${{ github.event.pull_request.head.sha }} \
                  --tasks arc_easy,hellaswag --output_path candidate.json

      - id: gate
        uses: CJstate/EvalGuard@v0.1.0
        with:
          baseline: evals/baseline.json
          candidate: candidate.json
          config: gate.toml
          report-path: evalguard-report.md
          json-output: evalguard-report.json

      - name: Comment on the PR
        if: always() && steps.gate.outputs.passed == 'false'
        uses: marocchino/sticky-pull-request-comment@v2
        with:
          path: evalguard-report.md

      - name: Upload machine-readable report
        if: always()
        run: echo '${{ steps.gate.outputs.json }}' > gate-summary.json
```

The action captures the gate's exit code and exposes:

| Output | Meaning |
| --- | --- |
| `passed` | `true` when the gate passed, `false` on regression or error |
| `exit-code` | `0` pass, `1` regression, `2` input error |
| `report-path` | Markdown report path |
| `json` | Compact `evalguard/v1` JSON document |
| `json-path` | JSON report path |

The gate step still exits non-zero, so downstream steps should use `if: always()` as above.

## Exit codes

| Code | Meaning |
| ---: | --- |
| `0` | the gate passed |
| `1` | a regression (or another configured failure) was detected |
| `2` | usage or input error — unreadable/invalid results file, bad config |

## Design notes

- **Verdicts and presentation are separate.** `compare_runs()` returns a plain
  `ComparisonReport`; text, Markdown, JSON, and GitHub renderers are pure functions of it. Every
  verdict is testable without capturing stdout.
- **Missing data is explicit.** A task absent from the candidate run, a disappeared metric, and a
  candidate-only task are distinct states with their own flags. Treating "absent" as "unchanged"
  is how regressions hide.
- **Harness formats are normalized carefully.** `acc,none` and `acc_stderr,none` become `acc` plus
  its stderr; filtered variants like `acc,strict-match` keep their filter instead of overwriting
  the plain metric; non-numeric entries, booleans, and `NaN`/`Inf` are dropped.
- **Never crash on a wall of Unicode.** Windows consoles still use legacy code pages, so the CLI
  degrades markers to ASCII when stdout cannot encode them (`--ascii` forces it).

## Limitations

- The noise band is only as good as the standard errors in the results file. Sparse tasks
  (`limit: 10`) report optimistic stderr — pass `--sigma 3` or add an override.
- Metrics are not independent (e.g. `acc` and `acc_norm` correlate strongly); EvalGuard does not
  correct for multiple comparisons.
- It compares results files only. It does not run evaluations, shard them, or normalize across
  harness versions.
- `tomllib` is standard library from Python 3.11.

## Development

```console
$ python -m pip install -e ".[dev]"
$ python -m pytest -q          # 107 tests, no network, no API keys
$ bash demo/run_demo.sh
```

## License

[MIT](LICENSE)

---

Prove your AI runs are tamper-evident with
[Veritas-Harness](https://github.com/CJstate/Veritas-Harness) — a zero-dependency agent harness
with hash-chained, byte-level replay.
