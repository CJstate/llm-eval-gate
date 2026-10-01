# llm-eval-gate

**A statistical gate for LLM evaluation runs: fail CI when a benchmark actually regressed, pass when it only moved within the noise.**

[![CI](https://github.com/CJstate/llm-eval-gate/actions/workflows/ci.yml/badge.svg)](https://github.com/CJstate/llm-eval-gate/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Zero dependencies](https://img.shields.io/badge/dependencies-0-brightgreen.svg)](pyproject.toml)

`llm-eval-gate` compares two [`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness)
results files and answers one question: **did this change make the model worse, or did
the numbers just wobble?**

It exits non-zero only for changes that are both **large relative to the baseline** and
**outside the run-to-run noise band implied by the harness's own standard errors**.

---

## The problem

Every serious LLM pipeline has this step somewhere:

```yaml
- run: lm_eval --model hf --tasks arc_easy,hellaswag,gsm8k --output_path new.json
- run: <compare new.json to baseline.json>
```

The comparison is usually a hand-rolled diff or a spreadsheet, and it fails in both
directions:

- **False alarms.** `acc` moves 0.75 → 0.74 on a 200-sample task. The harness already
  reports `acc_stderr = 0.03`; the difference is one sample. A naive
  `if new < old: fail` blocks an unrelated refactor and people learn to ignore the gate.
- **Missed regressions.** `word_perplexity` moves 12.5 → 13.4 with a small stderr. Every
  metric lives on a different scale, so a single "allow 2% change" rule is either too
  loose for perplexity or too tight for accuracy.

Both come from the same mistake: comparing raw numbers while ignoring the uncertainty the
harness already computed for you, and assuming one threshold fits every metric.

## What this does

For every metric of every task, the gate computes the movement in the direction that is
actually bad for that metric, compares it against a per-task/per-metric relative
threshold, and checks it against a pooled standard-error noise band:

```
band      = sigma * sqrt(se_baseline² + se_candidate²)
significant = |candidate - baseline| > band
```

Then:

| Condition | Verdict |
| --- | --- |
| worsened by more than `max_rel_regression`, and significant | **regression → exit 1** |
| worsened by more than `max_rel_regression`, but inside the noise band | pass (reported as "within the noise band") |
| improved by more than the threshold | improvement (never fails) |
| baseline is ~0 (relative change undefined) | falls back to `max_abs_regression` |

## Install

Zero runtime dependencies — standard library only.

```bash
pip install llm-eval-gate
```

Or from source:

```bash
git clone https://github.com/CJstate/llm-eval-gate
cd llm-eval-gate
pip install -e ".[dev]"
```

## Quickstart

```bash
llm-eval-gate baseline.json candidate.json
```

```
llm-eval-gate: FAIL - 2 regressions

  baseline : hf
  candidate: hf
  git      : 11111111 -> 22222222

  ❌ arc_easy  acc                    0.75 ->       0.69     -8.00%  8.00% worse than baseline (limit 5.00%)
  ❌ wikitext  byte_perplexity         1.8 ->       2.05    +13.89%  13.89% worse than baseline (limit 5.00%)

  4 metrics compared | 2 regression(s) | 0 improvement(s)
```

Exit code `1`, so the CI step fails. For a passing run — including one where a metric
*dropped* but stayed inside its noise band — the same command exits `0`.

### Formats

```bash
llm-eval-gate baseline.json candidate.json --format markdown --output report.md
llm-eval-gate baseline.json candidate.json --format json     # stable llm-eval-gate/v1 schema
llm-eval-gate baseline.json candidate.json --format github   # ::error:: annotations on the PR diff
```

### The noise band, concretely

Both of these are a `0.75 → 0.69` drop on `arc_easy/acc` (−8%, well past the 5% limit).
The standard error in the results file is the only difference:

| Baseline SE | Candidate SE | Pooled band (σ=2) | Verdict |
| ---: | ---: | ---: | --- |
| 0.02 | 0.02 | 0.057 | ❌ regression — the 0.06 drop is bigger than the band |
| 0.02 | 0.25 | 0.502 | ➖ pass — a 0.06 drop on a 0.50 band is noise |

That second row is the whole point. It is covered by
`tests/fixtures/candidate_noisy.json` and `tests/test_compare.py`.

## Example report

The Markdown renderer produces something you can post straight onto a pull request
(full file: [`docs/example-report.md`](docs/example-report.md)):

> ## LLM eval gate — ❌ **FAIL**
>
> `11111111` → `22222222` · 4 metrics · 2 regression(s) · 0 improvement(s)
>
> ### Regressions
>
> | Task | Metric | Baseline | Candidate | Δ | Detail |
> | --- | --- | ---: | ---: | ---: | --- |
> | `arc_easy` | `acc` | 0.75 | 0.69 | -8.00% | 8.00% worse than baseline (limit 5.00%) |
> | `wikitext` | `byte_perplexity` | 1.8 | 2.05 | +13.89% | 13.89% worse than baseline (limit 5.00%) |
>
> <sub>Gate: max relative regression 5.0%, σ=2, require significance: yes</sub>

## Configuration

CLI flags are enough for simple cases; a TOML file is easier to review and share:

```toml
[gate]
max_rel_regression = 0.05   # relative movement in the bad direction that fails
sigma = 2.0                 # noise-band multiplier; higher = more forgiving
require_significance = true # never fail a change that is inside the band
fail_on_missing_metric = true

# The most specific key wins: "task.metric" > "task" > [gate] defaults.
[overrides."arc_easy.acc"]
max_rel_regression = 0.02

[overrides."gsm8k.exact_match"]
sigma = 3.0                 # math benchmarks are high-variance

[overrides."wikitext.word_perplexity"]
max_rel_regression = 0.01
```

```bash
llm-eval-gate baseline.json candidate.json --config gate.toml
```

See [`gate.example.toml`](gate.example.toml) for every field. Any CLI flag overrides the
file, which makes one shared policy usable from both a local run and CI.

### Direction detection

Perplexity-style metrics must go **down**; accuracy-style metrics must go **up**. The gate
infers this from the metric name (`perplexity`, `bits_per_byte`, `error_rate`, or a token
in `ppl / loss / bpw / wer / cer / ter` are lower-is-better), and you can always override:

```bash
llm-eval-gate b.json c.json --lower-is-better latency_ms --higher-is-better exact_match
```

## Use as a GitHub Action

The repository is itself a composite action — no build, no Node, no Docker.

```yaml
name: eval regression gate
on: [pull_request]

jobs:
  gate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run the candidate evals
        run: |
          lm_eval --model hf --model_args pretrained=${{ github.event.pull_request.head.sha }} \
                  --tasks arc_easy,hellaswag --output_path candidate.json

      - id: gate
        uses: CJstate/llm-eval-gate@v0.1.0
        with:
          baseline: evals/baseline.json
          candidate: candidate.json
          config: gate.toml
          report-path: gate-report.md

      - name: Comment on the PR
        if: always() && steps.gate.outputs.passed == 'false'
        uses: marocchino/sticky-pull-request-comment@v2
        with:
          path: gate-report.md
```

The action exposes `passed`, `exit-code`, and `report-path` as outputs, so the comment
step still runs (using `if: always()`) after the gate fails the job.

## Exit codes

The exit code is the contract CI depends on:

| Code | Meaning |
| ---: | --- |
| `0` | the gate passed |
| `1` | a regression (or another configured failure) was detected |
| `2` | usage or input error — unreadable/invalid results file, bad config |

## Design notes

- **The gate and its presentation are separate.** `compare_runs()` returns a plain
  `ComparisonReport`; the text/Markdown/JSON/GitHub renderers are pure functions of it.
  Every verdict is testable without capturing stdout, and `--format json` is the same
  verdicts the terminal shows.
- **Missing data is explicit, not silently equal.** A task absent from the candidate run,
  a metric that disappeared, or a task that only the candidate has are all distinct
  states with their own flags — silently treating "absent" as "unchanged" is how
  evaluation regressions hide.
- **The harness format is dirty in known ways.** `acc,none` and `acc_stderr,none` keys are
  normalized into `acc` plus its `stderr`; filtered variants like `acc,strict-match` keep
  their filter as `acc[strict-match]` instead of overwriting the plain metric; non-numeric
  entries (`alias`), booleans, and `NaN`/`Inf` are dropped rather than compared.
- **Never crash on a wall of Unicode.** Windows consoles still default to legacy code
  pages, so the CLI degrades its `❌`/`✅` markers to `[FAIL]`/`[PASS]` when stdout cannot
  encode them (`--ascii` forces it). A gate that dies before printing its verdict is
  worse than no gate.

## Limitations

- The noise band is only as good as the standard errors in the results file. Sparse tasks
  (`limit: 10`) report optimistic stderr — pass `--sigma 3` or add an override.
- Many harness metrics are not independent (e.g. `acc` and `acc_norm` correlate strongly).
  The gate checks each metric separately and does not correct for multiple comparisons.
- It compares results files only. It does not run evaluations, shard them, or normalize
  across different harness versions.
- `tomllib` is stdlib from Python 3.11; older interpreters are not supported.

## Development

```bash
pip install -e ".[dev]"
python -m pytest -q          # 102 tests, no network, no API keys
```

## License

[MIT](LICENSE)
