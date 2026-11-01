## EvalGuard — ❌ **FAIL**

`11111111` → `22222222` · 4 metrics · 2 regression(s) · 0 improvement(s)

### Regressions

| Task | Metric | Baseline | Candidate | Δ | Detail |
| --- | --- | ---: | ---: | ---: | --- |
| `arc_easy` | `acc` | 0.75 | 0.69 | -8.00% | 8.00% worse than baseline (limit 5.00%) |
| `wikitext` | `byte_perplexity` | 1.8 | 2.05 | +13.89% | 13.89% worse than baseline (limit 5.00%) |

<details><summary>All compared metrics</summary>

| Task | Metric | Baseline | Candidate | Δ | Status |
| --- | --- | ---: | ---: | ---: | --- |
| `arc_easy` | `acc` | 0.75 | 0.69 | -8.00% | ❌ regression |
| `hellaswag` | `acc` | 0.42 | 0.415 | -1.19% | ➖ pass |
| `wikitext` | `byte_perplexity` | 1.8 | 2.05 | +13.89% | ❌ regression |
| `wikitext` | `word_perplexity` | 12.5 | 12.6 | +0.80% | ➖ pass |

</details>

<sub>Gate: max relative regression 5.0%, σ=2, require significance: yes</sub>
