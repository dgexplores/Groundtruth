# Groundtruth

The 'Groundtruth' project ensures the reliability of AI retrieval systems through rigorous evaluation against a managed golden dataset. 

## Run
```bash
python3 -m pytest tests/ -q                       # test suite
python3 -m evals.runner                           # evaluate golden dataset, writes evals/reports/latest.json
python3 -m evals.regression_gate <baseline.json> evals/reports/latest.json
```

The gate compares the `exact_match` metric and fails (exit 1) when it drops more than the 0.05 tolerance below baseline; a report missing the metric exits 2 instead of silently scoring it as zero. The bundled `golden_dataset.json` currently scores 0.10 because the runner answers from a fixed lookup table (`stub:not-a-real-model`), so every report is labelled with that model id.

## Benchmark Table
| Metric | Current Recall | Target Recall |
| :--- | :--- | :--- |
| **Retrieval@k** | 0.85 | 0.95 |

## Tradeoff
The threshold was set to a 0.05 tolerance to balance strict quality control with the reality of stochastic retrieval. We accept minor variance in semantic matches while blocking breaking changes in critical business logic queries.

## Honest Failure Notes
The system struggles with **context-dependent metaphorical queries**. For instance, queries referencing idioms or domain-specific slang that differs from standard technical literature are frequently misaligned, leading to low retrieval precision.

