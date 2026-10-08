# Groundtruth

The 'Groundtruth' project ensures the reliability of AI retrieval systems through rigorous evaluation against a managed golden dataset. 

## Benchmark Table
| Metric | Current Recall | Target Recall |
| :--- | :--- | :--- |
| **Retrieval@k** | 0.85 | 0.95 |

## Tradeoff
The threshold was set to a 0.05 tolerance to balance strict quality control with the reality of stochastic retrieval. We accept minor variance in semantic matches while blocking breaking changes in critical business logic queries.

## Honest Failure Notes
The system struggles with **context-dependent metaphorical queries**. For instance, queries referencing idioms or domain-specific slang that differs from standard technical literature are frequently misaligned, leading to low retrieval precision.

