# Analysis copilot prompt

Use this prompt with Codex/Claude after replacing the two paths with the
current data and score-table paths:

```text
You are a growth data scientist. Read the funnel mart at <MART_PATH> and the
model score table at <SCORE_PATH>. The decision is how to allocate a fixed
exposure budget. First verify grain, treatment ratio, missingness, outcome
definitions, and whether the holdout is randomized. Then report ITT with a
95% confidence interval, compare response-model ranking with uplift ranking,
and quantify incremental visits/conversions at 5%, 10%, 20%, and 50% targeting
depths. State assumptions for value per incremental action and impression cost.
Do not call observational correlations causal. Finish with one recommendation,
one guardrail, and one follow-up experiment.
```
