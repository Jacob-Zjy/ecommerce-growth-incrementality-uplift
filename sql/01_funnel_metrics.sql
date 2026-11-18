-- DuckDB: funnel baseline and randomization checks.
-- Run from the repository root after `python scripts/fetch_hf_sample.py`.
WITH src AS (
    SELECT * FROM read_csv_auto('data/sample/criteo_sample_10000.csv')
)
SELECT
    treatment,
    COUNT(*) AS users,
    AVG(exposure) AS exposure_rate,
    AVG(visit) AS visit_rate,
    AVG(conversion) AS conversion_rate
FROM src
GROUP BY treatment
ORDER BY treatment;

-- The causal comparison is the treated-minus-control difference in means.
SELECT
    AVG(CASE WHEN treatment = 1 THEN visit END)
      - AVG(CASE WHEN treatment = 0 THEN visit END) AS visit_itt,
    AVG(CASE WHEN treatment = 1 THEN conversion END)
      - AVG(CASE WHEN treatment = 0 THEN conversion END) AS conversion_itt
FROM src;
