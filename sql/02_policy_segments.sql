-- Join model scores back to a user-level mart and inspect budget tiers.
-- `reports/policy_scores.csv` is produced when the full pipeline is extended
-- to score every eligible user.  The same query works with DuckDB or SparkSQL.
WITH scored AS (
    SELECT * FROM read_csv_auto('reports/policy_scores.csv')
), tiers AS (
    SELECT
        *,
        NTILE(10) OVER (ORDER BY uplift_score DESC) AS uplift_decile
    FROM scored
)
SELECT
    uplift_decile,
    COUNT(*) AS users,
    AVG(uplift_score) AS avg_predicted_uplift,
    AVG(CASE WHEN uplift_decile = 1 THEN 1 ELSE 0 END) AS top_decile_flag
FROM tiers
GROUP BY uplift_decile
ORDER BY uplift_decile;
