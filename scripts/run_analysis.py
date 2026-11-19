"""Run the e-commerce growth incrementality analysis.

The public Criteo uplift benchmark is a randomized marketing-touch experiment.  We
use it as a transparent proxy for an e-commerce platform's promotion/product-touch
decision: predict heterogeneous response, then spend only on users with
positive incremental conversion rather than users who were already likely to
convert.
"""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split


SEED = 20261119
VALUE_PER_VISIT = 5.0
COST_PER_IMPRESSION = 0.50


def fit_rf(X: pd.DataFrame, y: pd.Series, sample_weight: np.ndarray | None = None) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=240,
        min_samples_leaf=25,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=SEED,
    )
    model.fit(X, y, sample_weight=sample_weight)
    return model


def potential_outcomes(frame: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray, features: list[str], outcome: str = "visit"):
    train = frame.iloc[train_idx]
    test = frame.iloc[test_idx]
    X_train, X_test = train[features], test[features]
    t_train, t_test = train["treatment"].to_numpy(), test["treatment"].to_numpy()
    y_train = train[outcome].to_numpy()

    # T-learner: separate response models for exposed and unexposed users.
    models: dict[int, RandomForestClassifier] = {}
    for treatment in (0, 1):
        mask = t_train == treatment
        y = y_train[mask]
        weights = np.where(y == 1, 12.0, 1.0)
        models[treatment] = fit_rf(X_train.loc[mask], pd.Series(y), weights)
    t0 = models[0].predict_proba(X_test)[:, 1]
    t1 = models[1].predict_proba(X_test)[:, 1]
    tau_t = t1 - t0

    # S-learner: one model with treatment as a feature, a useful baseline.
    Xs_train = X_train.copy()
    Xs_train["treatment"] = t_train
    Xs_test_0 = X_test.copy()
    Xs_test_0["treatment"] = 0
    Xs_test_1 = X_test.copy()
    Xs_test_1["treatment"] = 1
    s_model = fit_rf(Xs_train, pd.Series(y_train), np.where(y_train == 1, 12.0, 1.0))
    s0 = s_model.predict_proba(Xs_test_0)[:, 1]
    s1 = s_model.predict_proba(Xs_test_1)[:, 1]
    tau_s = s1 - s0
    return test, tau_t, tau_s, models, s_model


def diff_ci(frame: pd.DataFrame, outcome: str) -> dict[str, float]:
    treated = frame.loc[frame.treatment == 1, outcome]
    control = frame.loc[frame.treatment == 0, outcome]
    lift = treated.mean() - control.mean()
    se = np.sqrt(treated.var(ddof=1) / len(treated) + control.var(ddof=1) / len(control))
    return {"treated_rate": float(treated.mean()), "control_rate": float(control.mean()), "lift": float(lift), "ci_low": float(lift - 1.96 * se), "ci_high": float(lift + 1.96 * se)}


def uplift_table(frame: pd.DataFrame, score: np.ndarray, outcome: str, value_per_outcome: float) -> pd.DataFrame:
    d = frame[["treatment", outcome]].copy().reset_index(drop=True)
    d["score"] = score
    p = d.treatment.mean()
    d = d.sort_values("score", ascending=False).reset_index(drop=True)
    rows = []
    for pct in (1, 5, 10, 20, 30, 50, 100):
        n = max(1, int(len(d) * pct / 100))
        top = d.iloc[:n]
        treated = top.loc[top.treatment == 1, outcome]
        control = top.loc[top.treatment == 0, outcome]
        inc_rate = treated.mean() - control.mean() if len(control) else np.nan
        # IPW estimate is stable even when the top slice has few controls.
        ipw_inc = np.mean(top.treatment * top[outcome] / p - (1 - top.treatment) * top[outcome] / (1 - p))
        net_value_per_user = ipw_inc * value_per_outcome - (pct / 100) * COST_PER_IMPRESSION
        rows.append({"target_pct": pct, "n": n, "uplift_rate_diff": inc_rate, "ipw_incremental_rate": ipw_inc, "incremental_per_1000": ipw_inc * 1000, "net_value_per_user": net_value_per_user})
    return pd.DataFrame(rows)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_path = root / "data" / "sample" / "criteo_sample_10000.csv"
    out_dir = root / "reports"
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(data_path)
    features = [f"f{i}" for i in range(12)]
    frame[features] = frame[features].fillna(frame[features].median())
    frame["treatment"] = frame["treatment"].astype(int)
    frame["conversion"] = frame["conversion"].astype(int)
    frame["visit"] = frame["visit"].astype(int)

    train_idx, test_idx = train_test_split(np.arange(len(frame)), test_size=0.35, random_state=SEED, stratify=frame["treatment"])
    test, tau_t, tau_s, models, s_model = potential_outcomes(frame, train_idx, test_idx, features)
    test = test.reset_index(drop=True)
    baseline = {outcome: diff_ci(frame, outcome) for outcome in ("visit", "conversion")}
    metrics_t = uplift_table(test, tau_t, "visit", VALUE_PER_VISIT)
    metrics_s = uplift_table(test, tau_s, "visit", VALUE_PER_VISIT)
    conversion_t = uplift_table(test, tau_t, "conversion", 35.0)
    metrics_t.to_csv(out_dir / "metrics_t_learner_visit.csv", index=False)
    metrics_s.to_csv(out_dir / "metrics_s_learner_visit.csv", index=False)
    conversion_t.to_csv(out_dir / "metrics_t_learner_conversion.csv", index=False)

    # A transparent model comparison at the 10% targeting budget.
    comparison = pd.DataFrame([
        {"model": "T-learner(RandomForest)", "target_pct": 10, **metrics_t.iloc[2].to_dict()},
        {"model": "S-learner(RandomForest)", "target_pct": 10, **metrics_s.iloc[2].to_dict()},
    ])
    comparison.to_csv(out_dir / "model_comparison.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(metrics_t.target_pct, metrics_t.ipw_incremental_rate * 100, marker="o", label="T-learner")
    ax.plot(metrics_s.target_pct, metrics_s.ipw_incremental_rate * 100, marker="o", label="S-learner")
    ax.axhline(0, color="#555", linewidth=0.8)
    ax.set(xlabel="Targeted users (%)", ylabel="Estimated incremental conversion (pp)", title="Incremental conversion by targeting depth")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "uplift_curve.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(metrics_t.target_pct, metrics_t.net_value_per_user, marker="o", color="#e45756")
    ax.axhline(0, color="#555", linewidth=0.8)
    ax.set(xlabel="Targeted users (%)", ylabel="Estimated net value / eligible user", title="Budget policy curve (conversion value $35, impression cost $0.50)")
    fig.tight_layout()
    fig.savefig(fig_dir / "policy_value_curve.png", dpi=180)
    plt.close(fig)

    perm = permutation_importance(models[1], test[features], test["visit"], n_repeats=5, random_state=SEED, n_jobs=-1, scoring="roc_auc")
    importance = pd.DataFrame({"feature": features, "importance_mean": perm.importances_mean, "importance_std": perm.importances_std}).sort_values("importance_mean", ascending=False)
    importance.to_csv(out_dir / "feature_importance.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    top = importance.head(8).sort_values("importance_mean")
    ax.barh(top.feature, top.importance_mean, color="#4c78a8")
    ax.set(xlabel="Permutation importance (ROC-AUC drop)", title="Exposed-group conversion model: behavior feature signal")
    fig.tight_layout()
    fig.savefig(fig_dir / "feature_importance.png", dpi=180)
    plt.close(fig)

    summary = {
        "n_rows": int(len(frame)),
        "treatment_rate": float(frame.treatment.mean()),
        "conversion_rate": float(frame.conversion.mean()),
        "visit_rate": float(frame.visit.mean()),
        "baseline": baseline,
        "best_10pct_policy_visit": metrics_t.iloc[2].to_dict(),
        "best_10pct_policy_conversion": conversion_t.iloc[2].to_dict(),
        "model_comparison_10pct": comparison.to_dict(orient="records"),
        "data_note": "Results use a 3,000-row stratified-by-offset sample fetched from the public Criteo uplift benchmark; rerun fetch_hf_sample.py for the reproducible sample and scale the same pipeline to the full split.",
    }
    (out_dir / "analysis_results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 决策备忘录：电商用户增量转化与机器学习分层",
        "",
        "## 业务问题",
        "在电商平台的商品促销、用户召回和商品触达中，预算不应只投给预测上最容易购买的人，而应优先投给触达后会产生额外购买的人。项目把决策拆成：是否有增量、谁有增量、预算投到多少比例。",
        "",
        "## 结果（公开代理数据）",
        f"- 样本量：{len(frame):,}；触达组占比：{frame.treatment.mean():.1%}；购买转化率：{frame.conversion.mean():.2%}。",
        f"- 全量 ITT：商品访问增量 {baseline['visit']['lift'] * 100:.2f}pp（95% CI {baseline['visit']['ci_low'] * 100:.2f}–{baseline['visit']['ci_high'] * 100:.2f}pp）；购买转化增量 {baseline['conversion']['lift'] * 100:.2f}pp（95% CI {baseline['conversion']['ci_low'] * 100:.2f}–{baseline['conversion']['ci_high'] * 100:.2f}pp）。",
        f"- T-learner 选取预测增量最高的 10% 用户时，IPW 估计增量商品访问约 {metrics_t.iloc[2].ipw_incremental_rate * 100:.2f}pp/用户；按单次增量商品访问价值 $5、触达成本 $0.50 的情景，策略净价值估计为 ${metrics_t.iloc[2].net_value_per_user:.3f}/合格用户。",
        f"- 同一人群的购买转化增量估计为 {conversion_t.iloc[2].ipw_incremental_rate * 100:.2f}pp；购买事件样本稀疏，因此只作方向性 guardrail，不能替代线上 holdout。",
        "",
        "## 建议",
        "先以 10% 分层人群做小流量 holdout，保留随机对照组；同时监控增量购买、触达成本、频控和自然购买被重复触达的比例。上线前用平台真实的触达成本、订单毛利和人群特征重新校准策略价值。",
        "",
        "## 解释边界",
        "Criteo 是公开随机营销触达实验；f0–f11 是匿名特征，不能把它们解释成具体商品或用户标签。因果结论来自随机化，机器学习分层只用于估计异质性，最终仍需线上 holdout 验证。",
    ]
    (out_dir / "decision_memo.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
