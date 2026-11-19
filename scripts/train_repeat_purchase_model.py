"""Train a customer repeat-purchase model on the UCI Online Retail data.

The target is whether a customer places another order in the 60 days after a
fixed cutoff. Features are built only from orders before the cutoff, making the
exercise a forward-looking e-commerce prediction task rather than a leakage-
prone random row split.
"""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split


SEED = 20261119
FEATURES = [
    "order_count",
    "total_revenue",
    "total_items",
    "unique_skus",
    "avg_order_value",
    "avg_items_per_order",
    "active_days",
    "recency_days",
]


def build_customer_table(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    raw = pd.read_csv(path, encoding="latin1")
    raw["InvoiceDate"] = pd.to_datetime(raw["InvoiceDate"], errors="coerce")
    raw["InvoiceNo"] = raw["InvoiceNo"].astype(str)
    raw["is_cancelled"] = raw["InvoiceNo"].str.upper().str.startswith("C")
    raw["Revenue"] = raw["Quantity"] * raw["UnitPrice"]
    raw = raw.dropna(subset=["CustomerID", "InvoiceDate", "InvoiceNo"])
    raw = raw[(~raw.is_cancelled) & (raw["Quantity"] > 0) & (raw["UnitPrice"] > 0)]
    orders = raw.groupby("InvoiceNo", as_index=False).agg(
        customer_id=("CustomerID", "first"),
        order_date=("InvoiceDate", "first"),
        items=("Quantity", "sum"),
        skus=("StockCode", "nunique"),
        revenue=("Revenue", "sum"),
    )
    cutoff = orders["order_date"].max() - pd.Timedelta(60, unit="D")
    history = orders[orders["order_date"] < cutoff].copy()
    future = orders[(orders["order_date"] >= cutoff) & (orders["order_date"] <= orders["order_date"].max())]
    features = history.groupby("customer_id", as_index=False).agg(
        order_count=("order_date", "nunique"),
        total_revenue=("revenue", "sum"),
        total_items=("items", "sum"),
        unique_skus=("skus", "sum"),
        avg_order_value=("revenue", "mean"),
        avg_items_per_order=("items", "mean"),
        first_order=("order_date", "min"),
        last_order=("order_date", "max"),
    )
    features["active_days"] = (features["last_order"] - features["first_order"]).dt.days.clip(lower=0)
    features["recency_days"] = (cutoff - features["last_order"]).dt.days.clip(lower=0)
    repeaters = set(future["customer_id"].unique())
    features["repeat_60d"] = features["customer_id"].isin(repeaters).astype(int)
    return features, orders, cutoff


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out = root / "reports"
    fig_dir = root / "reports" / "ecommerce_figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    features, orders, cutoff = build_customer_table(root / "data" / "ecommerce" / "online_retail.csv")

    train_idx, test_idx = train_test_split(
        np.arange(len(features)), test_size=0.30, random_state=SEED, stratify=features["repeat_60d"]
    )
    model = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=20,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=SEED,
    )
    model.fit(features.iloc[train_idx][FEATURES], features.iloc[train_idx]["repeat_60d"])
    test = features.iloc[test_idx].copy()
    test["repeat_score"] = model.predict_proba(test[FEATURES])[:, 1]
    y = test["repeat_60d"].to_numpy()
    score = test["repeat_score"].to_numpy()
    baseline = float(features["repeat_60d"].mean())
    top_n = max(1, int(len(test) * 0.10))
    top = test.sort_values("repeat_score", ascending=False).head(top_n)
    top_precision = float(top["repeat_60d"].mean())
    metrics = {
        "cutoff": cutoff.isoformat(),
        "history_customers": int(len(features)),
        "future_60d_repeat_rate": baseline,
        "roc_auc": float(roc_auc_score(y, score)),
        "pr_auc": float(average_precision_score(y, score)),
        "top_decile_precision": top_precision,
        "top_decile_lift": float(top_precision / baseline) if baseline else None,
        "top_decile_n": top_n,
        "note": "Features use only orders before the cutoff; the target is any order in the following 60 days.",
    }
    test[["customer_id", "repeat_60d", "repeat_score"] + FEATURES].sort_values("repeat_score", ascending=False).to_csv(out / "repeat_purchase_scores.csv", index=False)
    pd.DataFrame({"feature": FEATURES, "importance": model.feature_importances_}).sort_values("importance", ascending=False).to_csv(out / "repeat_purchase_feature_importance.csv", index=False)
    (out / "repeat_purchase_model_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    deciles = test.assign(decile=pd.qcut(test["repeat_score"].rank(method="first"), 10, labels=False, duplicates="drop") + 1)
    lift = deciles.groupby("decile", as_index=False).agg(predicted_rate=("repeat_score", "mean"), actual_rate=("repeat_60d", "mean"))
    lift["lift_vs_baseline"] = lift["actual_rate"] / baseline
    lift.to_csv(out / "repeat_purchase_decile_lift.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(lift["decile"], lift["lift_vs_baseline"], marker="o", color="#4c78a8")
    ax.axhline(1, color="#555", linewidth=0.8)
    ax.set(title="Repeat-purchase model: lift by score decile", xlabel="Score decile (1 = lowest)", ylabel="Lift vs. average repeat rate")
    fig.tight_layout()
    fig.savefig(fig_dir / "repeat_purchase_lift.png", dpi=180)
    plt.close(fig)

    imp = pd.read_csv(out / "repeat_purchase_feature_importance.csv").sort_values("importance")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh(imp["feature"], imp["importance"], color="#f28e2b")
    ax.set(title="Repeat-purchase model feature importance", xlabel="Random Forest importance")
    fig.tight_layout()
    fig.savefig(fig_dir / "repeat_purchase_feature_importance.png", dpi=180)
    plt.close(fig)

    memo = [
        "# 复购预测模型结果",
        "",
        f"- 特征截止日：{cutoff.date()}；历史客户数：{len(features):,}；未来 60 天复购率：{baseline:.1%}。",
        f"- Random Forest ROC-AUC：{metrics['roc_auc']:.3f}；PR-AUC：{metrics['pr_auc']:.3f}。",
        f"- 预测分数最高的 10% 客户，复购率：{top_precision:.1%}；相对整体提升：{metrics['top_decile_lift']:.2f}x。",
        "- 该模型用于客户运营优先级排序，不把预测结果解释为促销因果效果；如要评估优惠券或推荐位的真实增量，需要继续使用随机 holdout。",
    ]
    (out / "repeat_purchase_model_memo.md").write_text("\n".join(memo) + "\n", encoding="utf-8")
    print("\n".join(memo))


if __name__ == "__main__":
    main()
