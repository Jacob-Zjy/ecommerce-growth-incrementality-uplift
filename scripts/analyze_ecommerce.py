"""Build an order-level e-commerce business view and portfolio figures."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "data" / "ecommerce" / "online_retail.csv"
    out = root / "reports"
    fig_dir = root / "reports" / "ecommerce_figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(source, encoding="latin1")
    df.columns = [c.strip() for c in df.columns]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    df["is_cancelled"] = df["InvoiceNo"].astype(str).str.upper().str.startswith("C")
    df = df.dropna(subset=["CustomerID", "InvoiceDate", "InvoiceNo", "StockCode"])
    df = df[(~df.is_cancelled) & (df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

    orders = df.groupby("InvoiceNo", as_index=False).agg(
        customer_id=("CustomerID", "first"),
        order_date=("InvoiceDate", "first"),
        items=("Quantity", "sum"),
        skus=("StockCode", "nunique"),
        revenue=("Revenue", "sum"),
        country=("Country", "first"),
    )
    orders["month"] = orders["order_date"].dt.to_period("M").astype(str)
    monthly = orders.groupby("month", as_index=False).agg(
        orders=("InvoiceNo", "nunique"),
        customers=("customer_id", "nunique"),
        revenue=("revenue", "sum"),
        avg_order_value=("revenue", "mean"),
    )
    customer = orders.groupby("customer_id", as_index=False).agg(
        orders=("InvoiceNo", "nunique"),
        revenue=("revenue", "sum"),
        avg_order_value=("revenue", "mean"),
        last_order=("order_date", "max"),
        first_order=("order_date", "min"),
    )
    customer["active_days"] = (customer["last_order"] - customer["first_order"]).dt.days
    customer["repeat_customer"] = (customer["orders"] >= 2).astype(int)
    customer.to_csv(out / "ecommerce_customer_metrics.csv", index=False)
    monthly.to_csv(out / "ecommerce_monthly_metrics.csv", index=False)

    top_products = df.groupby(["StockCode", "Description"], as_index=False).agg(
        revenue=("Revenue", "sum"), items=("Quantity", "sum"), orders=("InvoiceNo", "nunique")
    ).sort_values("revenue", ascending=False).head(10)
    top_products.to_csv(out / "ecommerce_top_products.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(monthly["month"], monthly["revenue"], marker="o", color="#2f6f9f")
    ax.set(title="Monthly revenue trend", xlabel="Month", ylabel="Revenue")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(fig_dir / "monthly_revenue.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    counts = customer["orders"].clip(upper=6).value_counts().sort_index()
    counts.index = counts.index.astype(str)
    ax.bar(counts.index, counts.values, color="#e07a5f")
    ax.set(title="Customer repeat-order distribution", xlabel="Orders per customer (6 = 6+)", ylabel="Customers")
    fig.tight_layout()
    fig.savefig(fig_dir / "repeat_order_distribution.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5.2))
    top = top_products.sort_values("revenue")
    labels = top["Description"].fillna(top["StockCode"]).str.slice(0, 28)
    ax.barh(labels, top["revenue"], color="#59a14f")
    ax.set(title="Top products by revenue", xlabel="Revenue")
    fig.tight_layout()
    fig.savefig(fig_dir / "top_products_revenue.png", dpi=180)
    plt.close(fig)

    repeat_rate = customer["repeat_customer"].mean()
    summary = [
        "# 电商订单分析补充结果",
        "",
        f"- 清洗后订单数：{len(orders):,}；客户数：{customer['customer_id'].nunique():,}；商品 SKU 数：{df['StockCode'].nunique():,}。",
        f"- 清洗后 GMV：£{orders['revenue'].sum():,.0f}；客单价：£{orders['revenue'].mean():,.2f}。",
        f"- 复购客户占比（订单数≥2）：{repeat_rate:.1%}。",
        "- 该交易数据用于电商经营分析与机器学习特征构建；它是观察性订单数据，因此复购差异不能直接解释为促销或推荐的因果效果。因果结论继续由随机触达实验模块负责。",
    ]
    (out / "ecommerce_analysis_memo.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
