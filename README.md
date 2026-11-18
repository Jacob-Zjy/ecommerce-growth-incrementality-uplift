# 互联网用户增长增量实验与机器学习分层系统

**Internet Growth Incrementality & Uplift Decision System**

这是一个面向互联网大厂数据科学、数据分析和增长运营岗位的可复现项目。项目回答一个具体的业务问题：**当平台有固定曝光预算时，应该把触达资源投给谁，才能带来更多“额外”的访问和转化？**

项目把问题拆成三层：

1. **是否有效**：用随机实验估计曝光对访问、转化的 ITT（Intent-to-Treat）增量，并检查实验分流是否可信。
2. **谁有效**：用 S-learner 和 T-learner（Random Forest）估计用户层面的 CATE/uplift，区分“被曝光才会行动”的 persuadable 用户与本来就会行动的 sure-things。
3. **投多少人**：用 IPW 评估 5%/10%/20%/30%/50% 人群策略，在曝光成本与增量动作价值的情景假设下输出预算曲线。

## 数据与边界

项目使用 [Criteo Uplift Prediction Dataset](https://ailab.criteo.com/criteo-uplift-prediction-dataset/)，这是公开广告随机实验数据，含匿名用户特征、treatment、exposure、visit 和 conversion。原始 benchmark 规模很大，数据集按 CC BY-NC-SA 4.0 发布；仓库不提交原始压缩包，只通过 Hugging Face datasets server 拉取 3,000 行可复现样本。f0–f11 是匿名特征，不能解释成具体业务标签；项目重点是验证互联网增长场景中的实验评估、异质性建模和预算决策方法。

## 结果摘要（本仓库样本）

运行 `python scripts/run_analysis.py` 后，结果写入 `reports/`：

- 3,000 行跨分片样本，曝光组占比 73.33%；访问率 3.97%，转化率 0.17%。
- 全量曝光对访问的 ITT 为 **+2.00pp**（95% CI **+0.61pp 至 +3.39pp**）。
- T-learner 选取预测增量最高的 10% 用户，测试集 IPW 估计增量访问 **+3.25pp**（约 **32.5 次/1,000 个目标用户**）。在“每次增量访问价值 $5、每次曝光成本 $0.50”的情景假设下，净价值约 **$0.112/合格用户**。
- 转化在这个 3,000 行样本中很稀疏，10% 策略没有足够事件支持稳定估计，因此只作为 guardrail；线上使用前必须扩大样本并保留 holdout。

这些数字是一次可复现的公开样本运行结果，不应被写成平台线上收益。真正上线时需要用平台自己的曝光成本、订单毛利、频控和人群特征重新校准。

## 目录

```text
data/sample/                         可复现的公开样本（不含内部数据）
scripts/fetch_hf_sample.py           从公开数据服务拉取样本
scripts/run_analysis.py              训练模型、估计 uplift、生成结果图
sql/01_funnel_metrics.sql            SQL 漏斗与 ITT 查询
sql/02_policy_segments.sql           预算分层查询模板
prompts/analysis_copilot.md          Codex/Claude 分析工作流提示词
reports/decision_memo.md             决策备忘录
reports/figures/                     uplift、策略价值、特征重要性图
```

## 运行

```powershell
python -m pip install -r requirements.txt
python scripts/fetch_hf_sample.py
python scripts/run_analysis.py
```

如果需要扩大样本，把 `scripts/fetch_hf_sample.py` 中的随机 offset 数量调大，或下载 Criteo 原始压缩包后替换输入文件；分析脚本会自动生成模型对比、策略曲线、特征重要性和决策备忘录。

## 面试可讲的分析链路

- **业务拆解**：北极星指标是增量访问/转化；驱动指标是曝光覆盖、频控、点击/访问率；护栏指标是曝光成本、自然转化被重复触达的比例和实验 holdout 差异。
- **因果推断**：先看随机化与 treatment 比例，再算 ITT 和置信区间；用户分层使用 IPW uplift 曲线，不把相关性写成因果。
- **机器学习**：S-learner 作为基线，T-learner 让 treatment/control 分别拟合，用 Qini/uplift@k 和策略价值比较模型，而不是只看 AUC。
- **落地**：以 10% 小流量策略做线上 holdout，按周重训分层模型；只有当增量动作和净价值都稳定时，才扩大预算。
