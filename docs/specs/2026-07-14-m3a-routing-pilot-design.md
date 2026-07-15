# M3a — Routing-Signal Pilot (Design)

> Spec 版本:1.0 · 2026-07-14
> 属于 [DESIGN.md](../DESIGN.md) 的 M3;M3 已拆为 **M3a(本文,路由信号 pilot)**
> 与 **M3b(全量采样 + 路由器训练 + Pareto,后续独立循环)**。

## 1. 目标与背景

在花大钱做全量采样(1063 题 × 3 模型)之前,用一次**便宜的 pilot(~150 题 × 3
模型,~$1–2)**回答一个先决问题:**这三个模型之间到底有没有"可路由信号"?**
如果它们对同一批题的成败高度一致、成本又差不多,那"路由"没有可利用的信号,
全量采样白花钱。pilot 产出一个**可证伪的 go/no-go 决策** + 复用于全量的
(成败, 成本)矩阵。

复用 M2a 已建好的评测流水线(loader / runner / 三判分器 / sandbox / store)。
M3a 只新增:成本核算修正、采样编排、信号分析。

## 2. 核心决策

- **D1 成本核算用公开 list price 当边际成本代理。** litellm 的
  `completion_cost` 对 Kimi/GLM 的自定义端点返回 $0,不可用于成本轴。新增
  `evaluator/pricing.py`:每模型 USD 价格表(输入/输出每百万 token)+
  `cost(model, in_tokens, out_tokens) -> float`。用各模型**公开 API list price**
  作为边际成本代理(RouteLLM 等路由研究的标准做法)——即使 Kimi/GLM 实际走的是
  coding 订阅计划,list price 仍是比较成本的合理统一口径。DeepSeek $0.14/$0.28
  已确认;GLM-5.2、Kimi-k2 落地时查公开定价并记录来源。
- **D2 Pilot 抽样 = 锁版套件的确定性子集。** 从 `configs/suite.manifest.json`
  的 1063 题里 seeded 分层抽 ~150(按 599/300/164 比例 ≈ 85 mmlu / 42 math /
  23 humaneval)。**这 150 题是全量的子集**,sampler 可续跑,全量(M3b)直接复用
  pilot 的冻结输出,不重花钱。
- **D3 Sampler 可续跑、复用 M2a 组件。** `sample(models, tasks, completion_fns,
  run_dir)`:对每个 (模型, 题),若该 run_dir 的冻结存储里已有 → 跳过(断点重连);
  否则 `run_one` → 用该源的判分器打分 → `pricing.cost` 算成本 → 追加冻结 + 记
  `ResultRow`。`status=error` 如实记录、不计对。
- **D4 go/no-go 双信号(质量头room 或 成本节省,任一显著即 GO)。** 见 §4 指标。
  阈值:**质量头room(oracle − best-static)≥ 5 个百分点** 或 **iso-quality 成本
  节省 ≥ 15%** → GO;两者都接近零 → NO-GO(模型冗余,重议模型池/策略)。
- **D5 算力摆位。** pilot 的代码执行(~23 humaneval × 3 ≈ 69 次)量小、benign,
  **本地资源限额沙箱跑**;全量(M3b)的 492 次代码执行才上隔离的 cookie 机。
- **D6 隔离沿用 M2a。** 评测器仍与网关零耦合;答案/测试绝不进 prompt;判分在
  候选输出冻结后离线进行;评测器不读写网关 SQLite。

## 3. 架构与组件(扩展 `evaluator/`)

```
evaluator/
├── pricing.py    # PRICES 表 + cost(model, in_tok, out_tok)
├── sampler.py    # sample(models, tasks, completion_fns, run_dir) -> list[ResultRow]
│                 #   可续跑;复用 runner.run_one + scorers + store + pricing
└── pilot.py      # 分层抽样 150 子集 + analyze(rows) -> 信号报告 + go/no-go verdict
```

复用(不改):`suite/loader`、`runner`、`scorers/*`、`sandbox`、`store`、
`report.ResultRow`、`hf_fetchers`、`validate.MODELS`(三模型的 completion_fn 工厂)。

**数据流:** manifest → load_suite → 分层 150 子集 → sampler(3 模型,可续跑)→
(成败, 成本)矩阵(冻结 + ResultRow)→ analyze → go/no-go 报告 `docs/M3A_PILOT.md`。

## 4. 信号指标(analyze 的输出)

对每题、每模型有 (correct: bool, cost: float)。计算:

- **每模型**:准确率、均成本、总成本。
- **best-static**:准确率最高的单模型,及其在这批题上的总成本。
- **oracle 准确率**:每题"是否存在至少一个正确模型"的比例(每题选最优)。
  → **质量头room = oracle − best-static 准确率**。
- **分歧率**:模型间对 correctness 不一致的题占比(近零 = 模型冗余)。
- **iso-quality 成本**:一个"每题选最便宜的正确模型、无正确则选最便宜"的 oracle-
  cheap 策略的总成本;→ **成本节省 = 1 − (iso-quality 成本 / best-static 成本)**。
- **verdict**:按 D4 阈值输出 GO / NO-GO + 触发的信号。

> 注:oracle 与 iso-quality 都是**离线事后**上界,只用于判断"信号是否存在",
> **不是**可实现的在线策略(在线路由器是 M3b 要学的)。这条边界写进报告。

## 5. 错误处理

- **provider 调用失败**:`run_one` 返回 `status=error` 的冻结记录;该 (模型,题)
  计为 incorrect、成本按已用 token(通常 0)算;sampler 继续,不中断整批。
- **续跑**:sampler 每题写完即持久化,中途挂掉重跑只补未完成的 (模型,题)。
- **沙箱超时/崩溃**:该 humaneval 题计 incorrect(沿用 M2a 判分器语义)。
- **价格表缺模型**:`pricing.cost` 对未知模型抛错(绝不静默按 0 算,否则成本轴失真)。

## 6. 测试计划(TDD)

- `pricing.cost`:fixture 价格表,验算 in/out token × 单价。未知模型抛错。
- `sampler` 可续跑:fake completion_fn + 临时 run_dir,跑一半模拟中断,重跑断言
  只补未完成项、已完成项不重复调用(用调用计数验证)。
- `analyze`:构造已知 (成败,成本) fixture 矩阵,断言 oracle / best-static /
  分歧率 / iso-quality 成本 / verdict 全部等于手算值(含 GO 和 NO-GO 两种)。
- 真实 pilot 跑(~150×3,需 key,~$1–2)= 验收步骤,非单测。

## 7. 非目标

- 路由器**建模/训练**、λ 扫描、group-by-task CV(M3b)
- 全量 1063 × 模型池采样(M3b,复用本 sampler)
- Pareto 曲线拟合与 RouteLLM 对照(M3b/M5)
- 融合 / 级联(M4)
- 在线可实现策略(pilot 只算离线 oracle 上界判信号)

## 8. 验收标准

1. `pricing.py` / `sampler.py` / `pilot.py` 齐备,单测(含续跑、analyze GO/NO-GO)全绿。
2. 真实 pilot 跑:~150 题 × 3 模型,产出(成败,成本)矩阵 + 信号报告 +
   明确 go/no-go verdict,成本在 ~$1–2、账目清晰。
3. pilot 的 150 题是锁版套件子集,冻结输出可被 M3b 全量复用(可续跑验证)。
4. 评测器全程不碰网关 SQLite;答案不进 prompt。
