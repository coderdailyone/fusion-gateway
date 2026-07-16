# M3b — Cost-Aware Router + Full Sampling (Design)

> Spec 版本:1.0 · 2026-07-15
> 属于 [DESIGN.md](../DESIGN.md) 的 M3。前置:M3a pilot 已验证 DeepSeek(便宜)+
> Claude Sonnet 5(强)存在可路由信号(oracle 0.927 @ 省 86.5%),见
> [M3A_PILOT.md](../M3A_PILOT.md)。

## 1. 目标与背景

把 M3a 验证的"路由信号存在"落成一个**可上线、可证明**的成本感知路由策略:在
1063 题客观套件上,让**动态策略的成本-质量曲线包络所有静态单模型基线**——这是
"路由 SOTA"的硬证据。

分工(来自 pilot 的落地分析):
- **代码题**运行时可验证(跑测试),走**确定性验证级联**,不用学。
- **数学/选择题**运行时不可验证,走**学习式成本感知路由器**。

## 2. 核心决策

- **D1 先做可学习性检查,gate 住全量花费。** 直接测路由器的核心前提:在**已有
  pilot 数据**(150 题 × DeepSeek+Sonnet,零新增调用)上,简单特征(source / 题长 /
  问题文本 TF-IDF)能否**用 group-by-task CV 显著优于随机地预测每个模型的每题对错**
  (per-model `P(correct|features)` 的 CV AUC)——这正是 train 要建的分类器,直接
  验证"路由可学"。
  - **至少路由相关的模型 AUC ≥ 0.55 → 放行全量采样**;
  - **各模型 AUC ≈ 0.5 → 每题对错不可从特征预测 → 信号 oracle-only、不可学 → 停,
    重议(级联+自一致性),不花那 $12-18。**
- **D2 四模型池。** 便宜腿 DeepSeek;强腿 Claude Sonnet 5、Claude Opus 4.8、
  `gpt-5.6-sol`(mirror,已探活、快)。全量 1063 × 4 ≈ 4252 次调用,~$12-18
  (强模型是大头;master 已批)。
- **D3 全量采样复用 M3a sampler(可续跑)。** 代码题的沙箱执行(164×4≈656 次)
  按纪律**上 cookie 机**跑不受信代码,不在本地/生产 VPS。
- **D4 代码 = 确定性验证级联。** 跑最便宜模型 → 用隐藏测试验证 → 过则留,不过按
  成本阶梯升级(DeepSeek→Sonnet→gpt-5.6-sol→Opus)。无学习、可部署。
- **D5 数学/选择 = 学习式路由器。** 对每个模型训一个"给定问题特征,该模型答对的
  概率"分类器 `P(correct | features, model)`;上线按 `utility = P(correct) −
  λ·cost` 选模型;**λ 扫一遍得整条 Pareto 曲线**。特征起步简单(source/题长/
  TF-IDF),不够再上小模型 embedding(YAGNI)。**全程 group-by-task CV**。
- **D6 验收 = Pareto 包络。** 学习式动态策略曲线要包络所有静态单模型点 + 打赢随机
  路由线;并明确报告 non-oracle 实际效果 vs oracle 上界的差距。
- **D7 隔离沿用。** 评测器与网关零耦合;答案/测试绝不进候选 prompt;判分离线;
  路由器训练/验证只用公开特征,绝不用答案/裁判做特征。

## 3. 架构与组件(扩展 `evaluator/` + 新 `router/`)

```
evaluator/                      # 复用(不改核心)
  sampler.py / pilot.py / pricing.py / scorers / store ...
router/
├── features.py    # extract_features(task) -> 公开特征向量(source/len/TF-IDF)
├── learnability.py# 可学习性检查:CV AUC of 每题赢家可预测性(D1)
├── train.py       # 每模型 P(correct|features) 分类器 + group-by-task CV
├── policy.py      # 路由策略:utility=P−λ·cost;代码走验证级联;λ 扫
├── cascade.py     # 确定性代码验证级联(D4)
└── pareto.py      # 曲线拟合 + 包络判定 + 报告
configs/           # pricing.toml(加 opus/gpt-5.6-sol)、模型池
docs/M3B_REPORT.md # 全量结果 + Pareto 曲线 + 验收结论
```

**数据流:** pilot 数据 → learnability 检查(gate)→[放行]全量 sampler(4 模型)→
(成败,成本)矩阵 → features → train(每模型分类器,CV)→ policy(λ 扫)+ cascade →
pareto(包络判定)→ 报告。

## 4. 错误处理

- **可学习性检查不过**:明确输出"不可学",停在此,不进全量采样(省钱)。
- **采样失败/超时**:sampler 已有 status=error 记录 + 可续跑;某模型某题失败计
  incorrect、成本按已用 token。
- **某模型额度耗尽**(如 Kimi 那样):记录并从池中标记不可用,用可用模型继续,
  报告注明池的实际有效性。
- **分类器退化**(某模型 P(correct) 常数):λ 扫仍有效(退化为静态选择),报告标注。
- **价格缺失**:pricing.cost 抛错(不静默按 0),沿用 M3a。

## 5. 测试计划(TDD)

- `features.extract_features`:fixture task → 稳定特征向量;绝不含答案/测试字段
  (泄漏测试)。
- `learnability`:构造"可预测"与"不可预测"两个合成矩阵,断言 AUC 判定与 gate 结论。
- `train`:合成可分数据 → 分类器 CV AUC 达标;**group-by-task**(同任务不跨折)验证。
- `cascade`:fixture(便宜模型对/错)→ 断言级联选择与成本累加正确。
- `policy`:给定 P 与 cost,断言 λ 不同取值下的选择;λ=0 全便宜、λ→∞ 全最准。
- `pareto`:合成点集 → 断言包络判定(动态包络静态=True/False 两种)。
- 真实全量采样 + 训练 = 验收步骤(需 key + ~$12-18,gated,在 learnability 放行后)。

## 6. 非目标

- RouteLLM 等外部路由器对照(M5)
- 把路由器接进在线网关 / shadow→切流(M6)
- 融合 / synthesis(多路输出合成)(M4)
- 小模型 embedding 特征(仅当简单特征不够时才加,否则 YAGNI)
- 开放题 / 裁判(M2b)

## 7. 验收标准

1. 可学习性检查跑通并给出明确 gate 结论(放行/停)。
2. (放行后)全量 4 模型 × 1063 采样完成,(成败,成本)矩阵齐、账目清、可续跑。
3. 学习式路由器 + 代码级联实现,单测(含泄漏、group-by-task CV、包络判定)全绿。
4. Pareto 报告:动态策略曲线**包络所有静态单模型点**并打赢随机线;明确报告实际
   (non-oracle)效果与 oracle 上界差距。
5. 全程账目 ≤ 批准预算;评测器不碰网关 SQLite;路由特征只用公开信息。
