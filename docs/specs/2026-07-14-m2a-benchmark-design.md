# M2a — Objective Benchmark Workshop (Design)

> Spec 版本:1.0 · 2026-07-14
> 属于 [DESIGN.md](../DESIGN.md) 的 M2;M2 已拆为 **M2a(本文,客观基准工坊)** 与
> **M2b(裁判稳定化,后续独立循环)**。

## 1. 目标与背景

为 fusion-gateway 建立一套**客观、可复现**的评测基础设施,产出"每 (模型, 任务)
的成败 + 成本"记录——这是后续 M3 训练成本感知路由器、M5 画成本-质量 Pareto
曲线证明 SOTA 的地基。

M2a **不需要裁判**:所有任务都有标准答案或可执行测试,质量轴是客观准确率。
开放题/裁判稳定化(方差大、需 ≥0.85 一致率门槛)是 M2b,不在本文范围。

**关键澄清:** 全量 ≥1000×模型池的付费采样是 **M3**;M2a 只建套件 + 运行器 +
判分器 + 沙箱,并用**小样本验证跑通**。本轮真实 API 花费极小。

## 2. 核心决策

- **D1 套件构成(共 ~1000,分层锁版)**
  - MMLU-Pro ~600(10 选项选择题;`TIGER-Lab/MMLU-Pro`)
  - MATH-500 ~300(数值/符号答案;`HuggingFaceH4/MATH-500`)
  - HumanEval 164(全量;代码生成 + 断言测试;`openai/openai_humaneval`)—— 合计 ~1064 ≥ 1000。
    原定 LiveCodeBench 因 `datasets 5.0`
    删除了脚本式加载器支持(且其测试用例 base64/zlib 编码、stdin 与函数式混杂)而改用
    HumanEval(纯 Parquet、沙箱友好;判分走"跑 solution + test + check(entry_point),退出码 0 为对")
  - 分层依据各数据集自带的 subject/difficulty 标签;锁版 = 钉住 HF 数据集
    revision + 选中 task ID 列表 + 内容 SHA-256。
- **D2 运行器直连 provider,用 LiteLLM。** 评测器是独立离线 harness,经 LiteLLM
  直接调模型(DeepSeek 原生;GLM 走 OpenAI-兼容自定义端点),**不经过在线网关**
  (符合 D5 隔离)。LiteLLM 只在评测器内使用;**网关核心保持自有 httpx 适配层、
  不引入 LiteLLM**。LiteLLM 许可 MIT(落地时复核版本)。
- **D3 套件只存清单 + 下载脚本 + 哈希**,不 vendor 原始数据(规避数据集重发布
  许可风险,仓库轻、可复现)。
- **D4 客观判分**:选择题=抽取最终字母+精确匹配;数学=抽取 boxed/最终答案+
  sympy 符号等价;代码=沙箱内跑隐藏测试用例,全过为对。
- **D5 沙箱非容器级,跑隔离机**:模型生成代码用子进程 + `resource` 限额
  (CPU/内存/文件大小)+ 墙钟超时 + 精简环境 + 临时 cwd + 尽力断网执行。
  **明确安全边界:这不是容器/nsjail 级隔离**;沙箱与验证跑放在隔离的 cookie
  机,绝不在生产 VPS 上跑不受信代码。
- **D6 冻结输出、文件存储、不碰网关真相库**:每 (任务, 模型) 运行产出一条冻结
  记录(prompt、原始输出、解析答案、判分、成本、延迟、token),存
  `evaluator/runs/<suite>_<ts>/*.jsonl` + 结果索引。确定性,可无新 API 调用
  重新判分。评测器**绝不读写网关的 SQLite**。

## 3. 架构与组件(独立 `evaluator/` 模块)

```
evaluator/
├── suite/
│   ├── manifest.py     # 锁版清单:数据源+revision+选中 ID+内容 SHA
│   └── loader.py       # 从 HF 拉取、校验哈希 → Task(id, source, problem,
│                       #   answer_or_tests, meta);答案/测试与 problem 分开持有
├── runner.py           # (task, model) --LiteLLM--> 冻结输出;prompt 只含公开 problem
├── scorers/
│   ├── base.py         # score(task, frozen) -> Score(correct: bool, detail: dict)
│   ├── mcq.py          # 选择题:字母抽取 + 精确匹配
│   ├── math.py         # 数值/符号等价(sympy)
│   └── code.py         # 抽取代码 → sandbox 跑测试
├── sandbox.py          # 受限子进程执行不受信代码
├── store.py            # 冻结输出 JSONL run 目录 + 结果索引;可重新判分
└── report.py           # 样本聚合:每模型准确率 + 平均成本(Pareto 全量在 M3)
```

**数据流:** loader → tasks → runner(LiteLLM→provider)→ 冻结输出 → scorer
(mcq/math/code+sandbox)→ 每 (模型,任务) 结果 → report 聚合。

**边界(D5 隔离):** 答案/测试字段绝不进候选 prompt;评测器有独立文件存储;
在线网关不在评测路径;判分在候选输出冻结后离线进行。

## 4. 错误处理

- **provider 调用失败/超时**:LiteLLM 异常记为该 (任务,模型) 的 `error` 冻结记录
  (非 crash),报告里单列,不计入准确率分母的"已答"但计入"尝试"。
- **哈希不匹配**:loader 拒绝加载并报错(套件被篡改或 HF revision 漂移),绝不
  静默用错数据。
- **沙箱超时/OOM/崩溃**:该测试用例记为 fail(不是评测器崩溃);沙箱本身的
  异常与被测代码的异常严格区分。
- **判分器无法解析输出**(如模型没给出可提取答案):记为 `unparseable`,计为
  incorrect,但在 detail 里留原因供审计。

## 5. 测试计划(TDD)

- 判分器:对/错/无法解析的 fixture 输出,断言 Score。
- 沙箱:安全代码(通过)、超时代码、OOM 代码、抛异常代码、(尽力)尝试联网的
  代码——断言各自被正确限制且分类。
- **泄漏测试**:断言 runner 构造的候选 prompt 中绝不含答案/测试字段(DISCIPLINES
  泄漏纪律的自动化)。
- loader 哈希校验:用一个微型 fixture 数据集,篡改后断言拒绝加载。
- runner:mock 的 LiteLLM(不花钱),断言冻结记录字段完整、prompt 无泄漏。
- 一个**小真实验证跑**(几题 × DeepSeek+GLM,需 key,成本极小)证明端到端。

## 6. 非目标

- 全量 ≥1000×模型池的付费采样(M3)
- 训练路由器(M3)
- 成本-质量 Pareto 曲线报告 + RouteLLM 对照(M3/M5)
- 开放题裁判稳定化 / 准则级评测器(M2b)
- 代码执行以外的更强隔离(容器/nsjail)——留待需要更大规模时

## 7. 验收标准

1. 套件锁版:manifest + loader,哈希校验通过,可复现拉取 ~1000 题。
2. 运行器 + 三个判分器 + 沙箱齐备,单元测试(含泄漏测试、沙箱安全测试)全绿。
3. 小验证跑:几题 × 两个真实模型,端到端产出冻结输出 + 判分 + 成本,无新调用可
   重新判分。
4. 评测器全程不读写网关 SQLite;答案绝不进 prompt(测试保证)。
