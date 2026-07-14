# Fusion Gateway — OpenRouter 式工业化融合网关(目标:成本-质量 Pareto SOTA)

> Spec 版本:1.0 · 2026-07-14
> 决策人:master · 执行:Claude(网关核心全新重写)
> 前置研究:`prior internal fusion research`(8 轮商业化研究 + OpenRouter 全景 + DRACO-20 实验)、`internal research reports`(两条研究线现状)

---

## 1. 背景与问题

已有两条研究线,都摸到了天花板:

- **the prior online-routing research line**(cookie 机,活跃):真实 DRACO-20 运行证明动态路由省 24–26% 成本、降 13–14% 延迟,**但质量从未打过池内最优静态策略**。两个阻塞点:①快速裁判方差过大(固定输出重复打分符号一致率仅 0.63–0.74),fusion-gain 标签不可训练;②白盒表征(Qwen-0.5B last-token/256)只是 n=20 pilot。
- **the prior offline control-plane line**(VPS 镜像,已暂停):reviewer 治理的离线控制平面,工程纪律扎实(事件溯源、成本账本、可重放、58 测试),但设计中心是"离线+人工门禁",与 7×24 在线服务方向相反。

本项目把两条线收敛为**一个工业化系统**:自用生产网关 + 可复现基准报告,目标是**成本-质量 Pareto SOTA**。

## 2. 核心决策

### D1 网关核心全新重写,旧线降级为"规范来源 + 评测资产"
master 拍板(2026-07-14 对话)。移植的是**纪律而非代码**:
- 事件溯源与可重放 trace、"STOP 必须携带停止条件"、成本账本语义 —— 来自 the prior offline control-plane line;
- 泄漏纪律(裁判/rubric 绝不进路由输入;候选只见 `id/domain/problem` 类公开字段)、group-by-task 验证、秘密处理规则 —— 来自 the prior online-routing research line 的 AGENTS.md;
- the prior online-routing research line 的 47 个 run 目录与冻结输出保留为评测/回放资产,工作区不动。
- the prior offline control-plane line 保持暂停,永久转为参考资料(其"解冻护栏"由本 spec 的 M0 决策记录满足)。

### D2 SOTA 定义与验收(可证伪)
在固定基准套件上画成本-质量曲线(λ 扫权衡系数),验收三级:
1. **必达**:动态策略(路由/级联/融合)的曲线**整体包络**池内所有静态单模型策略 —— DRACO-20 未达成的事;
2. **必达**:RouteLLM 公开路由器在同套件复跑作对照,APGR(average performance gap recovered,RouteLLM 论文指标)/ 同质量成本节省优于它;
3. **争取**:RouterArena(arXiv:2510.00202)提交或按其协议自测,拿外部锚点。

### D3 部署拓扑
- **网关常驻 VPS**(root@REDACTED-HOST:1156,4 核 8G):OpenAI 兼容 API、路由、账本、观测。
- **cookie 机**(cookie@REDACTED-HOST:6000,RTX 3050 4GB)只做训练、批量评测、重实验 —— 笔记本不承担 7×24 服务。
- 生产特征抽取在 VPS 用 CPU 跑小模型(Qwen-0.5B 级,ONNX/量化),这既是资源约束也是产品要求:路由决策本身必须便宜、快。

### D4 SQLite 是唯一执行真相
沿用 a prior internal runtime 哲学:单文件 SQLite,表:`requests`(请求元数据)、`events`(append-only 事件流,父链接,可重放)、`decisions`(路由决策 + 特征快照 + 策略版本)、`ledger`(每次 provider 调用的成本/延迟/token,预估与实际对账)、`budgets`(里程碑级预算与消耗)。任何真实调用前必须先落 `ledger` 预估行(preflight),完成后 settle。

### D5 评测体系:客观任务为主,裁判须过重复性门槛
- 基准套件以**有标准答案的任务为主**(MMLU-Pro / AIME / LiveCodeBench 子集等,准确率客观),规模 ≥1000 任务,分层抽样、版本锁定;
- 开放题为辅:准则级评测器(rubric 分解 + 位置交换 + 中位数聚合),**固定输出重复打分符号一致率 ≥0.85 才允许其标签进入训练或结论**;
- DRACO 风格深研任务保留为开放题 track;
- 全部质量评测与在线执行严格隔离,评测输入溯源可校验。

### D6 路由器与融合策略
- 特征:公开任务特征 + 小模型 last-token 表征(D3);
- 路由器:预测每模型成功概率与成本,按 `utility = quality − λ·cost` 选动作(单模型 / 级联 / 面板);λ 扫出整条 Pareto 曲线;
- 级联:便宜模型先行,可验证任务(数学/代码)用确定性检查器决定是否升级;
- 融合:学习的分歧门控(替换 0.42 词面阈值)决定是否值得付合成成本;**融合点必须扩展 Pareto 前沿,否则砍掉**;
- 训练与验证一律 group-by-task,禁止 row-wise 泄漏。

### D7 模型池与预算
- 池:DeepSeek、GLM 官方直连为主;Kimi 修复输出健康度后入池;强模型(gpt-5.5 经 REDACTED-MIRROR、或按需直连)作为质量上界与级联终点;按需扩池,新模型走 M6 的自动准入评测。
- 预算:本期 API 实验 $200+(已批准)。每里程碑设硬顶,账本消耗 80% 告警、100% 熔断(kill switch 拒绝新调用,进行中请求放完)。

### D8 治理
- 仓库:`fusion-gateway`(私有),本地 `~/git_projects/fusion-gateway` 开发,分支 + PR;
- Codex MCP(gpt-5.5)对抗互审:spec/里程碑收口/研究结论必须过一次跨模型 review;
- 研究结论三分级:已确认(可复现+过互审)/ 初步(单次实验)/ 证伪(禁止引用),沿用商业化研究文档的置信标注纪律。

## 3. 架构与数据流

```
客户端(an internal service / 研究工作流 / curl)
  │  OpenAI 兼容 /v1/chat/completions(+流式)
  ▼
网关(VPS,常驻)
  ├─ 认证 + 预算 preflight(budgets/ledger)
  ├─ 特征抽取(公开特征 + CPU 小模型表征;超时→降级)
  ├─ 策略引擎(静态规则 → M3 起换训练策略;版本化,可 shadow)
  │    动作:单模型 │ 级联(检查器升级) │ 面板(N 路并行)
  ├─ provider 适配层(DeepSeek/GLM/Kimi/强模型;统一参数规范化、
  │    健康检查、超时、重试、fallback 链)
  ├─ [面板] 分歧门控 → 融合合成(值得才付成本)
  └─ 响应 + events/decisions/ledger 落库(全程可重放)

评测器(独立进程,cookie 机或本地)
  ├─ 基准套件运行器(冻结输出 → 客观打分/准则裁判)
  └─ Pareto 报告生成(曲线、APGR、对照复跑)
     —— 评测产物只读消费网关 trace,绝不反向进入路由输入
```

## 4. 里程碑(每个都有可证伪出口)

| # | 内容 | 出口条件 |
|---|---|---|
| M0 | 仓库/治理/决策记录落地;两条旧线纪律文档移植为 `docs/DISCIPLINES.md` | 仓库建立,本 spec 过 Codex 互审 |
| M1 | 网关最小可用:兼容 API + 适配层(DeepSeek/GLM)+ 静态路由 + 账本/trace/预算护栏/kill switch + systemd 常驻 VPS + an internal service 接入 | 真实流量跑 1 周零事故,账本对账无漂移 |
| M2 | 评测体系:≥1000 任务客观套件 + 开放题裁判重复性验证 + 评测运行器 | 套件锁版;裁判符号一致率 ≥0.85 或明确不启用 |
| M3 | 采样与路由训练:全池×套件 (成败,成本) 矩阵;成本感知路由器;group-by-task CV | 离线曲线包络全部静态策略 |
| M4 | 级联 + 融合:检查器升级级联;学习分歧门控;融合前沿检验 | 融合/级联点扩展 Pareto 前沿(稳定评测下显著) |
| M5 | SOTA 基准报告:全套件复跑 + RouteLLM 对照 + (争取)RouterArena;冻结输出、锁模型版本、可复现 harness | D2 三级验收出结果,报告过互审 |
| M6 | 工业化收尾:策略灰度(shadow → 切流)、新模型自动准入评测、月度报告自动化 | 训练策略承载真实流量 ≥2 周,指标不劣于影子期 |

M1(工程)与 M2(研究)可并行;M3 依赖 M2;M4 依赖 M3;预算大头在 M3–M5。

## 5. 错误处理

- **provider 失败/超时**:按 fallback 链降级,事件记录降级原因;全链失败返回结构化错误,绝不静默换答案。
- **特征抽取超时/宕机**:降级为静态规则路由,decisions 标记 degraded。
- **预算越界**:80% 告警(PushNotification/日志),100% 熔断;熔断状态只允许显式人工解除。
- **账本漂移**:预估 vs 实际 settle 差异超阈值(>20%)告警并进入对账队列 —— 沿用手动对账报告思想,自动化。
- **崩溃恢复**:事件流 append-only,重启后从 events 重建在途状态;进行中请求标记 orphaned 并 settle 为未知成本上界。

## 6. 观测与调试

- 每请求:trace id、决策快照(特征、策略版本、动作)、每 provider 调用的成本/延迟/token、最终动作;
- 日汇总:各策略流量占比、成本、P50/P95 延迟、降级率、熔断事件;
- 重放工具:给定 trace id 离线重建完整决策序列(确定性);
- shadow 模式:新策略只记录"本该怎么路由"不生效,对照报告自动生成。

## 7. 非目标

- 对外多租户服务、对外计费、SLA 承诺;
- 自托管大模型推理(只有 0.5B 级特征模型);
- 模型训练/微调(除路由器与门控的小模型);
- 绝对质量 SOTA(超越前沿旗舰)—— 本期只主张 Pareto SOTA;
- 恢复 the prior offline control-plane line 旧代码库的开发。

## 8. 涉及文件(新仓库骨架)

```
fusion-gateway/
├── gateway/            # 服务核心(API、策略引擎、适配层、账本)
├── evaluator/          # 基准套件运行器与报告生成(独立进程)
├── router_training/    # 特征、采样、训练、CV(重活在 cookie 跑)
├── configs/            # 模型池、价格表(版本化)、预算、策略版本
├── docs/
│   ├── superpowers/specs/   # 本 spec 及后续
│   ├── superpowers/plans/   # writing-plans 产出
│   ├── DISCIPLINES.md       # 从两条旧线移植的纪律
│   └── adr/                 # 关键决策记录(动真相/动钱必须有 ADR)
├── tests/              # 单测 + 重放确定性 + 泄漏 + 预算护栏测试
└── PROGRESS.md
```

## 9. 测试计划

- TDD:每个功能先写失败测试(沿用 an internal project/a prior internal runtime 流程);
- 关键不变量测试:①重放确定性(同一事件流重建出同一决策序列);②泄漏(rubric/答案字段出现在候选 prompt 即 fail);③预算(超顶必熔断,熔断后新调用必拒);④账本(每次真实调用必有 preflight 行,settle 后必对账);
- 测试用注入时钟,禁止 sleep;
- 集成冒烟:真实 provider 各 1 次最小调用(计入账本)验证适配层,其余用录制回放。

## 10. 验收标准(全项目)

1. 网关在 VPS 常驻,承载自有真实流量 ≥2 周,可审计可重放;
2. D2 的两条必达验收在锁定套件上成立,报告可复现(冻结输出 + harness 公开在仓库);
3. 全程 API 支出不超批准预算,账本与实际账单可对账;
4. 关键结论全部过 Codex 跨模型互审,置信分级标注。
