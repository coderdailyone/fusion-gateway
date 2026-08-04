# fusion-gateway 交接文档

> 写于 2026-08-04,为**更换服务器**准备。
> 目标读者:接手这个项目的人(或换机后的我自己)。
> 本文档力求可执行:每一节要么给命令,要么给判据。

---

## 0. 一句话

一个自建的、OpenAI 兼容的 LLM 网关,目标是**成本-质量的 Pareto 前沿**。
它有两条腿:

- `gateway/` —— 线上服务。路由、回退、账本、预算闸门、融合面板、工具调用。
- `evaluator/` —— 离线评测。冻结样本 + 官方打分器,用来证明"质量"那一维不是自说自话。

两者**互不导入**(见 §4 边界纪律)。

---

## 1. 当前状态速览

| 项 | 值 |
|---|---|
| 仓库 | `https://github.com/coderdailyone/fusion-gateway` —— **PUBLIC** |
| 主分支 | `master`,头 `a2c1c6f` |
| 测试 | 515 个,`.venv/bin/python -m pytest` |
| 线上端点 | `https://fusion.xinshu.ai/v1`(运行中,已实测) |
| 旧端点 | `gateway.cutecookie.xyz` —— 机器已清空,**DNS 仍指着它,需摘掉** |
| 里程碑 | M1–M9 全部合入 master(PR #2,`2ae2110`,86 个提交) |
| 未合分支 | PR #3 `eval/pool-decorrelation`(1 提交)、PR #4 `fix/deploy-prereqs`(3 提交) |
| 评测机 | `gaozhi-lagos`(VM,`192.168.102.139`)—— 冻结样本与密钥都在这台 |

### 已完成的里程碑

| 里程碑 | 内容 |
|---|---|
| M1 | 网关骨架:路由、回退链、SQLite 账本、预算闸门、killswitch |
| M2a–M2d | 基准套件锁定(1063 题标准档 + 657 题硬档),官方打分器对齐 |
| M3a/M3b | 路由试点与 router 原型 |
| M4 | SWE-bench agentic 档 |
| M5 | **融合面板**离线验证:候选 → 交叉评审 → LLM 融合器,0.8901 |
| M6 | 自一致性投票;确立"客观可验证"条件 |
| M7 | Anthropic wire 适配器(GLM 付费额度只在 Anthropic 端点可达) |
| M8 | 融合面板接入线上网关(**opt-in**,不是默认) |
| M9 | 工具调用穿过融合面板 + 结构化比对 |

---

## 2. 换服务器:迁移清单

**这是本次交接的核心。** 按顺序做。

### 2.1 要搬的东西

| 什么 | 在哪 | 怎么搬 | 注意 |
|---|---|---|---|
| 代码 | GitHub `master` | `git clone` | 别 rsync 旧机器,以 git 为准 |
| 密钥 | `runs/secrets/.env`(评测机)/ `/opt/fusion-gateway/.env`(线上) | **手工重建** | 见 §2.2,**绝不进 git** |
| 网关 DB | `/opt/fusion-gateway/data/gateway.sqlite` | 见 §2.3 | **WAL 模式,不能直接 cp** |
| 冻结样本 | 评测机 `evaluator/runs/`(49 MB) | `rsync -az` | git-ignored,是几百美元的沉没成本,**务必带走** |
| 套件缓存 | 评测机 `runs/cache/` | `rsync -az` | 省去重新拉 HuggingFace |

### 2.2 线上机器的 `.env`

`deploy/fusion-gateway.service` 用 `EnvironmentFile=/opt/fusion-gateway/.env`,
所以这个文件必须存在、mode 600。它需要:

```
DEEPSEEK_API_KEY=...
GLM_API_KEY=...            # 同时服务 glm-4.5-flash 和 glm-5.2
MOONSHOT_API_KEY=...       # kimi-k3
GATEWAY_TOKENS=<principal>:<tok>,admin:<tok>
GATEWAY_CONFIG=/opt/fusion-gateway/configs/gateway.toml
GATEWAY_DB=/opt/fusion-gateway/data/gateway.sqlite
```

> **失败是惰性的。** 少一个 key,服务照样启动、`/healthz` 照样绿,
> 只有第一个路由到该 provider 的请求才会炸。所以部署后**必须**跑
> `scripts/smoke.py`,不能只看健康检查。

### 2.3 搬 SQLite 账本(有坑)

网关的 DB 开着 WAL。**服务运行时直接 `cp` 会拿到不完整的数据** ——
这个坑在 2026-07 踩过一次:copy 出来 59 行/277 事件,实际是 75 行/345 事件。

正确做法:

```bash
systemctl stop fusion-gateway
sqlite3 /opt/fusion-gateway/data/gateway.sqlite "PRAGMA wal_checkpoint(TRUNCATE);"
cp /opt/fusion-gateway/data/gateway.sqlite /tmp/gateway-backup.sqlite
systemctl start fusion-gateway
```

### 2.4 部署

```bash
HOST=<新机器的 ssh 别名> bash scripts/deploy.sh
```

脚本做:rsync 源码 → 建 venv → `pip install -e .` → 装 systemd unit → 重启 → 健康检查。

**脚本不做**、但公网端点必需的:

- 反向代理 + TLS。见 `deploy/nginx.conf.example`。
- **DNS A 记录必须先指向新机器**,certbot 才能应答 HTTP challenge。
- 网关只监听 `127.0.0.1:8800`,没有反代就只有本机能访问。

### 2.5 迁移后验收

```bash
# 1. 健康(不够,但必须过)
curl -fsS http://127.0.0.1:8800/healthz

# 2. 每个模型真实打一发 —— 这才能发现惰性 key 失败
GATEWAY_URL=http://127.0.0.1:8800 GATEWAY_TOKEN=<客户端 token> \
  .venv/bin/python scripts/smoke.py

# 3. 融合路径(注意 max_tokens,见 §8.3)
curl -sS https://<新域名>/v1/chat/completions \
  -H "Authorization: Bearer <token>" -H 'Content-Type: application/json' \
  -d '{"model":"fusion","max_tokens":2048,
       "messages":[{"role":"user","content":"What is 17*23?"}]}' | jq .fusion

# 期望:{"path":"quorum", "degraded":false, "answered_by":"fuser", ...}
```

### 2.6 收尾

- [ ] 摘掉 `gateway.cutecookie.xyz` 的 DNS(指向已清空的旧 VPS)
- [ ] 轮换所有密钥(§10)
- [ ] 确认新机器 `evaluator/runs/` 已到位,`du -sh` 应约 49 MB

---

## 3. 部署拓扑现状(实测)

```
客户端 ──► fusion.xinshu.ai (DNS → 23.144.68.229)
              │  ← 中转/CDN 层,不是网关所在机器的直连 IP
              ▼
           nginx (TLS 终结)
              │
              ▼
           127.0.0.1:8800  uvicorn --factory gateway.app:create_app_from_env
              │
              ├──► api.deepseek.com                  (deepseek-chat → deepseek-v4-flash)
              ├──► open.bigmodel.cn/api/anthropic    (glm-5.2,Anthropic wire)
              ├──► open.bigmodel.cn/api/paas/v4      (glm-4.5-flash,免费)
              └──► api.kimi.com/coding/v1            (kimi-k3 → 上游 id 是 "k3")
```

**已知不一致**:线上那台机器的目录布局是 `~/test/deploy/generated`,
而 `scripts/deploy.sh` 假定 `/opt/fusion-gateway`。而且那个 checkout 里
既没有 `.env` 也没有 `configs/gateway.toml` —— 说明**跑着的进程很可能不是从那个目录起的**。
换机时正好统一到 `/opt/fusion-gateway`。

---

## 4. 架构与边界纪律

**这些是硬约束,改了会让结论失效或让服务出错。**

| 纪律 | 为什么 |
|---|---|
| `evaluator/` 绝不导入 `gateway.*` | 评测必须能独立于线上代码复现 |
| `gateway/` 绝不导入 `evaluator/` 或 `router/` | 线上服务不背评测依赖 |
| `configs/suite.manifest.json` 与 `suite.hard.manifest.json` **字节不可变** | 套件一动,所有历史分数不可比 |
| 投票/验证代码**绝不读** `task.tests` | 否则是拿答案投票,结论循环 |
| 分支 + PR,不直推 master | |

`gateway/fusion_prompts.py` 是从 `evaluator/fusion/prompts.py` **移植**过来的
(剥掉基准脚手架),不是导入 —— 这就是为了守住第一条纪律。
`render_candidate` 用 duck-typing 而非导入 `gateway.*` 类型。

---

## 5. 网关运行时

### 5.1 融合面板怎么工作

```
候选阶段   deepseek-chat ─┐
          glm-5.2       ─┼─► 三个同时发起(t=0 全部启动)
          kimi-k3       ─┘
                │
        ┌───────┴────────┐
        │ quorum 短路?   │  deepseek + glm 互评 "correct" 且答案一致
        └───────┬────────┘
         是 ────┴──── 否
         │             │
    直接采纳       全量路径:交叉评审 → fuser(glm-5.2)综合
    (~15s)          (~73s,等 kimi)
```

`quorum` 短路的依据:M5 自己的多数复制规则 —— 三个候选里多数是 2,
所以两个快成员一致时,慢的那个改变不了结果。

> **这是一次质量换延迟的交易,不是无损优化。** 两个"互评正确"
> 不等于"文本相同";两个不同答案可以互评为正确,此时多数复制规则不触发,
> 走的是 fuser 的"消解异议"分支 —— 而被取消的 kimi-k3 本可能改变结果。
> 配置注释里写清楚了,别把它当成免费的午餐。

### 5.2 工具调用

- 结构化比对:`(name, json.dumps(json.loads(arguments), sort_keys=True, separators=(",",":")))`
- **默认拒绝**分类:`readonly_tools = ["read","ls","grep","find"]`,
  没列出的一律算写类,即使模型一致也保留交叉评审。
  `bash` **故意不在**白名单里 —— 它可以 `ls -la` 也可以 `rm -rf`,名字看不出来。
- 治理不变式:**任何写类调用都不会在缺少"干净交叉评审"或"fuser 裁决"的情况下被发出。**
- 网关会拒绝客户端**从未声明过**的工具调用。

### 5.3 账本与预算

- `preflight` → `settle`/`fail`。`preflight` 是 CONSUMING_STATE,只在启动时清理。
- 被取消的调用以 `usage_source="estimated"` **结算**,绝不记 `fail` —— 上游确实干了活。
- **预算上限当前是关闭的**(`[budgets.M1]` 无 `cap_usd`)。
  唯一的刹车是 `POST /admin/killswitch/trip`(admin token)。
  **没有任何东西会自动帮你停。** 用 `scripts/rollup.py` 盯花费。

### 5.4 端点

| 端点 | 用途 |
|---|---|
| `GET /healthz` | 健康检查,零上游零 DB |
| `GET /v1/models` | 模型列表(含 `fusion` 伪模型) |
| `POST /v1/chat/completions` | 主入口,支持 `stream` |
| `GET /admin/status` | 账本已消耗、预算状态(需 admin token) |
| `POST /admin/killswitch/trip` | **紧急刹车** |
| `POST /admin/killswitch/release` | 解除 |

---

## 6. 评测基建(基准测试)

### 6.1 分层

**采样层(花钱,产出冻结样本)**

| 脚本 | 说明 |
|---|---|
| `scripts/resample_official.py::run_budgeted` | 串行、预算闸门、可续跑、全模型 |
| `scripts/sample_one.py` | 单模型串行 |
| `scripts/sample_one_par.py` | 单模型**并行**,7 小时 → 33 分钟 |

```bash
PYTHONPATH=. .venv/bin/python scripts/sample_one_par.py <model> [n] [ceiling] [run_dir] [workers]
```

并行采样器把网络等待扇出到线程池,但把**两件必须串行的事**加锁串行:
写入(`append_frozen` 以 "a" 模式打开,并发会交错半行)、
花费记账(上限在**派发时**检查,已在飞的调用允许完成并记录 —— 钱已经付了)。

**融合/投票层(花钱)**

```bash
PYTHONPATH=. .venv/bin/python scripts/run_fusion.py oracle        # $0 免费闸门,先看上限
PYTHONPATH=. .venv/bin/python scripts/run_fusion.py run [--limit N]
PYTHONPATH=. .venv/bin/python scripts/run_consistency.py
```

**先跑 `oracle`。** 它 $0,给出这个池的 oracle 天花板 —— 不划算就别付钱跑融合。

**离线分析层($0,不调模型)**

`scripts/pool_oracle.py`(在 `eval/pool-decorrelation` 分支)、
`final_numbers.py`、`fusion_report.py`、`hard_report.py`、`agentic_report.py`、`rollup.py`

### 6.2 两阶段设计(为什么能省钱)

1. **采样(可续跑)**:run 目录里已冻结的 `(model, task)` 对**绝不重调**。
   中断了直接对着同一个 run_dir 再跑一次就续上。
2. **打分(纯离线)**:重读所有冻结输出,打分 + 计价。不调模型,可无限重跑。

> **坑**:"已冻结"包括 `status != "ok"` 的**失败行**。
> 失败行不会被续跑重试。要补,得先剔掉失败行或者开新目录。
> `evaluator/runs/sample_deepseek-v4-pro_20260731T084018Z` 就是这个情况:
> 1063 行但只有 1001 行 ok。

### 6.3 模型注册表

`evaluator/validate.py::MODEL_SPECS` —— 模型名 → litellm 配置。
`make_completion_fn` 的 `**overrides` 直接透传 `api_base`/`api_key`。

> **这意味着"经网关跑基准"不需要写新基建**:往 `MODEL_SPECS` 加一条指向
> `https://fusion.xinshu.ai/v1` 的条目即可。但先读 §8.1,融合响应缺 `usage`,
> 预算闸门会失明。

**推理模型必须给大 max_tokens**(kimi/GLM 用 8192)。给小了,额度全被隐藏推理吃掉,
返回**空内容**。M3a 试点里 kimi 在 2048 下产出过 86 个空答案。

---

## 7. 冻结数据清单

评测机 `evaluator/runs/`,49 MB。**对 1063 题标准档的 `status=ok` 覆盖率:**

| 模型 | 覆盖 | 状态 |
|---|---:|---|
| deepseek-chat(旧 flash) | 1063/1063 | 完整 |
| glm-5.2 | 1063/1063 | 完整 |
| claude-sonnet-5 | 1063/1063 | 完整 |
| gpt-5.5 | 1063/1063 | 完整 |
| claude-opus-4-8 | 1062 | 缺 1 |
| gpt-5.6-sol | 1058 | 缺 5 |
| deepseek v4-flash 新版 | 1048 | 缺 15 |
| kimi-k3(两处目录合并) | 1033 | 缺 30 |
| deepseek v4-pro | 1001 | 缺 62(见 §6.2 失败行坑) |
| kimi-k2 | 159 | 缺 904,当年额度断了 |

目录:`m2c_full`(标准档全量)、`m2d_hard`(硬档)、`m5_fusion`(含 kimi-k3 分片与融合产物)、
`m6_consistency`、`sample_deepseek-*`(2026-07-31 采的 DeepSeek 新版)。

kimi-k3 分散在 `m2c_full/kimi-k3*` 与 `m5_fusion/kimi-k3*` 两处,
`run_fusion.py` 的 `PANEL_DIRS` 按 first-ok-wins 合并。

---

## 8. 今天(2026-08-04)的实测发现

### 8.1 【缺陷】融合响应不返回 `usage`

实测(经 `fusion.xinshu.ai`):

```
顶层键: ['id','object','created','model','choices','fusion']
有 usage: False
```

单模型路径**有** `usage`,融合路径**没有**。根因:`gateway/fusion.py::openai_response`
构造的字典里就没有 `usage` 键,而 `app.py` 的融合分支直接返回它。

**两个后果:**

1. **评测的预算闸门会失明。** `make_completion_fn` 读 `resp.usage.prompt_tokens`,
   拿到 0 → 花费恒为 0 → 上限永不触发。而网关的 `cap_usd` 也是关的。
   **在修掉之前,不要经网关跑大规模付费基准。**
2. 任何按 token 计费的 OpenAI 客户端,对融合请求的成本完全不可见。

钱本身没丢 —— 网关账本逐调用记了 `in_tokens/out_tokens/latency_ms/cost`,
只是**没暴露给客户端**。

### 8.2 延迟归因

从干净机房节点(旧 VPS)测:

| 端点 | TLS 握手 | 首字节 | 总计 |
|---|---:|---:|---:|
| `fusion.xinshu.ai/healthz` ×3 | 0.46–0.49s | 0.66–0.71s | 0.66–0.71s |
| `fusion.xinshu.ai/v1/models` | 0.43s | 0.59s | 0.59s |
| 对照 `api.deepseek.com` | **0.07s** | — | 0.22s |
| 对照 `baidu.com` | **0.08s** | — | **0.09s** |

`/healthz` 零上游零 DB 零计算,却要 0.66s,其中 0.47s 花在 TLS 握手上 ——
**那时请求还没到网关**。握手后到首字节约 0.2s,那才是网关+反代的真实开销,正常。

结论:**慢在 `fusion.xinshu.ai` 前面那层**(DNS 解析到 `23.144.68.229`,
不是机器直连 IP,特征是中转/CDN/跨境回源),不是网关进程。
从本地测同一端点 TLS 2.2–6.5s、首字节 3.0s —— 约 0.5s 是中转层固定开销(所有人都付),
剩下 2–6s 是本地到中转层的链路,抖动极大。

**换服务器时这是个机会**:如果新机器有可直连的公网 IP,把 DNS 直接指过去,
能省掉大部分固定开销。**决定性证据仍缺**:网关本机回环
(`curl http://127.0.0.1:8800/healthz`)的数字,预期毫秒级。

工具:`scripts/diagnose_latency.py`(在 `fix/deploy-prereqs` 分支),
三个距离 × 四个深度,**要在网关本机上跑**:

```bash
python3 scripts/diagnose_latency.py <token> [domain] [local_base]
```

### 8.3 【方法论坑】推理模型 + 小 max_tokens = 假的"降级"

用 `max_tokens=16` 打融合,得到:

```json
{"path":"full","panel":["glm-5.2"],"degraded":true,"answered_by":"candidate"}
```

看起来像三缺二、密钥坏了。**不是。** deepseek-chat 和 kimi-k3 返回了
**空内容**(`reasoning_tokens: 16`,额度全被隐藏推理吃光),
而 `Candidate.__bool__` 把空文本当假 → 被当作失败候选丢弃 → 面板塌成一员。

`max_tokens=2048` 重测,一切正常:

```json
{"path":"quorum","panel":["deepseek-chat","glm-5.2"],"fuser":"glm-5.2",
 "degraded":false,"answered_by":"fuser"}
```

**验收融合时永远用 ≥2048 的 max_tokens。** 单模型实测:
deepseek-chat 1.6s / glm-5.2 2.3s / kimi-k3 2.7s / glm-4.5-flash 1.7s;
融合 quorum 路径 5.7–6.1s。

---

## 9. 未决事项登记册

| # | 事项 | 状态 |
|---|---|---|
| 1 | **融合响应缺 `usage`** | 新发现,未修。见 §8.1 |
| 2 | 写类工具调用的交叉评审是**自评**,不是独立评审 | 用户已知,暂缓 |
| 3 | `glm-5.2` / `kimi-k3` 的**定价未经核实** | `configs/pricing.toml` 里标了 VERIFY。它们决定计费 |
| 4 | 工具调用 / agent 能力**没有打分器** | 所以 DeepSeek 声称增强的 Agent 能力我们量不出来 |
| 5 | `gateway.cutecookie.xyz` DNS 仍指着已清空的旧 VPS | 待摘 |
| 6 | PR #3、PR #4 未合并 | |
| 7 | 网关预算上限**关闭** | 有意为之,但唯一刹车是手动 killswitch |
| 8 | 线上目录布局与 `deploy.sh` 假设不一致 | 换机时统一 |
| 9 | 经网关的端到端基准**从未做过** | 所有历史分数都是直连上游量的。融合路径的线上质量未知。**前置依赖是 #1** |

### 已量出来的结论(别重复劳动)

- **池去相关**($0,复用 M2c 冻结样本):国产池 oracle 0.9036 → 加 claude-opus 0.9442
  (+4.1pt,吃掉 95 题盲区的 42.1%);加 v4-pro 是 +2.5pt / 26.3%。两者盲区不重叠。
- **DeepSeek 新版 flash**:与旧版无显著差异(p=0.60)。v4-pro 反而比新 flash 高 2.0pt
  (p=0.065),几乎全在 HumanEval(+7.8pt)。这**不是**"厂商吹牛"的证据,
  而是**我们的套件看不见他们声称改进的 Agent 能力**(见 #4)。
- **融合 vs 最佳单成员**:M5 上 +1.1pt,**p=0.176,不显著**,且是基准题不是聊天。
  写宣传语时别把它说成已证实的胜利。

---

## 10. 安全:密钥轮换清单

**所有下列密钥都应视为已泄露,需要轮换。** 它们在会话中被明文粘贴过,
而聊天记录是永久保留的(`cleanupPeriodDays=36500`)。

- `DEEPSEEK_API_KEY`
- `GLM_API_KEY`
- `MOONSHOT_API_KEY`
- `OPENAI_MIRROR_KEY`
- `CLAUDE_MIRROR_KEY`
- `ANTHROPIC_API_KEY`
- `HF_TOKEN`
- 网关客户端 token 与 admin token(`fgw_…` 以及 `fusion.xinshu.ai` 的那个)

**纪律:** 密钥只存在于 `runs/secrets/.env`(mode 600,git-ignored)与线上机器的
`/opt/fusion-gateway/.env`。**绝不完整打印、绝不提交。**
`scripts/deploy.sh` 的 rsync 排除项里 `.env` **没有前导斜杠** ——
这样它在任意深度都匹配,正是这一点挡住了 `runs/secrets/.env` 被推上线上机器。

---

## 11. 文件地图

```
gateway/              线上服务
  app.py              FastAPI 应用;路由、流式、融合编排、admin 端点
  fusion.py           面板编排:候选/评审/融合、quorum 短路、工具决策树
  fusion_prompts.py   从 evaluator 移植的提示词(移植,不是导入)
  tool_vote.py        工具调用的纯比对核心
  providers.py        OpenAI wire 适配器
  providers_anthropic.py / anthropic_translate.py   Anthropic wire
  ledger.py / db.py / events.py                     账本与事件
  config.py / policy.py                             配置与路由策略

evaluator/            离线评测(绝不导入 gateway)
  suite/              套件加载与 manifest(字节不可变)
  official/           官方打分管线(HumanEval/MATH/MMLU/LiveCodeBench)
  fusion/             M5 融合研究
  consistency/        M6 自一致性投票
  agentic/            M4 SWE-bench 档
  sampler.py store.py runner.py validate.py pricing.py

scripts/              驱动与运维
configs/              gateway.toml / pricing.toml / suite*.manifest.json
deploy/               systemd unit + nginx 示例
docs/                 报告、ADR、spec/plan
  HANDOVER.md         ← 本文档
  tmp/                临时:会话转录(已脱敏)
runs/secrets/.env     密钥(mode 600,git-ignored)
evaluator/runs/       冻结样本 49 MB(git-ignored,**换机必须带走**)
```

---

## 12. 换机后第一天该做什么

按优先级:

1. **验收部署**(§2.5)—— 特别是 `smoke.py`,别只看 `/healthz`
2. **摘掉旧 DNS**、轮换密钥(§10)
3. **修 §8.1**(融合响应加 `usage`)—— 这是解锁 #9 端到端基准的前置条件
4. 在新机器上跑 `diagnose_latency.py`,拿到回环数字,确认中转层开销是否还在
5. 合并 PR #3、#4
