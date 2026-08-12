# SWE eval 国内源补丁提示

这个文档记录的是本地 SWE eval 运行环境补丁，不是 `fusion-gateway` 核心服务逻辑。

`fusion-gateway` 负责模型融合网关；SWE-agent / SWE-ReX / Docker build 属于评测环境。国内源补丁应该被当作“运行环境准备步骤”，不要混进 gateway 的请求路由、融合策略或账本逻辑里。

## 背景

在 `/DATA/gaozhi/swe-eval/m4` 跑 SWE-agent 时，SWE-ReX 默认会在任务镜像外再做一次 Docker build，用来安装：

- standalone Python；
- 系统依赖；
- `swe-rex` / `swerex-remote`。

这个 build 默认会访问：

- Debian 默认 apt 源；
- `python.org`；
- 默认 PyPI。

在当前机器上这些下载很慢，之前会卡在 Docker build 阶段。解决方式是修改 SWE-ReX 生成 Dockerfile 的模板，让它走国内源。

## 实际修改位置

当前本地补丁改的是 installed package：

```text
/DATA/gaozhi/swe-eval/m4/.venv-agent/lib/python3.12/site-packages/swerex/deployment/docker.py
```

注意：这不是本仓库文件。它在 SWE-agent 的虚拟环境里。

如果重新运行：

```bash
cd /DATA/gaozhi/swe-eval
bash setup_env.sh
```

这个文件可能会被重建/覆盖，国内源补丁需要重新检查或重打。

## 国内源内容

补丁目标是 `DockerDeployment.glibc_dockerfile` 生成的 Dockerfile 字符串。

需要让生成的 Dockerfile 使用：

```dockerfile
ARG DEBIAN_MIRROR=http://mirrors.aliyun.com/debian
ARG DEBIAN_SECURITY_MIRROR=http://mirrors.aliyun.com/debian-security
ARG PYTHON_TARBALL_URL=https://repo.huaweicloud.com/python/3.11.8/Python-3.11.8.tgz
ARG PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
```

对应行为：

- builder 阶段把 apt 源改为阿里云；
- builder 阶段从华为云下载 `Python-3.11.8.tgz`；
- final 阶段把 apt 源改为阿里云；
- final 阶段用阿里云 PyPI 安装 `swe-rex`。

## 检查补丁是否存在

```bash
rg -n "mirrors.aliyun|repo.huaweicloud|PIP_INDEX_URL|PYTHON_TARBALL_URL" \
  /DATA/gaozhi/swe-eval/m4/.venv-agent/lib/python3.12/site-packages/swerex/deployment/docker.py
```

如果能看到这些字符串，说明 installed SWE-ReX 里已有国内源补丁。

语法检查：

```bash
/DATA/gaozhi/swe-eval/m4/.venv-agent/bin/python -m py_compile \
  /DATA/gaozhi/swe-eval/m4/.venv-agent/lib/python3.12/site-packages/swerex/deployment/docker.py
```

## 当前机器上的记录

本次调试记录放在：

```text
/home/yangjia/fusion/workload/SWEREX_CN_SOURCE_PATCH_RECORD.md
/home/yangjia/fusion/workload/SWEREX_CN_BUILD_RESULT.md
/home/yangjia/fusion/workload/SWE_SMOKE_CN_BUILD_RESULT.md
```

原文件备份：

```text
/home/yangjia/fusion/workload/docker.py.swerex_before_cn_20260812.bak
```

## 回滚方式

如果国内源补丁引入问题，可以恢复备份：

```bash
cp /home/yangjia/fusion/workload/docker.py.swerex_before_cn_20260812.bak \
  /DATA/gaozhi/swe-eval/m4/.venv-agent/lib/python3.12/site-packages/swerex/deployment/docker.py

/DATA/gaozhi/swe-eval/m4/.venv-agent/bin/python -m py_compile \
  /DATA/gaozhi/swe-eval/m4/.venv-agent/lib/python3.12/site-packages/swerex/deployment/docker.py
```

## 已验证的 smoke 路线

已经验证过这条链路可以跑通：

```text
fusion-gateway
→ SWE-agent
→ SWE-ReX
→ rootless Docker
→ task image 二次 build
→ runtime 启动
→ agent 执行任务
→ 调 fusion 模型
→ 写出 preds.json
```

关键结果：

```text
python_standalone_dir='/root'
Runtime started in 1.70s
rc=0
```

本地结果目录：

```text
/home/yangjia/fusion/workload/swe_smoke_cn_build
```

完整日志：

```text
/home/yangjia/fusion/workload/swe_smoke_cn_build.full.log
```

退出状态：

```yaml
instances_by_exit_status:
    submitted (exit_cost):
    - pypsa__pypsa-1012
total_cost: 0.5098305999999999
```

这说明 e2e 环境流程已跑通；退出原因是 smoke 的成本上限，不是环境失败。

## 后续建议

如果需要长期复现，建议再做一个独立脚本：

```text
tools/patch_swerex_cn_sources.sh
```

脚本职责只做三件事：

1. 定位当前 SWE-ReX `docker.py`；
2. 备份原文件；
3. 应用国内源补丁并执行 `py_compile`。

不要把这个补丁写入 gateway runtime 路径；它属于 SWE eval 环境准备。

