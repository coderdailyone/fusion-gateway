# Agentic tier execution rig

`evaluator/agentic/` holds the pure, testable half of the agentic tier —
dataset loading, the cascade, the verifier, the budget-gated driver. This
directory holds the other half: the shell that actually drives SWE-agent
inside Docker on a host, which lived only on one machine until now.

`evaluator/agentic/runner.py::run()` is still `NotImplementedError`. These
scripts are what stands in for it. Bringing them here is not an endorsement of
any particular result — it is so the next person does not re-derive the
environment from scratch.

---

## What runs what

| script | role |
|---|---|
| `run_model.sh` | one arm, one model. `MODEL OUTDIR COSTLIMIT INSTFILE NWORKERS [API_BASE] [TEMP]` |
| `run_fusion.sh` | one arm against a gateway pseudo-model over loopback |
| `orchestrate.sh` | runs two arms concurrently |
| `build_cascade.py` | post-hoc: applies the escalation rule, writes per-arm prediction files |
| `build_fusion_preds.py` | same, for a gateway arm; reads per-instance `.pred` so a straggler does not block grading |
| `grade_all.sh`, `grade_fusion.sh` | the official SWE-bench harness. **Identical flags per arm, or the numbers are not comparable** |
| `prepull.sh`, `prepull_retry.sh`, `retry3.sh` | pull instance images serially before a run |
| `select_pilot.py`, `imgname.py`, `compute_pilot.py` | instance selection and reporting helpers |
| `sitecustomize.py` | registers the gateway pseudo-model with LiteLLM (see below) |
| `swe-agent-drop-git-fetch.patch` | see below |

## Host requirements

- Docker (rootless is fine; see `m4_rootless_start.sh` pattern — `DOCKER_HOST=unix://$XDG_RUNTIME_DIR/docker.sock`)
- SWE-agent checked out with its own venv (`install_sweagent3.sh`, `install_fork.sh`)
- The SWE-bench harness venv, separate from SWE-agent's — they pin conflicting deps
- Instance images pulled **before** the run

## Four things that will cost you a day if you don't know them

### 1. A cost limit on an unknown model aborts the run

SWE-agent prices every response with `litellm.completion_cost()`. For a model
LiteLLM has never heard of that **raises**, and `models.py` turns the exception
into a hard `ModelConfigurationError` whenever a cost limit is set — so the run
dies on its first call. Setting the limits to 0 silences it but removes the
only per-instance brake there is.

`sitecustomize.py` registers the pseudo-model instead. Install it into the
**SWE-agent venv's** site-packages so it runs at interpreter startup:

```bash
SP=$(~/m4/.venv-agent/bin/python -c "import site;print(site.getsitepackages()[0])")
cp sitecustomize.py "$SP/sitecustomize.py"
```

The rate in it is deliberately an **upper bound**, not an estimate: a fused
answer's tokens come from several models at different prices, so no single rate
is correct, and one that over-prices makes the brake trip early — the safe
direction. It is not a number to report. The gateway's own ledger prices each
upstream call at that model's real rate.

### 2. `--instances.deployment.docker_args "[]"` is not "no arguments"

An empty list splices a literal empty argument into the command swerex builds:

```
docker run --rm -p 34655:8000 '' --name ... sha256:...
docker: invalid reference format
```

The container dies before swerex can reach it, and the failure surfaces several
layers away as a connection refused on the published port. **Omit the flag
entirely** when you have no docker args.

### 3. The container's network is not the host's network

A container under rootless Docker routes through slirp4netns. `git fetch` from
inside can hang until it dies —

```
fatal: unable to access 'https://github.com/.../': Failed to connect to
github.com port 443 after 76839 ms
```

— while the same fetch from the host completes in 0.11s. This is why a manual
`docker run ... git fetch` can succeed while the identical command under
SWE-agent fails, and why lowering concurrency does not help: it was never
contention.

Two mitigations, in order of preference:

- **`swe-agent-drop-git-fetch.patch`** removes the fetch outright. The base
  commit is already in every SWE-bench-Live image — verify for yourself with
  `docker run --rm --network none <img> bash -lc 'cd /testbed && git cat-file -e <sha>^{commit}'`.
  The commit checked out is byte-identical either way, so this changes the
  harness's reliability, not its verdict.
- A **live** HTTP proxy on the host, injected via `docker_args`, reached at the
  host's LAN address (the container's `127.0.0.1` is its own). `gost -L http://:7890`
  works. Note the original `run_model.sh` injects `192.168.0.103:7890` and
  **nothing was listening** — injecting a dead proxy is worse than none, because
  `git fetch` is the first command in an `&&` chain.

`NO_PROXY` must carry `127.0.0.1` or LiteLLM will try to reach a loopback
gateway through the proxy.

### 4. Two shell habits that silently ruin a run

- **`pkill -f <pattern>` matches your own command line.** It has killed the
  running shell, and worse, has *failed to match* the process it was aimed at
  when that process reaches its binary by a relative path. Take the pid from
  the socket instead: `ss -ltnp | grep :8800`.
- **`cmd | tail` returns tail's exit code, not cmd's.** A run script that ends
  `sweagent run-batch ... | tail -30` throws away the entire beginning of the
  log, which is exactly where a container/startup failure is reported, and
  reports success regardless. `run_fusion.sh` writes the full log to a file and
  tails a copy.

## Before spending money

```bash
# is the gateway serving the code you think it is, and is every panel member alive?
python scripts/panel_health.py http://127.0.0.1:8800 <admin-token>   # exits 1 if not
```

A panel that has lost members still answers 200 and still calls itself a
fusion. Check before the run, not after.
