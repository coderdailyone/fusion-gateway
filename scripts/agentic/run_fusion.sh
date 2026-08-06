#!/usr/bin/env bash
# The fusion arm of the M4 pilot. args: OUTDIR COSTLIMIT INSTFILE NWORKERS
# Mirrors run_model.sh, with three deliberate differences:
#   * the model is the gateway's `fusion` pseudo-model, reached over loopback,
#     so there is no relay/TLS overhead between SWE-agent and the panel;
#   * NO_PROXY must cover 127.0.0.1 or litellm would try to reach the local
#     gateway through a proxy that is not running;
#   * cost comes from the gateway ledger afterwards, not from litellm. The
#     per-instance limit here is a brake priced at the panel's HIGHEST member
#     rate (see sitecustomize.py), so it trips early rather than late.
OUTDIR="$1"; COST="$2"; INST="$3"; NW="${4:-3}"
export HF_ENDPOINT=https://hf-mirror.com
export XDG_RUNTIME_DIR="/run/user/$(id -u)"
export DOCKER_HOST="unix://${XDG_RUNTIME_DIR}/docker.sock"
export PATH="$HOME/bin:$PATH"
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
set -a; source ~/fusion-gateway/.env; set +a
export OPENAI_API_KEY="${GATEWAY_TOKENS%%,*}"; OPENAI_API_KEY="${OPENAI_API_KEY#swe:}"
export NO_PROXY=localhost,127.0.0.1,hf-mirror.com
# Containers need a proxy to reach GitHub, and it must be a LIVE one.
#
# The story here cost three runs. run_model.sh injected http://192.168.0.103:7890
# but nothing was listening, so `git fetch` failed and every instance died before
# the agent took a step. Dropping the proxy entirely looked right -- the HOST
# reaches github.com in 0.11s, 5/5 -- but the CONTAINER does not: rootless Docker
# routes through slirp4netns, and from inside a container the same fetch
# intermittently hangs until it dies with
#     fatal: unable to access 'https://github.com/Pyomo/pyomo.git/':
#     Failed to connect to github.com port 443 after 76839 ms
# That is why a manual `docker run ... git fetch` succeeded while the identical
# command under SWE-agent failed, and why dropping concurrency from 6 to 2
# changed nothing: it was never contention.
#
# So the proxy is restored, pointed at a gost listener actually running on this
# host (`gost -L http://:7890`), reached at the LAN address because the
# container's 127.0.0.1 is its own. Verified: 3/3 clean reset chains through it.
#
# NO_PROXY must carry 127.0.0.1, or litellm would try to reach the gateway on
# loopback THROUGH the proxy.
P=http://192.168.0.103:7890
DARGS="[\"-e\",\"HTTP_PROXY=$P\",\"-e\",\"HTTPS_PROXY=$P\",\"-e\",\"http_proxy=$P\",\"-e\",\"https_proxy=$P\",\"-e\",\"NO_PROXY=localhost,127.0.0.1\"]"
ARGS=(--instances.type file --instances.path "$INST"
      --agent.model.name openai/fusion
      --agent.model.api_base http://127.0.0.1:8800/v1
      --agent.model.per_instance_cost_limit "$COST"
      --agent.model.completion_kwargs '{"drop_params": true}'
      --output_dir "$OUTDIR" --num_workers "$NW"
      --instances.deployment.type docker --instances.deployment.docker_args "$DARGS")
rm -rf "$OUTDIR"
echo "start $(date +%T) model=openai/fusion cost=$COST nw=$NW inst=$INST"
# tee, not tail: piping to tail discarded everything before the last 30
# lines, which is exactly where a container/startup failure is reported.
timeout 10800 ~/m4/.venv-agent/bin/sweagent run-batch "${ARGS[@]}" > "$OUTDIR.full.log" 2>&1
rc=$?; tail -25 "$OUTDIR.full.log"
echo "end $(date +%T) rc=$rc  完整日志: $OUTDIR.full.log"
echo "=== per-instance ==="
for t in "$OUTDIR"/*/*.traj; do
  echo "--- $(basename $(dirname $t))"
  grep -oE "\"instance_cost\"[: ]+[0-9.]+|\"api_calls\"[: ]+[0-9]+|\"exit_status\"[: ]*\"[a-z_ ]+\"" "$t" 2>/dev/null | tail -3
done
