#!/usr/bin/env bash
# args: MODEL OUTDIR COSTLIMIT INSTFILE NWORKERS [API_BASE] [TEMP]
MODEL="$1"; OUTDIR="$2"; COST="$3"; INST="$4"; NW="${5:-2}"; APIBASE="$6"; TEMP="$7"
export HF_ENDPOINT=https://hf-mirror.com
export XDG_RUNTIME_DIR="/run/user/$(id -u)"; export DOCKER_HOST="unix://${XDG_RUNTIME_DIR}/docker.sock"; export PATH="$HOME/bin:$PATH"
set -a; source ~/m4/secrets.env; set +a
export ANTHROPIC_API_KEY="$CLAUDE_MIRROR_KEY"
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
# opus (has APIBASE=aicodemirror) needs the proxy for its LLM calls; deepseek goes direct
if [ -n "$APIBASE" ]; then
  export HTTPS_PROXY=http://127.0.0.1:7890 HTTP_PROXY=http://127.0.0.1:7890
  export NO_PROXY=localhost,127.0.0.1,hf-mirror.com,api.deepseek.com
fi
P=http://192.168.0.103:7890   # container reaches host proxy at LAN ip (for git fetch)
DARGS="[\"-e\",\"HTTP_PROXY=$P\",\"-e\",\"HTTPS_PROXY=$P\",\"-e\",\"http_proxy=$P\",\"-e\",\"https_proxy=$P\",\"-e\",\"NO_PROXY=localhost,127.0.0.1\"]"
ARGS=(--instances.type file --instances.path "$INST" --agent.model.name "$MODEL"
      --agent.model.per_instance_cost_limit "$COST" --output_dir "$OUTDIR" --num_workers "$NW"
      --agent.model.completion_kwargs "{\"drop_params\": true}"
      --instances.deployment.type docker --instances.deployment.docker_args "$DARGS")
[ -n "$APIBASE" ] && ARGS+=(--agent.model.api_base "$APIBASE")
[ -n "$TEMP" ] && ARGS+=(--agent.model.temperature "$TEMP")
rm -rf "$OUTDIR"
echo "start $(date +%T) model=$MODEL cost=$COST nw=$NW temp=$TEMP proxy=${APIBASE:+on}"
timeout 7200 ~/m4/.venv-agent/bin/sweagent run-batch "${ARGS[@]}" 2>&1 | tail -20
echo "end $(date +%T) rc=$?"
echo "=== per-instance ==="; for t in "$OUTDIR"/*/*.traj; do grep -oE "\"instance_cost\"[: ]+[0-9.]+|\"api_calls\"[: ]+[0-9]+|\"exit_status\"[: ]*\"[a-z_ ]+\"" "$t" 2>/dev/null | tail -3; done
