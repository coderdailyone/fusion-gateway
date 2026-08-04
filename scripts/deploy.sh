#!/usr/bin/env bash
# Deploy the fusion gateway to the production host and (re)start it under systemd.
#
# Prereqs on the operator machine:
#   - SSH access to the host under the alias set in HOST (default: vps).
#     Real connection details live in your local ~/.ssh/config, never in this repo.
# One-time prereqs on the host (NOT done here — see notes):
#   - /opt/fusion-gateway/.env exists, mode 600, with ONE key per provider in
#     configs/gateway.toml (check `api_key_env` there -- this list goes stale
#     whenever a provider is added, and a missing key fails LAZILY: the service
#     starts, /healthz is green, and only the first request routed to that
#     provider raises). As of 2026-07-31 the config needs all three:
#       DEEPSEEK_API_KEY=...
#       GLM_API_KEY=...          # serves both glm-4.5-flash and glm-5.2
#       MOONSHOT_API_KEY=...     # kimi-k3
#       GATEWAY_TOKENS=<principal>:<tok>,admin:<tok>
#       GATEWAY_CONFIG=/opt/fusion-gateway/configs/gateway.toml
#       GATEWAY_DB=/opt/fusion-gateway/data/gateway.sqlite
#   - python3 with venv support (apt-get install -y python3-venv if missing).
#
# NOT done by this script, and needed for a PUBLIC endpoint:
#   - a reverse proxy + TLS. See deploy/nginx.conf.example; the DNS A record
#     must point at this host BEFORE certbot can answer its HTTP challenge.
#     Without it the gateway is reachable only on 127.0.0.1:8800.
#
# Verify after deploying with scripts/smoke.py, not just /healthz -- the lazy
# key failure above is invisible to a health check.
#
# Usage: HOST=vps bash scripts/deploy.sh
set -euo pipefail

HOST="${HOST:-vps}"
DEST="/opt/fusion-gateway"
UNIT="fusion-gateway"

echo "→ syncing source to ${HOST}:${DEST}"
# .gitignore does not bind rsync: runs/ (~1 GB of frozen evaluation samples)
# and evaluator/runs/ are git-ignored but were still being shipped, making the
# payload 1000x larger than the ~1 MB the gateway actually needs. The
# production host has no use for evaluation artifacts. Note '.env' has no
# slash, so it matches at any depth -- that is what keeps runs/secrets/.env
# off the production host.
rsync -az --delete \
  --exclude '.git' --exclude '.venv' --exclude 'internal' \
  --exclude '__pycache__' --exclude '*.sqlite*' --exclude '.env' \
  --exclude '.superpowers' \
  --exclude 'runs' --exclude 'evaluator/runs' \
  --exclude '.pytest_cache' --exclude '*.egg-info' \
  --exclude 'data' \
  ./ "${HOST}:${DEST}/"

echo "→ building venv + installing on ${HOST}"
ssh "${HOST}" bash -s <<'REMOTE'
set -euo pipefail
cd /opt/fusion-gateway
mkdir -p data
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv || { echo "venv failed — apt-get install -y python3-venv"; exit 1; }
fi
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/pip install -q -e .
REMOTE

echo "→ installing systemd unit + restarting"
scp deploy/fusion-gateway.service "${HOST}:/etc/systemd/system/${UNIT}.service"
ssh "${HOST}" "systemctl daemon-reload && systemctl enable --now ${UNIT} && systemctl restart ${UNIT}"

echo "→ health check"
ssh "${HOST}" "sleep 1 && curl -fsS http://127.0.0.1:8800/healthz && echo"
echo "✓ deploy complete"
