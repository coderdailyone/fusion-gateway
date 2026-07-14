#!/usr/bin/env bash
# Deploy the fusion gateway to the production host and (re)start it under systemd.
#
# Prereqs on the operator machine:
#   - SSH access to the host under the alias set in HOST (default: vps).
#     Real connection details live in your local ~/.ssh/config, never in this repo.
# One-time prereqs on the host (NOT done here — see notes):
#   - /opt/fusion-gateway/.env exists, mode 600, containing:
#       DEEPSEEK_API_KEY=...
#       GLM_API_KEY=...
#       GATEWAY_TOKENS=prism:<tok>,admin:<tok>
#       GATEWAY_CONFIG=/opt/fusion-gateway/configs/gateway.toml
#       GATEWAY_DB=/opt/fusion-gateway/data/gateway.sqlite
#   - python3 with venv support (apt-get install -y python3-venv if missing).
#
# Usage: HOST=vps bash scripts/deploy.sh
set -euo pipefail

HOST="${HOST:-vps}"
DEST="/opt/fusion-gateway"
UNIT="fusion-gateway"

echo "→ syncing source to ${HOST}:${DEST}"
rsync -az --delete \
  --exclude '.git' --exclude '.venv' --exclude 'internal' \
  --exclude '__pycache__' --exclude '*.sqlite*' --exclude '.env' \
  --exclude '.superpowers' \
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
