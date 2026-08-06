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

echo "→ verifying the NEW code is the code answering"
# A green /healthz proves something is listening, not that it is what you just
# shipped. A gateway process can outlive a deploy -- if the restart silently
# fails, the port stays bound and the old process keeps serving, indefinitely,
# with every health check passing. That is not hypothetical: on 2026-08-05 a
# process served for hours after the code and config beneath it had been
# replaced twice, because the restart's pattern kill never matched its cmdline.
#
# /healthz reports the sha256 of the config AS LOADED. Comparing it against the
# file we just shipped is the difference between "a gateway is up" and "MY
# gateway is up".
want=$(sha256sum configs/gateway.toml | cut -c1-12)
got=$(ssh "${HOST}" "sleep 2 && curl -fsS --max-time 10 http://127.0.0.1:8800/healthz" \
        | sed -n 's/.*"config_sha":"\([^"]*\)".*/\1/p')
if [ -z "${got}" ]; then
  echo "✗ /healthz did not answer, or is too old to report config_sha." >&2
  echo "  An older gateway may still be bound to 8800. On the host:" >&2
  echo "    ss -ltnp | grep :8800     # take the pid from the SOCKET, not a pattern" >&2
  exit 1
fi
if [ "${want}" != "${got}" ]; then
  echo "✗ the gateway is serving a DIFFERENT config than the one just deployed." >&2
  echo "    shipped: ${want}" >&2
  echo "    serving: ${got}" >&2
  echo "  The restart did not take. The old process is still bound to 8800." >&2
  exit 1
fi
echo "  config_sha ${got} — matches what was shipped"

echo "→ smoke (a missing provider key fails LAZILY; /healthz cannot see it)"
echo "  run: GATEWAY_URL=http://127.0.0.1:8800 GATEWAY_TOKEN=<tok> python scripts/smoke.py"
echo "  and, if a fusion panel is configured:"
echo "       python scripts/panel_health.py http://127.0.0.1:8800 <admin-tok>"
echo "✓ deploy complete"
