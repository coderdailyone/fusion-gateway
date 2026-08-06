#!/usr/bin/env bash
# Pre-pull the retry set's images. Two of the seven failures were `docker build`
# races -- SWE-agent builds a thin layer on top of the base image, and several
# workers racing to pull the same base is what broke it. Pulling serially first
# removes that race entirely; retry3.sh did the same for the baselines.
export XDG_RUNTIME_DIR="/run/user/$(id -u)"
export DOCKER_HOST="unix://${XDG_RUNTIME_DIR}/docker.sock"
export PATH="$HOME/bin:$PATH"
for iid in flexget__flexget-4306 aiogram__aiogram-1670 \
           Azure__azure-sdk-for-python-40487 Textualize__textual-5743 \
           apify__crawlee-python-1155 BerriAI__litellm-10284 Pyomo__pyomo-3588; do
  repo="${iid%%__*}"; rest="${iid#*__}"
  img="starryzhang/sweb.eval.x86_64.$(echo "${repo}_1776_${rest}" | tr 'A-Z' 'a-z'):latest"
  if docker image inspect "$img" >/dev/null 2>&1; then echo "已有 $iid"; continue; fi
  for a in 1 2 3; do
    timeout 500 docker pull "$img" >/dev/null 2>&1 && { echo "拉到 $iid"; break; } \
      || echo "  重试$a $iid"
  done
done
echo PREPULL_DONE
