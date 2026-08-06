#!/usr/bin/env bash
export XDG_RUNTIME_DIR="/run/user/$(id -u)"; export DOCKER_HOST="unix://${XDG_RUNTIME_DIR}/docker.sock"; export PATH="$HOME/bin:$PATH"
for img in \
  starryzhang/sweb.eval.x86_64.arviz-devs_1776_arviz-2404:latest \
  starryzhang/sweb.eval.x86_64.apify_1776_crawlee-python-1155:latest \
  starryzhang/sweb.eval.x86_64.pyomo_1776_pyomo-3588:latest; do
  for a in 1 2 3; do timeout 400 docker pull "$img" >/dev/null 2>&1 && { echo "ok $img"; break; } || echo "retry$a $img"; done
done
echo RETRY3_DONE
