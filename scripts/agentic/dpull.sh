#!/usr/bin/env bash
export XDG_RUNTIME_DIR="/run/user/$(id -u)"; export DOCKER_HOST="unix://${XDG_RUNTIME_DIR}/docker.sock"; export PATH="$HOME/bin:$PATH"
IMG=starryzhang/sweb.eval.x86_64.pylint-dev_1776_pylint-9771:latest
echo "start $(date +%T)  img=$IMG"
timeout 3600 docker pull "$IMG"; rc=$?
echo "end $(date +%T) rc=$rc"
docker images "$IMG" --format "PULLED {{.Repository}}:{{.Tag}} {{.Size}}"
