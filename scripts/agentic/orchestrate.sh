#!/usr/bin/env bash
export XDG_RUNTIME_DIR="/run/user/$(id -u)"; export DOCKER_HOST="unix://${XDG_RUNTIME_DIR}/docker.sock"; export PATH="$HOME/bin:$PATH"
echo "$(date +%T) waiting for prepull..."
while pgrep -f prepull.sh >/dev/null; do sleep 30; done
have=$(docker images --format '{{.Repository}}' | grep -c starryzhang)
echo "$(date +%T) prepull finished, have=$have starryzhang images. Launching pilot (deepseek nw6 + opus nw3)."
bash ~/m4/run_model.sh deepseek/deepseek-chat ~/m4/runs/pilot_deepseek 0.5 ~/m4/inst_pilot.json 6 > ~/m4/pilot_deepseek.log 2>&1 &
DS=$!
bash ~/m4/run_model.sh anthropic/claude-opus-4-8 ~/m4/runs/pilot_opus 2.5 ~/m4/inst_pilot.json 3 https://api.aicodemirror.com/api/claudecode 1.0 > ~/m4/pilot_opus.log 2>&1 &
OP=$!
wait $DS $OP
echo "$(date +%T) PILOT_ALL_DONE"
