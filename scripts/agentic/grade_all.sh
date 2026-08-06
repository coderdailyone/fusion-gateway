#!/usr/bin/env bash
export HF_ENDPOINT=https://hf-mirror.com
export XDG_RUNTIME_DIR="/run/user/$(id -u)"; export DOCKER_HOST="unix://${XDG_RUNTIME_DIR}/docker.sock"; export PATH="$HOME/bin:$PATH"
cd ~/m4
grade () {
  local preds=$1 rid=$2
  rm -f ~/m4/*.$rid.json; rm -rf ~/m4/logs/run_evaluation/$rid
  timeout 3600 ~/m4/.venv/bin/python -m swebench.harness.run_evaluation \
    --dataset_name SWE-bench-Live/SWE-bench-Live --split verified --namespace starryzhang \
    --predictions_path "$preds" --run_id "$rid" --max_workers 4 >/dev/null 2>&1
  echo "$rid: $(cat ~/m4/*.$rid.json 2>/dev/null | ~/m4/.venv/bin/python -c "import json,sys;d=json.load(sys.stdin);print(f\"resolved {d[\"resolved_instances\"]}/{d[\"total_instances\"]} (completed {d[\"completed_instances\"]})\")" 2>/dev/null)"
}
echo "start $(date +%T)"
grade ~/m4/runs/ds_preds.json ds_grade
grade ~/m4/runs/op_preds.json op_grade
grade ~/m4/runs/cascade_preds.json casc_grade
echo "end $(date +%T)"
echo GRADE_ALL_DONE
