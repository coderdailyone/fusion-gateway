#!/usr/bin/env bash
# Grade the fusion arm with the SAME official harness the M4 baselines used.
# Deliberately identical to grade_all.sh's `grade()` -- same dataset, split,
# namespace and worker count -- because a headline compared across arms is only
# meaningful if the grader was byte-identical for each.
export HF_ENDPOINT=https://hf-mirror.com
export XDG_RUNTIME_DIR="/run/user/$(id -u)"
export DOCKER_HOST="unix://${XDG_RUNTIME_DIR}/docker.sock"
export PATH="$HOME/bin:$PATH"
cd ~/m4
rid=fu_grade
rm -f ~/m4/*.$rid.json; rm -rf ~/m4/logs/run_evaluation/$rid
echo "start $(date +%T)"
timeout 3600 ~/m4/.venv/bin/python -m swebench.harness.run_evaluation \
  --dataset_name SWE-bench-Live/SWE-bench-Live --split verified --namespace starryzhang \
  --predictions_path ~/m4/runs/fusion_preds.json --run_id $rid --max_workers 4 \
  > ~/m4/grade_fusion.full.log 2>&1
echo "end $(date +%T) rc=$?"
cat ~/m4/*.$rid.json 2>/dev/null | ~/m4/.venv/bin/python -c "
import json,sys
d=json.load(sys.stdin)
print('resolved', d['resolved_instances'], '/', d['total_instances'],
      '(completed', d['completed_instances'], ')')
print('resolved_ids:', d.get('resolved_ids'))
"
echo GRADE_FUSION_DONE
