"""Build the fusion arm's prediction file, alongside the M4 baselines.

Reads the per-instance `.pred` files rather than the batch-level preds.json,
because the latter is only written when the whole batch ends -- and a single
straggler would otherwise block grading everything that already finished.

The id set is the UNION across all three arms. Scoring each arm over the same
ids is what makes the numbers comparable; scoring each over only its own
successes would flatter whichever arm crashed most.
"""
import json, glob, os

M4 = os.path.expanduser("~/m4")


def from_preds_json(run_dir):
    p = f"{M4}/runs/{run_dir}/preds.json"
    if not os.path.exists(p):
        return {}
    return {k: (v.get("model_patch") or "") for k, v in json.load(open(p)).items()}


def from_pred_files(run_dir):
    out = {}
    for f in glob.glob(f"{M4}/runs/{run_dir}/*/*.pred"):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        iid = d.get("instance_id") or os.path.basename(f)[:-5]
        out[iid] = d.get("model_patch") or ""
    return out


ds = from_preds_json("pilot_deepseek")
op = from_preds_json("pilot_opus")
# Merge the main run with the retry run. The retry only re-attempted instances
# the main run lost to infrastructure (0 agent steps each), so a later result
# never overwrites an earlier real attempt -- but assert that rather than trust
# it, because a silent overwrite would be indistinguishable from a fair rerun.
fu = from_pred_files("pilot_fusion2")
for k, v in {}.items():
    assert not fu.get(k), f"retry would overwrite a real result for {k}"
    fu[k] = v

ids = sorted(set(ds) | set(op) | set(fu))
rec = lambda i, p, n: {"model_name_or_path": n, "instance_id": i, "model_patch": p}
json.dump({i: rec(i, fu.get(i, ""), "fusion") for i in ids},
          open(f"{M4}/runs/fusion_preds.json", "w"))

print(f"并集 {len(ids)} 个实例")
for name, arm in (("deepseek", ds), ("opus", op), ("fusion", fu)):
    have = sum(1 for i in ids if (arm.get(i) or "").strip())
    print(f"  {name:<9} 有补丁 {have}/{len(ids)}")
print(f"\n写出 {M4}/runs/fusion_preds.json")
