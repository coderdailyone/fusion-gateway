import json, glob, os
def cost(m, iid):
    tr = glob.glob(os.path.expanduser(f"~/m4/runs/pilot_{m}/{iid}/*.traj"))
    if not tr: return 0.0
    try:
        info = json.load(open(tr[0])).get("info", {})
        return info.get("model_stats", {}).get("instance_cost", 0.0) or info.get("instance_cost", 0.0) or 0.0
    except Exception:
        return 0.0
ver = json.load(open(os.path.expanduser("~/m4/runs/ds_verifier.json")))
ids = list(ver); kept = [i for i in ids if ver[i]]; esc = [i for i in ids if not ver[i]]
ds_total = sum(cost("deepseek", i) for i in ids)
op_total = sum(cost("opus", i) for i in ids)
op_esc = sum(cost("opus", i) for i in esc)
casc_cost = ds_total + op_esc
def res(pat):
    f = glob.glob(os.path.expanduser("~/m4/" + pat))
    return set(json.load(open(f[0])).get("resolved_ids", [])) if f else set()
ds_r, op_r, ca_r = res("deepseek.ds_grade.json"), res("opus.op_grade.json"), res("cascade.casc_grade.json")
kept_res = len([i for i in kept if i in ds_r])
print(f"union={len(ids)} kept_cheap={len(kept)} escalated={len(esc)}")
print(f"COST deepseek_all=${ds_total:.3f} opus_all=${op_total:.3f} opus_on_escalated=${op_esc:.3f} CASCADE=${casc_cost:.3f}")
for name, r, c in [("deepseek", len(ds_r), ds_total), ("opus", len(op_r), op_total), ("cascade", len(ca_r), casc_cost)]:
    print(f"  {name}: resolved={r} cost=${c:.3f}" + (f" cost/successful=${c/r:.3f}" if r else " (0 resolved)"))
print(f"VERIFIER precision (kept-cheap that deepseek actually resolved) = {kept_res}/{len(kept)}")
print(f"  cascade missed but opus solved (verifier wrongly kept deepseek): {sorted(op_r - ca_r)}")
