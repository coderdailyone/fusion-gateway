import json, glob, os
def load(m):
    d = {}
    p = os.path.expanduser(f"~/m4/runs/pilot_{m}/preds.json")
    if not os.path.exists(p): return d
    for iid, rec in json.load(open(p)).items():
        tr = glob.glob(os.path.expanduser(f"~/m4/runs/pilot_{m}/{iid}/*.traj"))
        st = ""
        if tr:
            try: st = json.load(open(tr[0])).get("info", {}).get("exit_status", "")
            except Exception: pass
        d[iid] = {"patch": rec.get("model_patch", "") or "", "exit": st}
    return d
ds, op = load("deepseek"), load("opus")
ids = sorted(set(ds) | set(op))
def rec(iid, patch, name): return {"model_name_or_path": name, "instance_id": iid, "model_patch": patch}
casc, esc = {}, 0
for iid in ids:
    d = ds.get(iid, {}); o = op.get(iid, {})
    keep_cheap = d.get("exit") == "submitted" and d.get("patch", "").strip()
    if keep_cheap:
        casc[iid] = rec(iid, d["patch"], "cascade")
    else:
        casc[iid] = rec(iid, o.get("patch", ""), "cascade"); esc += 1
# also write clean single-model preds over the union (empty patch where missing)
json.dump({i: rec(i, ds.get(i, {}).get("patch", ""), "deepseek") for i in ids}, open(os.path.expanduser("~/m4/runs/ds_preds.json"), "w"))
json.dump({i: rec(i, op.get(i, {}).get("patch", ""), "opus") for i in ids}, open(os.path.expanduser("~/m4/runs/op_preds.json"), "w"))
json.dump(casc, open(os.path.expanduser("~/m4/runs/cascade_preds.json"), "w"))
# verifier signal (deepseek submitted) for agreement analysis later
json.dump({i: (ds.get(i, {}).get("exit") == "submitted" and bool(ds.get(i, {}).get("patch", "").strip())) for i in ids}, open(os.path.expanduser("~/m4/runs/ds_verifier.json"), "w"))
print(f"union={len(ids)} cascade_escalated={esc} kept_cheap={len(ids)-esc}")
