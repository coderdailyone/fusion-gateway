import hashlib, json, os, sys
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from datasets import load_dataset
k = int(sys.argv[1]); seed = "m4pilot"
ds = load_dataset("SWE-bench-Live/SWE-bench-Live", split="verified")
rows = list(ds)
by_repo = {}
for r in sorted(rows, key=lambda r: hashlib.sha256((seed+r["instance_id"]).encode()).hexdigest()):
    by_repo.setdefault(r["repo"], []).append(r["instance_id"])
picked, repos = [], sorted(by_repo)
import itertools
for repo in itertools.cycle(repos):
    if len(picked) >= k: break
    if by_repo[repo]: picked.append(by_repo[repo].pop(0))
    if all(not v for v in by_repo.values()): break
print(" ".join(picked[:k]))
