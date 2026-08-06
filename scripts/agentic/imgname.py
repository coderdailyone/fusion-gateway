from datasets import load_dataset
from swebench.harness.test_spec.test_spec import make_test_spec
ds = load_dataset("SWE-bench-Live/SWE-bench-Live", split="verified")
inst = next(x for x in ds if x["instance_id"] == "pylint-dev__pylint-9771")
try:
    spec = make_test_spec(inst, namespace="starryzhang")
except TypeError:
    spec = make_test_spec(inst)
print("KEY", getattr(spec, "instance_image_key", "?"))
