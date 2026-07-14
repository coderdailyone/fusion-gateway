from evaluator.runner import FrozenOutput
from evaluator.store import new_run_dir, append_frozen, read_frozen


def test_roundtrip(tmp_path):
    rd = new_run_dir(tmp_path, "mmlu_pro", "20260714T000000Z")
    fo = FrozenOutput("q1", "mmlu_pro", "deepseek-chat", "prompt", "out",
                      10, 5, 0.0001, 800, "ok", None)
    append_frozen(rd, fo)
    append_frozen(rd, fo)
    got = read_frozen(rd)
    assert len(got) == 2 and got[0] == fo
