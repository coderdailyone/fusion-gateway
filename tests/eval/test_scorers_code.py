from evaluator.suite.types import Task
from evaluator.scorers.code import score, extract_code
from evaluator.sandbox import SandboxResult

def T():
    return Task(id="q", source="livecodebench", problem="echo", answer=None,
                tests=({"stdin": "hi", "expected_stdout": "hi"},), meta={})

def test_extract_python_block():
    assert extract_code("blah\n```python\nprint(1)\n```\nend").strip() == "print(1)"

def test_pass_with_fake_runner():
    def fake(code, stdin="", **kw): return SandboxResult("ok", stdin, "", 0)
    out = "```python\nimport sys;print(sys.stdin.read().strip())\n```"
    assert score(T(), out, runner=fake).correct

def test_fail_on_wrong_output():
    def fake(code, stdin="", **kw): return SandboxResult("ok", "WRONG", "", 0)
    assert not score(T(), "```python\npass\n```", runner=fake).correct

# --- HumanEval-style (pyfunc / exit-code) mode --------------------------

def HE():
    return Task(id="HumanEval/0", source="humaneval", problem="def f(x): ...",
                answer=None,
                tests=({"kind": "pyfunc",
                        "test": "def check(candidate):\n    assert candidate(1) == 2",
                        "entry_point": "f"},),
                meta={})

def test_pyfunc_pass_runs_harness_and_exits_clean():
    seen = {}
    def fake(code, stdin="", **kw):
        seen["script"] = code            # pyfunc calls runner(script) positionally
        return SandboxResult("ok", "", "", 0)   # clean exit = all asserts passed
    assert score(HE(), "```python\ndef f(x): return x + 1\n```", runner=fake).correct
    # the harness must concatenate solution + test + the check() call
    assert "def f(x): return x + 1" in seen["script"]
    assert "check(f)" in seen["script"]

def test_pyfunc_fail_on_assertion_error():
    def fake(code, stdin="", **kw):
        return SandboxResult("error", "", "AssertionError", 1)  # a test assert failed
    assert not score(HE(), "```python\ndef f(x): return x\n```", runner=fake).correct
