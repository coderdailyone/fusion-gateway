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
