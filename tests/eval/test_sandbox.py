from evaluator.sandbox import run_code


def test_ok_and_stdout():
    r = run_code("print('hello')")
    assert r.status == "ok" and r.stdout.strip() == "hello"


def test_stdin_is_passed():
    r = run_code("import sys; print(sys.stdin.read().strip())", stdin="ping")
    assert r.status == "ok" and r.stdout.strip() == "ping"


def test_timeout():
    r = run_code("while True: pass", timeout_s=1.0)
    assert r.status == "timeout"


def test_error_on_exception():
    r = run_code("raise ValueError('boom')")
    assert r.status == "error" and "ValueError" in r.stderr


def test_memory_limit():
    r = run_code("x = bytearray(10**9 * 4)", mem_mb=128)  # ~4GB alloc under 128MB cap
    assert r.status in ("error", "killed")
