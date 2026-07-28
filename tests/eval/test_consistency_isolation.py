import ast
import pathlib

CONSISTENCY = pathlib.Path("evaluator/consistency")


def test_consistency_modules_never_import_gateway():
    files = list(CONSISTENCY.glob("*.py"))
    assert files, "no modules found — is cwd the repo root?"
    for py in files:
        tree = ast.parse(py.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for n in node.names:
                    assert not n.name.startswith("gateway"), f"{py} imports {n.name}"
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("gateway"), \
                    f"{py} imports from {node.module}"


def test_voting_never_reads_the_grading_tests():
    """task.tests is the official grader for humaneval — voting must not use it."""
    for name in ("normalize.py", "vote.py"):
        src = (CONSISTENCY / name).read_text()
        assert ".tests" not in src, f"{name} references task.tests (the grader)"
