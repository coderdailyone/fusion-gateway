"""gateway/ must not depend on the evaluator or the offline router.

Those packages pull litellm, datasets and scikit-learn. The production venv is
36 MB precisely because they were never installed there, so an import added
here would break the deploy at runtime, not at test time.
"""
import ast
import pathlib

GATEWAY = pathlib.Path("gateway")
BANNED = {"evaluator", "router"}


def _imported_roots(path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.module.split(".")[0]


def test_gateway_never_imports_the_evaluator_or_router():
    offenders = []
    for path in sorted(GATEWAY.rglob("*.py")):
        for root in _imported_roots(path):
            if root in BANNED:
                offenders.append(f"{path}: {root}")
    assert offenders == [], offenders
