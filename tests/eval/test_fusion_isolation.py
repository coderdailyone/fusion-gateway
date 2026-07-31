import ast
import pathlib

FUSION = pathlib.Path("evaluator/fusion")


def test_fusion_modules_never_import_gateway():
    # Guard against the test passing vacuously (e.g. cwd != repo root, so the
    # glob matches nothing and the loop body never runs).
    assert list(FUSION.glob("*.py"))
    for py in FUSION.glob("*.py"):
        tree = ast.parse(py.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for n in node.names:
                    assert not n.name.startswith("gateway"), f"{py} imports {n.name}"
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("gateway"), \
                    f"{py} imports from {node.module}"
