import ast
import pathlib

AGENTIC = pathlib.Path("evaluator/agentic")


def test_agentic_modules_never_import_gateway():
    for py in AGENTIC.glob("*.py"):
        tree = ast.parse(py.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for n in node.names:
                    assert not n.name.startswith("gateway"), f"{py} imports {n.name}"
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("gateway"), \
                    f"{py} imports from {node.module}"
