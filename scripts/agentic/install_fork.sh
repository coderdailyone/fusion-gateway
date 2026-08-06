#!/usr/bin/env bash
set -e
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
echo "=== pip install -e fork ==="
~/m4/.venv/bin/pip install -e ~/m4/SWE-bench-Live 2>&1 | tail -8
echo "=== which swebench (应指向 ~/m4/SWE-bench-Live) ==="
~/m4/.venv/bin/python -c "import swebench; print(swebench.__file__)"
echo "=== datasets 版本(确认没被降坏)==="
~/m4/.venv/bin/python -c "import datasets; print('datasets', datasets.__version__)"
echo INSTALL_DONE
