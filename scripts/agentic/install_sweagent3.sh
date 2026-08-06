#!/usr/bin/env bash
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
export HTTP_PROXY=http://127.0.0.1:7890 HTTPS_PROXY=http://127.0.0.1:7890
export NO_PROXY=pypi.tuna.tsinghua.edu.cn,tuna.tsinghua.edu.cn,127.0.0.1,localhost,hf-mirror.com
echo "=== install uv ==="
~/m4/.venv/bin/pip install -q uv 2>&1 | tail -2
echo "=== create py3.12 agent venv (uv fetches standalone python via proxy) ==="
~/m4/.venv/bin/uv venv ~/m4/.venv-agent --python 3.12 2>&1 | tail -5
echo "=== install SWE-agent into agent venv ==="
cd ~/m4/SWE-agent
~/m4/.venv/bin/uv pip install --python ~/m4/.venv-agent/bin/python -e . 2>&1 | tail -8
echo "=== verify ==="
~/m4/.venv-agent/bin/python --version
~/m4/.venv-agent/bin/python -c "import sweagent; print('sweagent', getattr(sweagent,'__version__','?'))" 2>&1 | tail -1
~/m4/.venv-agent/bin/sweagent --help 2>&1 | head -2
echo INSTALL_DONE
