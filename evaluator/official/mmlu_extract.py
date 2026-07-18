"""Official MMLU-Pro answer extraction.

Ported verbatim from TIGER-AI-Lab/MMLU-Pro `evaluate_from_local.py`
(https://github.com/TIGER-AI-Lab/MMLU-Pro , commit
`26d41d02c9f6e1794e74a2740ed378cb0562faea`). Upstream is licensed under the
Apache License 2.0; this port is used under Apache-2.0 with attribution to
TIGER-AI-Lab/MMLU-Pro. The three regexes below (`extract_answer` /
`extract_again` / `extract_final`) are byte-for-byte identical to that pinned
revision.

DEVIATION (deliberate, for determinism): upstream's *extraction* chain already
returns None on total parse failure (identical to us). The divergence is
downstream in upstream's scorer (`save_res`), which, when a prediction is None,
credits a `random.randint`-chosen option and may count it correct. Our frozen
outputs must be reproducibly re-scorable, so we do NOT apply that random-guess
crediting: an unparseable answer stays None and the scorer marks it incorrect.
This is the one documented divergence from the official harness.
"""
from __future__ import annotations

import re


def extract_answer(text: str) -> str | None:
    m = re.search(r"answer is \(?([A-J])\)?", text)
    if m:
        return m.group(1)
    return _extract_again(text)


def _extract_again(text: str) -> str | None:
    m = re.search(r".*[aA]nswer:\s*([A-J])", text)
    if m:
        return m.group(1)
    return _extract_final(text)


def _extract_final(text: str) -> str | None:
    m = re.search(r"\b[A-J]\b(?!.*\b[A-J]\b)", text, re.DOTALL)
    if m:
        return m.group(0)
    return None
