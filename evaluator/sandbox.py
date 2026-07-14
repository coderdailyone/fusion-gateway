"""Resource-limited subprocess sandbox for running model-generated Python.

SECURITY BOUNDARY: this module provides resource isolation only — it caps
memory (RLIMIT_AS), CPU time (RLIMIT_CPU), and output file size
(RLIMIT_FSIZE), and enforces a wall-clock timeout via subprocess. It is
NOT namespace/container isolation: there is no chroot, no PID/mount/user
namespace, no seccomp filter, and no filesystem sandboxing beyond running
in a scratch cwd. Network access is NOT blocked — generated code can make
arbitrary outbound connections. Run this only on an isolated machine
(the offline eval "cookie box"), never on the production VPS.
"""

from __future__ import annotations

import os
import resource
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SandboxResult:
    status: str  # "ok" | "timeout" | "error" | "killed"
    stdout: str
    stderr: str
    returncode: int | None


def _make_preexec_fn(mem_mb: int, cpu_s: int):
    def _preexec():
        mem_bytes = mem_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
        fsize_bytes = 10 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_FSIZE, (fsize_bytes, fsize_bytes))
        os.setsid()

    return _preexec


def run_code(
    code: str,
    stdin: str = "",
    timeout_s: float = 5.0,
    mem_mb: int = 256,
    cpu_s: int = 5,
) -> SandboxResult:
    """Run `code` as a standalone Python script under resource limits.

    See module docstring for the security boundary this does (and does not)
    provide.
    """
    with tempfile.TemporaryDirectory() as d:
        script_path = Path(d) / "script.py"
        script_path.write_text(code)

        proc = subprocess.Popen(
            [sys.executable, str(script_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=d,
            env={"PATH": "/usr/bin:/bin"},
            preexec_fn=_make_preexec_fn(mem_mb, cpu_s),
        )

        try:
            stdout_b, stderr_b = proc.communicate(input=stdin.encode(), timeout=timeout_s)
        except subprocess.TimeoutExpired:
            # os.setsid() made the child a new process group leader, so kill
            # the whole group (not just the direct child) to catch anything
            # it may have spawned.
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                pass
            stdout_b, stderr_b = proc.communicate()
            return SandboxResult(
                status="timeout",
                stdout=stdout_b.decode(errors="replace"),
                stderr=stderr_b.decode(errors="replace"),
                returncode=None,
            )

        rc = proc.returncode
        if rc == 0:
            status = "ok"
        elif rc > 0:
            status = "error"
        else:
            status = "killed"

        return SandboxResult(
            status=status,
            stdout=stdout_b.decode(errors="replace"),
            stderr=stderr_b.decode(errors="replace"),
            returncode=rc,
        )
