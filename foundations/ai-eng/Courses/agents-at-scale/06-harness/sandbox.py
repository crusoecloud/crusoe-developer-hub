import os, subprocess, sys
from pathlib import Path

WORK = Path(os.environ.get("BRIEF_WORKSPACE", "workspace")).resolve()
IMAGE = os.environ.get("SANDBOX_IMAGE", "python:3.12-slim")


def run_python(code: str, timeout_s: int = 30) -> str:
    WORK.mkdir(parents=True, exist_ok=True)
    cmd = ["docker", "run", "--rm",
           "--network", "none",
           "--memory", "512m", "--cpus", "1", "--pids-limit", "128",
           "--read-only", "--tmpfs", "/tmp",
           "--env-file", "/dev/null",
           "-v", f"{WORK}:/work:ro", "-w", "/work",
           IMAGE, "python", "-I", "-c", code]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        return (out.stdout + out.stderr)[-4000:]
    except subprocess.TimeoutExpired:
        return f"error: timed out after {timeout_s}s"


SANDBOX_TOOL = {"type": "function", "function": {
    "name": "run_python",
    "description": "Run Python code in an isolated container with no network and no environment variables. "
                   "The workspace is mounted read-only at /work. Print what you need returned.",
    "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}}

PROOFS = {
    "env": "import os; print(sorted(os.environ))",
    "secrets": "import pathlib; print([p.name for p in pathlib.Path('/work').iterdir()]); print(pathlib.Path('/root/.env').exists())",
    "network": "import urllib.request\ntry:\n    urllib.request.urlopen('https://example.com', timeout=5)\nexcept Exception as e:\n    print('blocked:', type(e).__name__, e)",
    "fork": "import os\nfor _ in range(500):\n    os.fork()",
}

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in PROOFS:
        print(run_python(PROOFS[sys.argv[1]]))
    elif len(sys.argv) > 1:
        print(run_python(" ".join(sys.argv[1:])))
    else:
        for name, code in PROOFS.items():
            print(f"== {name} ==")
            print(run_python(code))
