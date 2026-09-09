"""
Stage 4 — Deployer + Self-Healer
Deploys generated code to Hugging Face Spaces.
If the build fails, Qwen3 reads the error logs and patches the code.
Up to MAX_RETRIES self-heal cycles before giving up.
"""

import re
import time
import requests
from openai import OpenAI
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError
from config import (
    CRUSOE_API_KEY,
    CRUSOE_BASE_URL,
    HEALER_MODEL,
    HF_TOKEN,
    HF_USERNAME,
)

MAX_RETRIES = 3
POLL_INTERVAL = 15  # seconds between status checks
BUILD_TIMEOUT = 300  # seconds before we give up waiting

healer_client = OpenAI(api_key=CRUSOE_API_KEY, base_url=CRUSOE_BASE_URL)

HEALER_SYSTEM_PROMPT = """You are an expert Python/Streamlit debugging assistant.
You will receive broken Streamlit app code and a build/runtime error message.
Fix the code so it runs cleanly on Hugging Face Spaces with:
  - Python 3.11
  - streamlit and openai installed
  - CRUSOE_API_KEY and CRUSOE_BASE_URL set as environment secrets

Output ONLY the corrected Python code. No markdown fences, no explanations."""

APP_REQUIREMENTS = "streamlit\nopenai\n"

# HF Spaces Docker spaces must run as uid 1000 (non-root)
DOCKERFILE = """\
FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \\
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user app.py .

EXPOSE 7860
CMD ["streamlit", "run", "app.py", \
"--server.port=7860", \
"--server.address=0.0.0.0", \
"--server.headless=true", \
"--server.enableCORS=false", \
"--server.enableXsrfProtection=false"]
"""

README_TEMPLATE = """---
title: "{title}"
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

{description}

*Powered by [Crusoe Managed Inference](https://crusoe.ai)*
"""


def deploy(code: str, intent: dict, on_progress=None) -> str:
    """
    Deploy code to HF Spaces and return the live URL.
    Calls on_progress(message) at each significant step.
    """
    api = HfApi(token=HF_TOKEN)
    space_name = _slugify(intent.get("title", "crusoe-demo"))
    repo_id = f"{HF_USERNAME}/{space_name}"

    _log(on_progress, f"Creating Space `{repo_id}`...")
    _create_space(api, repo_id)

    safe_title = intent.get("title", "AI Demo").replace('"', "'")
    readme = README_TEMPLATE.format(
        title=safe_title,
        description=intent.get("description", ""),
    )

    for attempt in range(1, MAX_RETRIES + 1):
        _log(on_progress, f"Uploading files (attempt {attempt}/{MAX_RETRIES})...")
        _upload_files(api, repo_id, code, readme)

        _log(on_progress, "Waiting for Hugging Face build...")
        status, error_log = _wait_for_build(api, repo_id, on_progress)

        if status == "RUNNING":
            app_url = _space_url(repo_id)
            _log(on_progress, f"Space is live at {app_url}")
            return app_url

        if attempt < MAX_RETRIES:
            _log(on_progress, f"Build failed. Asking {HEALER_MODEL} to fix the code...")
            code = _heal(code, error_log)
        else:
            raise RuntimeError(
                f"Deployment failed after {MAX_RETRIES} attempts.\nLast error:\n{error_log}"
            )

    raise RuntimeError("Unexpected exit from deploy loop")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _create_space(api: HfApi, repo_id: str) -> None:
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="docker",
            exist_ok=True,
            private=False,
        )
        # Inject Crusoe credentials as Space secrets
        from config import CRUSOE_API_KEY as KEY, CRUSOE_BASE_URL as BASE_URL
        api.add_space_secret(repo_id=repo_id, key="CRUSOE_API_KEY", value=KEY)
        api.add_space_secret(repo_id=repo_id, key="CRUSOE_BASE_URL", value=BASE_URL)
    except HfHubHTTPError as exc:
        raise RuntimeError(f"Failed to create HF Space: {exc}") from exc


def _upload_files(api: HfApi, repo_id: str, code: str, readme: str) -> None:
    files = {
        "app.py": code.encode(),
        "requirements.txt": APP_REQUIREMENTS.encode(),
        "Dockerfile": DOCKERFILE.encode(),
        "README.md": readme.encode(),
    }
    for path_in_repo, content in files.items():
        api.upload_file(
            path_or_fileobj=content,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="space",
        )


def _wait_for_build(
    api: HfApi, repo_id: str, on_progress=None
) -> tuple[str, str]:
    """
    Poll the Space runtime until it reaches a terminal state.
    Returns (stage_string, error_log).
    """
    deadline = time.time() + BUILD_TIMEOUT
    terminal_stages = {"RUNNING", "RUNTIME_ERROR", "BUILD_ERROR", "APP_CRASHED"}

    while time.time() < deadline:
        try:
            runtime = api.get_space_runtime(repo_id=repo_id)
            stage = str(runtime.stage)
            _log(on_progress, f"Build status: {stage}")

            if stage in terminal_stages:
                if stage == "RUNNING":
                    return "RUNNING", ""
                error_log = _fetch_build_logs(repo_id)
                return stage, error_log

        except Exception:
            pass  # transient API error — keep polling

        time.sleep(POLL_INTERVAL)

    return "TIMEOUT", "Build timed out after 5 minutes."


def _fetch_build_logs(repo_id: str) -> str:
    """Fetch build logs from the HF REST API."""
    try:
        owner, name = repo_id.split("/", 1)
        url = f"https://huggingface.co/api/spaces/{owner}/{name}/logs"
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.ok:
            logs = resp.json()
            # logs is a list of {"type": "...", "data": "..."}
            lines = [entry.get("data", "") for entry in logs if entry.get("data")]
            return "\n".join(lines[-100:])  # last 100 lines
    except Exception:
        pass
    return "Unable to retrieve build logs."


def _heal(code: str, error_log: str) -> str:
    """Ask HEALER_MODEL to fix the code given the error log."""
    response = healer_client.chat.completions.create(
        model=HEALER_MODEL,
        messages=[
            {"role": "system", "content": HEALER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Error log:\n```\n{error_log}\n```\n\n"
                    f"Code to fix:\n```python\n{code}\n```"
                ),
            },
        ],
        temperature=0.1,
        max_tokens=4096,
    )
    raw = response.choices[0].message.content or ""
    fenced = re.search(r"```(?:python)?\n(.*?)```", raw, re.DOTALL)
    return fenced.group(1).strip() if fenced else raw.strip()


def _space_url(repo_id: str) -> str:
    owner, name = repo_id.split("/", 1)
    slug = f"{owner}-{name}".replace(".", "-")
    return f"https://{slug}.hf.space"


def _slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")[:40]
    return f"crusoe-{text}" if text else "crusoe-demo"


def _log(callback, message: str) -> None:
    if callback:
        callback(message)
