"""Browser chat with your SGLang service.

    uv run python workshop/notebooks/chat_ui.py        # then open http://127.0.0.1:8080

Port-forwards to your `qwen3` service (in the namespace from your kubeconfig) and serves a
one-page chat with streaming answers, conversation history and a thinking-mode toggle.
Ctrl-C stops it. Ports: WORKSHOP_CHAT_PORT (page, default 8080) and WORKSHOP_LOCAL_PORT + 1
(port-forward, default 8002).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from workshop_utils.chat_ui import main  # noqa: E402

if __name__ == "__main__":
    main(default_name="qwen3", participant=True)
