"""A minimal browser chat for a deployed endpoint, NVIDIA or AMD.

    uv run python workshop/notebooks/chat_ui.py              # participant: your qwen3 service
    uv run python -m workshop_utils.chat_ui qwen3-amd        # instructor demo
    open http://127.0.0.1:8080

It port-forwards to the endpoint's workload Service (local port WORKSHOP_LOCAL_PORT + 1,
default 8002, so it never collides with the notebook's own port-forward) and serves one HTML page
on WORKSHOP_CHAT_PORT (default 8080). If another program holds either port, the next free one is
used and the printed address says so
that streams answers from the OpenAI-compatible ``/v1/chat/completions`` API.
Stop it with Ctrl-C.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import requests

from .chat import list_models
from .k8s import PORT_SPAN, PortForward, current_namespace, no_free_port, participant_namespace

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>SGLang chat</title>
<style>
  body { margin: 0; font-family: Menlo, 'Roboto Mono', monospace; background: #E4E2E0; color: #000; }
  header { background: #000; color: #fff; padding: 12px 20px; border-left: 10px solid #CEEB13; font-size: 14px; }
  header b { color: #CEEB13; }
  #log { max-width: 900px; margin: 16px auto; padding: 0 16px; }
  .msg { padding: 12px 14px; margin: 8px 0; white-space: pre-wrap; line-height: 1.45; font-size: 14px; }
  .user { background: #C3D6E7; border-left: 4px solid #7799B6; }
  .bot { background: #fff; border-left: 4px solid #395761; }
  .think { color: #666; font-size: 12px; border-left: 4px solid #99AE88; background: #f4f4f2; }
  form { max-width: 900px; margin: 0 auto 24px; padding: 0 16px; display: flex; gap: 8px; }
  input[type=text] { flex: 1; font: inherit; padding: 10px; border: 2px solid #000; background: #fff; }
  button { font: inherit; padding: 10px 18px; background: #000; color: #CEEB13; border: 0; cursor: pointer; }
  label { font-size: 12px; align-self: center; }
</style></head>
<body>
<header>SGLang on <b>__ENDPOINT__</b> &middot; model <b>__MODEL__</b></header>
<div id="log"></div>
<form id="f">
  <input id="q" type="text" placeholder="Ask something" autocomplete="off" autofocus>
  <label><input id="think" type="checkbox"> thinking</label>
  <button>Send</button>
</form>
<script>
const history = [];
const log = document.getElementById("log");
function add(cls, text) { const d = document.createElement("div"); d.className = "msg " + cls; d.textContent = text; log.appendChild(d); window.scrollTo(0, document.body.scrollHeight); return d; }
document.getElementById("f").onsubmit = async (e) => {
  e.preventDefault();
  const q = document.getElementById("q"); const text = q.value.trim(); if (!text) return; q.value = "";
  history.push({role: "user", content: text}); add("user", text);
  const think = add("think", ""); const bot = add("bot", ""); let answer = "", reasoning = "";
  const r = await fetch("/api/chat", {method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({messages: history, thinking: document.getElementById("think").checked})});
  const reader = r.body.getReader(); const dec = new TextDecoder(); let buf = "";
  while (true) {
    const {done, value} = await reader.read(); if (done) break;
    buf += dec.decode(value, {stream: true}); const lines = buf.split("\\n"); buf = lines.pop();
    for (const line of lines) {
      if (!line.startsWith("data: ") || line === "data: [DONE]") continue;
      const delta = (JSON.parse(line.slice(6)).choices || [{}])[0].delta || {};
      if (delta.reasoning_content) { reasoning += delta.reasoning_content; think.textContent = reasoning; }
      if (delta.content) { answer += delta.content; bot.textContent = answer.replace(/^\\s+/, ""); }
      window.scrollTo(0, document.body.scrollHeight);
    }
  }
  if (!reasoning) think.remove();
  history.push({role: "assistant", content: answer.trim()});
};
</script>
</body></html>
"""


def chat_payload(model: str, messages: list[dict], thinking: bool = False, max_tokens: int = 1024) -> dict:
    """The streaming chat-completions request body."""
    return {
        "model": model,
        "messages": messages,
        "stream": True,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": bool(thinking)},
    }


class Handler(BaseHTTPRequestHandler):
    upstream = ""
    model = ""
    endpoint = ""

    def do_GET(self) -> None:
        body = PAGE.replace("__ENDPOINT__", self.endpoint).replace("__MODEL__", self.model).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)) or b"{}"))
        payload = chat_payload(self.model, req.get("messages", []), req.get("thinking", False))
        r = requests.post(f"{self.upstream}/v1/chat/completions", json=payload, stream=True, timeout=600)
        self.send_response(r.status_code)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for line in r.iter_lines():
            if line:
                self.wfile.write(line + b"\n")
                self.wfile.flush()

    def log_message(self, *args) -> None:  # keep the terminal quiet
        pass


def parse_args(argv: list[str] | None = None, default_name: str | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Browser chat for a deployed LLMInferenceService.")
    ap.add_argument(
        "name",
        nargs="?" if default_name else None,
        default=default_name,
        help="LLMInferenceService name, e.g. qwen3 (participants) or qwen3-amd",
    )
    ap.add_argument(
        "--namespace",
        default=None,
        help="defaults to WORKSHOP_NAMESPACE, else the kubeconfig context's namespace, else sglang-workshop",
    )
    ap.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("WORKSHOP_CHAT_PORT", "8080")),
        help="local port for the chat page (default: WORKSHOP_CHAT_PORT or 8080)",
    )
    return ap.parse_args(argv)


def upstream_port() -> int:
    """Local port for the chat UI's port-forward: one above the notebook's (WORKSHOP_LOCAL_PORT)."""
    return int(os.environ.get("WORKSHOP_LOCAL_PORT", "8001")) + 1


def serve_on_free_port(start: int, handler) -> tuple[ThreadingHTTPServer, int]:
    """An HTTP server on ``start`` or, if another program holds it, the next free port after it."""
    for port in range(start, start + PORT_SPAN + 1):
        try:
            return ThreadingHTTPServer(("127.0.0.1", port), handler), port
        except OSError:
            continue
    raise no_free_port(start, "chat page", "WORKSHOP_CHAT_PORT")


def _interrupt(signum: int, frame: object) -> None:
    raise KeyboardInterrupt


def main(default_name: str | None = None, participant: bool = False) -> None:
    """``participant=True`` refuses to fall back to the admin namespace (see participant_namespace)."""
    args = parse_args(default_name=default_name)
    args.namespace = args.namespace or (participant_namespace() if participant else current_namespace())

    notes: list[str] = []  # one output line only: the notebook shows the first line it reads
    with PortForward(args.name, args.namespace, local_port=upstream_port(), log=notes.append) as upstream:
        Handler.upstream, Handler.endpoint = upstream, args.name
        Handler.model = list_models(upstream)[0]
        server, port = serve_on_free_port(args.port, Handler)
        if port != args.port:
            notes.append(f"chat port {args.port} busy, using {port} instead")
        extra = f"  [{'; '.join(notes)}]" if notes else ""
        print(
            f"chat with {args.name} in {args.namespace}: http://127.0.0.1:{port}  (Ctrl-C to stop){extra}",
            flush=True,
        )
        # The notebook stops the UI with terminate() (SIGTERM): handle it like Ctrl-C so the
        # PortForward context exits and its kubectl port-forward does not outlive this process.
        signal.signal(signal.SIGTERM, _interrupt)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()


if __name__ == "__main__":
    main()
