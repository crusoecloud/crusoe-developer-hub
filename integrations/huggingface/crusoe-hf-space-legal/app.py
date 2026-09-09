"""
Crusoe Foundry — Infinite Context Demo
HuggingFace Space showcasing MemoryAlloy™ & KV Cache sharing
"""

import os
import time
import tiktoken
import gradio as gr
from openai import OpenAI

# ── Crusoe Foundry client ─────────────────────────────────────────────────────
CRUSOE_API_KEY = os.environ.get("CRUSOE_API_KEY", "YOUR_API_KEY_HERE")
CRUSOE_BASE_URL = os.environ.get("CRUSOE_BASE_URL", "https://managed-inference-api-proxy.crusoecloud.com/v1/")
AVAILABLE_MODELS = [
    "Qwen/Qwen3-235B-A22B-Instruct-2507",
    "deepseek-ai/DeepSeek-R1-0528",
    "moonshotai/Kimi-K2-Thinking",
    "deepseek-ai/DeepSeek-V3-0324",
    "meta-llama/Llama-3.3-70B-Instruct",
    "openai/gpt-oss-120b",
    "google/gemma-3-12b-it",
]
MODEL = os.environ.get("CRUSOE_MODEL", AVAILABLE_MODELS[0])

client = OpenAI(api_key=CRUSOE_API_KEY, base_url=CRUSOE_BASE_URL)

# ── Token counting ────────────────────────────────────────────────────────────
try:
    enc = tiktoken.encoding_for_model("gpt-4")
except Exception:
    enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(enc.encode(text))


def format_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)


# ── Document ingestion helpers ────────────────────────────────────────────────
def read_uploaded_file(file_path: str) -> str:
    """Read text from uploaded file (txt, md, py, or pdf via pdfminer)."""
    if file_path is None:
        return ""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        try:
            from pdfminer.high_level import extract_text
            return extract_text(file_path)
        except Exception as e:
            return f"[PDF extraction error: {e}]"
    else:
        with open(file_path, "r", errors="replace") as f:
            return f.read()


# ── KV-cache simulation state ─────────────────────────────────────────────────
_cache_store: dict[str, dict] = {}


def get_cache_key(context: str) -> str:
    import hashlib
    return hashlib.md5(context.encode()).hexdigest()


# ── Shared chat logic ─────────────────────────────────────────────────────────
def stream_response(system_prompt: str, history: list, user_msg: str, model: str = None):
    """
    Streams a response from Crusoe Foundry.
    Returns (updated_history, token_info_str, latency_str, error_str)
    history is a list of {"role": "user"|"assistant", "content": str} dicts (Gradio 6.x format).
    """
    model = model or MODEL
    messages = [{"role": "system", "content": system_prompt}]
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_msg})

    total_ctx_tokens = sum(count_tokens(m["content"]) for m in messages)

    new_history = history + [{"role": "user", "content": user_msg}]

    t0 = time.perf_counter()
    reply = ""
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            max_tokens=2048,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            reply += delta
            yield (
                new_history + [{"role": "assistant", "content": reply}],
                f"📄 **{format_tokens(total_ctx_tokens)} tokens** in context",
                f"⏱ {time.perf_counter() - t0:.2f}s",
                "",
            )
    except Exception as e:
        reply = f"❌ API error: {e}"
        yield (
            new_history + [{"role": "assistant", "content": reply}],
            f"📄 {format_tokens(total_ctx_tokens)} tokens in context",
            "—",
            str(e),
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — LEGAL  (document Q&A)
# ─────────────────────────────────────────────────────────────────────────────
legal_doc_store = {"text": "", "tokens": 0}


def legal_ingest(files):
    if not files:
        return "No files uploaded.", "0 tokens", gr.update()
    combined = ""
    for f in files:
        combined += f"\n\n--- {os.path.basename(f)} ---\n\n"
        combined += read_uploaded_file(f)
    legal_doc_store["text"] = combined
    legal_doc_store["tokens"] = count_tokens(combined)
    tok_str = format_tokens(legal_doc_store["tokens"])
    preview = combined[:800] + ("…" if len(combined) > 800 else "")
    return (
        f"✅ Loaded {len(files)} document(s) — **{tok_str} tokens** ingested into context.",
        f"📄 {tok_str} tokens",
        gr.update(value=preview),
    )


def legal_chat(user_msg, history, model):
    if not user_msg.strip():
        yield history, "—", "—", ""
        return
    doc_context = legal_doc_store["text"]
    system = (
        "You are an expert analyst with access to the full text of the uploaded documents. "
        "Answer questions precisely, citing relevant sections when possible. "
        "If a question cannot be answered from the document, say so clearly.\n\n"
        f"=== DOCUMENT CONTEXT ===\n{doc_context}\n=== END CONTEXT ==="
        if doc_context
        else "You are a helpful document analyst. No documents have been loaded yet."
    )
    yield from stream_response(system, history, user_msg, model)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — DEV  (codebase Q&A)
# ─────────────────────────────────────────────────────────────────────────────
dev_code_store = {"text": "", "tokens": 0}


def dev_ingest(files, raw_paste):
    combined = raw_paste or ""
    for f in (files or []):
        combined += f"\n\n# === {os.path.basename(f)} ===\n\n"
        combined += read_uploaded_file(f)
    dev_code_store["text"] = combined
    dev_code_store["tokens"] = count_tokens(combined)
    tok_str = format_tokens(dev_code_store["tokens"])
    preview = combined[:800] + ("…" if len(combined) > 800 else "")
    return (
        f"✅ Codebase loaded — **{tok_str} tokens** in context.",
        f"📄 {tok_str} tokens",
        gr.update(value=preview),
    )


def dev_chat(user_msg, history, model):
    if not user_msg.strip():
        yield history, "—", "—", ""
        return
    code_context = dev_code_store["text"]
    system = (
        "You are a senior software engineer with full visibility into the provided codebase. "
        "Answer questions about architecture, bugs, refactoring, and code quality. "
        "Reference specific file names, function names, and line context when relevant.\n\n"
        f"=== CODEBASE ===\n{code_context}\n=== END CODEBASE ==="
        if code_context
        else "You are a helpful coding assistant. No code has been loaded yet."
    )
    yield from stream_response(system, history, user_msg, model)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — MEMORY DEMO  (KV-cache visibility)
# ─────────────────────────────────────────────────────────────────────────────
memory_state = {
    "cached_context": "",
    "cached_tokens": 0,
    "query_count": 0,
    "total_saved_tokens": 0,
}


def memory_set_context(context_text):
    memory_state["cached_context"] = context_text
    memory_state["cached_tokens"] = count_tokens(context_text)
    memory_state["query_count"] = 0
    memory_state["total_saved_tokens"] = 0
    tok_str = format_tokens(memory_state["cached_tokens"])
    return (
        f"✅ Context set — **{tok_str} tokens** ready. Savings below are estimated based on context size.",
        _render_cache_stats(),
    )


def _render_cache_stats():
    q = memory_state["query_count"]
    saved = memory_state["total_saved_tokens"]
    cached_tok = memory_state["cached_tokens"]
    return (
        f"**Context tokens:** {format_tokens(cached_tok)}\n\n"
        f"**Queries run:** {q}\n\n"
        f"**Estimated tokens saved\\*:** {format_tokens(saved)}\n\n"
        f"**Estimated cost savings\\*:** ~${saved * 0.000003:.4f} @ $3/1M tokens\n\n"
        f"_\\* Estimates assume full KV cache reuse per query. Actual savings depend on server-side cache availability._"
    )


def memory_chat(user_msg, history, model):
    if not user_msg.strip():
        yield history, "—", "—", _render_cache_stats(), ""
        return

    cached_ctx = memory_state["cached_context"]
    system = (
        "You are a helpful assistant with a pre-loaded context. "
        "The context below has been KV-cached — it does not need to be re-encoded for each query.\n\n"
        f"=== CACHED CONTEXT ===\n{cached_ctx}\n=== END CONTEXT ==="
        if cached_ctx
        else "You are a helpful assistant. No context has been cached yet."
    )

    # Simulate cache hit: saved tokens = cached context tokens (not re-encoded)
    memory_state["query_count"] += 1
    memory_state["total_saved_tokens"] += memory_state["cached_tokens"]

    for history_out, tok_info, latency, err in stream_response(system, history, user_msg, model):
        # Annotate with cache hit badge
        cache_badge = "🟢 **Cache HIT (estimated)** — context eligible for KV cache reuse" if cached_ctx else "⚪ No cache"
        yield history_out, tok_info, latency, _render_cache_stats(), cache_badge


# ─────────────────────────────────────────────────────────────────────────────
# GRADIO UI
# ─────────────────────────────────────────────────────────────────────────────
CRUSOE_BLUE = "#1B4FCC"
CRUSOE_DARK = "#0D1B2A"

css = """
.crusoe-header { text-align: center; padding: 1.5rem 0 0.5rem; }
.token-badge { font-size: 1.1rem; font-weight: 600; color: #1B4FCC; }
.cache-stats { background: #f0f4ff; border-radius: 8px; padding: 1rem; }
.cache-hit { color: #16a34a; font-weight: 700; font-size: 1rem; }
.stat-row { display: flex; gap: 1.5rem; align-items: center; }
footer { display: none !important; }
"""

with gr.Blocks(title="Crusoe Foundry — Infinite Context Demo", theme=gr.themes.Soft(primary_hue="blue"), css=css) as demo:

    # ── Header ────────────────────────────────────────────────────────────────
    gr.HTML("""
    <div class="crusoe-header">
      <h1 style="font-size:1.8rem;font-weight:700;color:#0D1B2A;margin:0">
        Infinite Context Demo
      </h1>
      <p style="color:#555;margin:0.3rem 0 0">
        Powered by <strong>Crusoe Foundry</strong> &nbsp;·&nbsp;
        MemoryAlloy™ &amp; KV Cache Sharing
      </p>
    </div>
    """)

    with gr.Row():
        model_selector = gr.Dropdown(
            choices=AVAILABLE_MODELS,
            value=MODEL,
            label="Model",
            scale=2,
        )

    with gr.Tabs():

        # ── TAB 1: LEGAL ──────────────────────────────────────────────────────
        with gr.Tab("📄 Document Analysis"):
            gr.Markdown(
                "Upload any documents — ask questions "
                "across the **entire document** with no chunking or retrieval needed."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    legal_files = gr.File(
                        label="Upload Documents (PDF, TXT, MD)",
                        file_count="multiple",
                        file_types=[".pdf", ".txt", ".md", ".docx"],
                    )
                    legal_ingest_btn = gr.Button("📥 Load into Context", variant="primary")
                    legal_status = gr.Markdown("No documents loaded.")
                    legal_token_badge = gr.Markdown("", elem_classes=["token-badge"])
                    legal_preview = gr.Textbox(
                        label="Document Preview",
                        lines=6,
                        interactive=False,
                        placeholder="Document text will appear here after loading…",
                    )
                with gr.Column(scale=2):
                    legal_chatbot = gr.Chatbot(label="Document Q&A", height=420)
                    with gr.Row():
                        legal_input = gr.Textbox(
                            placeholder="e.g. Summarize the key points of this document.",
                            label="Ask a question",
                            scale=4,
                        )
                        legal_send = gr.Button("Send", variant="primary", scale=1)
                    with gr.Row():
                        legal_tok_info = gr.Markdown("", elem_classes=["token-badge"])
                        legal_latency = gr.Markdown("")
                    legal_err = gr.Markdown("", visible=False)
                    gr.Examples(
                        examples=[
                            ["Summarize the key points of this document."],
                            ["What are the main topics covered?"],
                            ["List every date or deadline mentioned."],
                            ["What conclusions or recommendations are made?"],
                            ["Extract all named entities (people, organizations, places)."],
                        ],
                        inputs=legal_input,
                    )

            legal_ingest_btn.click(
                legal_ingest,
                inputs=[legal_files],
                outputs=[legal_status, legal_token_badge, legal_preview],
            )

            def legal_submit(msg, history, model):
                yield from legal_chat(msg, history, model)

            legal_send.click(
                legal_submit,
                inputs=[legal_input, legal_chatbot, model_selector],
                outputs=[legal_chatbot, legal_tok_info, legal_latency, legal_err],
            ).then(lambda: "", outputs=legal_input)

            legal_input.submit(
                legal_submit,
                inputs=[legal_input, legal_chatbot, model_selector],
                outputs=[legal_chatbot, legal_tok_info, legal_latency, legal_err],
            ).then(lambda: "", outputs=legal_input)

        # ── TAB 2: DEV ────────────────────────────────────────────────────────
        with gr.Tab("💻 Codebase Intelligence"):
            gr.Markdown(
                "Upload source files or paste code — reason across your **entire codebase** "
                "simultaneously. No embeddings, no retrieval, no chunking."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    dev_files = gr.File(
                        label="Upload Source Files",
                        file_count="multiple",
                        file_types=[".py", ".js", ".ts", ".go", ".rs", ".java", ".txt", ".md"],
                    )
                    dev_paste = gr.Textbox(
                        label="Or paste code directly",
                        lines=8,
                        placeholder="Paste your code here…",
                    )
                    dev_ingest_btn = gr.Button("📥 Load Codebase", variant="primary")
                    dev_status = gr.Markdown("No code loaded.")
                    dev_token_badge = gr.Markdown("", elem_classes=["token-badge"])
                    dev_preview = gr.Textbox(
                        label="Codebase Preview",
                        lines=5,
                        interactive=False,
                        placeholder="Loaded code will appear here…",
                    )
                with gr.Column(scale=2):
                    dev_chatbot = gr.Chatbot(label="Codebase Q&A", height=420)
                    with gr.Row():
                        dev_input = gr.Textbox(
                            placeholder="e.g. Where is the authentication logic and how does it work?",
                            label="Ask about your codebase",
                            scale=4,
                        )
                        dev_send = gr.Button("Send", variant="primary", scale=1)
                    with gr.Row():
                        dev_tok_info = gr.Markdown("", elem_classes=["token-badge"])
                        dev_latency = gr.Markdown("")
                    dev_err = gr.Markdown("")
                    gr.Examples(
                        examples=[
                            ["Explain the overall architecture of this codebase."],
                            ["Where are potential race conditions or concurrency issues?"],
                            ["List all API endpoints and their HTTP methods."],
                            ["Which functions have no error handling?"],
                            ["How would I add rate limiting to this service?"],
                        ],
                        inputs=dev_input,
                    )

            dev_ingest_btn.click(
                dev_ingest,
                inputs=[dev_files, dev_paste],
                outputs=[dev_status, dev_token_badge, dev_preview],
            )

            def dev_submit(msg, history, model):
                yield from dev_chat(msg, history, model)

            dev_send.click(
                dev_submit,
                inputs=[dev_input, dev_chatbot, model_selector],
                outputs=[dev_chatbot, dev_tok_info, dev_latency, dev_err],
            ).then(lambda: "", outputs=dev_input)

            dev_input.submit(
                dev_submit,
                inputs=[dev_input, dev_chatbot, model_selector],
                outputs=[dev_chatbot, dev_tok_info, dev_latency, dev_err],
            ).then(lambda: "", outputs=dev_input)

        # ── TAB 3: MEMORY DEMO ────────────────────────────────────────────────
        with gr.Tab("🧠 MemoryAlloy™ Demo"):
            gr.Markdown(
                "See KV cache sharing in action. Set a large context once — every subsequent "
                "query reuses the **cached key-value representations**, slashing compute and cost.\n\n"
                "> **Note:** Token savings shown below are *estimated* based on context size. "
                "Actual cache reuse depends on server-side KV cache availability on Crusoe Foundry."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 1. Set Shared Context")
                    memory_context_input = gr.Textbox(
                        label="Context to cache (paste any large text)",
                        lines=12,
                        placeholder="Paste a large document, knowledge base, or system context here. "
                                    "This will be cached and reused across all queries.",
                    )
                    memory_cache_btn = gr.Button("🔒 Lock into KV Cache", variant="primary")
                    memory_cache_status = gr.Markdown("No context cached.")

                    gr.Markdown("### 2. Cache Stats")
                    memory_stats = gr.Markdown("", elem_classes=["cache-stats"])

                with gr.Column(scale=2):
                    gr.Markdown("### 3. Query Against Cached Context")
                    memory_chatbot = gr.Chatbot(
                        label="Memory-Augmented Chat",
                        height=380,
                    )
                    with gr.Row():
                        memory_input = gr.Textbox(
                            placeholder="Ask anything — the context is already cached…",
                            label="Your question",
                            scale=4,
                        )
                        memory_send = gr.Button("Send", variant="primary", scale=1)
                    with gr.Row():
                        memory_tok_info = gr.Markdown("", elem_classes=["token-badge"])
                        memory_latency = gr.Markdown("")
                    memory_cache_hit = gr.Markdown("", elem_classes=["cache-hit"])
                    memory_err = gr.Markdown("")
                    gr.Examples(
                        examples=[
                            ["Summarize the key points in 3 sentences."],
                            ["What topics are covered in this context?"],
                            ["Extract all named entities mentioned."],
                            ["What are the most important dates or numbers?"],
                        ],
                        inputs=memory_input,
                    )

            memory_cache_btn.click(
                memory_set_context,
                inputs=[memory_context_input],
                outputs=[memory_cache_status, memory_stats],
            )

            def memory_submit(msg, history, model):
                yield from memory_chat(msg, history, model)

            memory_send.click(
                memory_submit,
                inputs=[memory_input, memory_chatbot, model_selector],
                outputs=[memory_chatbot, memory_tok_info, memory_latency, memory_stats, memory_cache_hit],
            ).then(lambda: "", outputs=memory_input)

            memory_input.submit(
                memory_submit,
                inputs=[memory_input, memory_chatbot, model_selector],
                outputs=[memory_chatbot, memory_tok_info, memory_latency, memory_stats, memory_cache_hit],
            ).then(lambda: "", outputs=memory_input)

    # ── Footer ────────────────────────────────────────────────────────────────
    gr.HTML("""
    <div style="text-align:center;color:#888;padding:1.5rem 0 0.5rem;font-size:0.85rem">
      Built on <strong>Crusoe Foundry</strong> &nbsp;·&nbsp;
      Sustainable AI compute &nbsp;·&nbsp;
      <a href="https://crusoe.ai" target="_blank">crusoe.ai</a>
    </div>
    """)


demo.launch()
