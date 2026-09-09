---
title: Crusoe Docs Analysis
emoji: 🌍
colorFrom: gray
colorTo: gray
sdk: gradio
sdk_version: 6.9.0
app_file: app.py
pinned: false
short_description: Short demo for docs and code interacting with Managed AI
---

# Crusoe Space Legal Docs

**Document analysis, codebase intelligence, and KV cache demo — powered by Crusoe Cloud Foundry**

A Gradio app showcasing Crusoe Foundry's long-context capabilities. Upload documents or code and interact with them using 7 open-source LLMs — no chunking, no embeddings, no RAG pipeline. Just raw context window power.

---

## Features

### 1. Document Analysis
Upload PDFs, TXT, MD, or DOCX files and ask questions about them. The entire document is passed to the model's context window — no chunking or retrieval needed. Responses cite relevant sections.

### 2. Codebase Intelligence
Upload source files (.py, .js, .ts, .go, .rs, .java) or paste code directly. Get architecture analysis, bug detection, API endpoint mapping, and error handling review — all from whole-codebase analysis.

### 3. MemoryAlloy Demo
Demonstrates KV cache sharing — set a large context once, then reuse it across multiple queries. Shows estimated token savings and cost reduction ($3/1M tokens).

---

## Architecture

```
crusoe-hf-space-legal/
├── app.py                # Full Gradio app — 3 tabs, 7 models, file upload + analysis
├── requirements.txt
└── README.md
```

### Single-file design

The entire app lives in `app.py` (~518 lines):

- **Lines 1-50:** Config — OpenAI client, model registry, API setup
- **Lines 50-100:** Utilities — token counting (tiktoken), file readers (PDF, TXT, MD)
- **Lines 100-250:** Tab 1 — Document Analysis with file upload + Q&A
- **Lines 250-380:** Tab 2 — Codebase Intelligence with code paste + file upload
- **Lines 380-500:** Tab 3 — MemoryAlloy KV cache demo
- **Lines 500-518:** Gradio app launch

## Models Available

| Model | ID |
|-------|----|
| Qwen3 235B | `Qwen/Qwen3-235B-A22B-Instruct-2507` |
| DeepSeek R1 | `deepseek-ai/DeepSeek-R1` |
| Kimi K2 | `moonshotai/Kimi-K2-Instruct` |
| DeepSeek V3 | `deepseek-ai/DeepSeek-V3-0324` |
| Llama 3.3 70B | `meta-llama/Llama-3.3-70B-Instruct` |
| GPT-OSS 120B | `openai/gpt-osse-120b` |
| Gemma 3 12B | `google/gemma-3-12b-it` |

---

## Quick Start

### Prerequisites

- Python 3.9+
- A [Crusoe Cloud](https://crusoe.ai) API key

### Local Development

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/crusoe-space-legal-docs
cd crusoe-space-legal-docs

pip install -r requirements.txt
export CRUSOE_API_KEY=your_key_here
python app.py
```

Open **http://localhost:7860**.

### Deploy to HuggingFace Spaces

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space)
2. Select **Gradio** as the SDK
3. Add `CRUSOE_API_KEY` as a Space secret in Settings
4. Push:

```bash
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/crusoe-space-legal-docs
git push hf main
```

---

## Use as a Framework

### Add a new model

In `app.py`, add to the `MODELS` list:

```python
MODELS = [
    ...
    "org/your-new-model",
]
```

### Add a new tab

Gradio's `gr.Tabs()` makes this straightforward:

```python
with gr.Tabs():
    with gr.Tab("Your New Tab"):
        # Your UI components here
        input_box = gr.Textbox(label="Input")
        output_box = gr.Textbox(label="Output")
        btn = gr.Button("Analyze")
        btn.click(fn=your_function, inputs=[input_box], outputs=[output_box])
```

### Swap the endpoint

Works with any OpenAI-compatible API:

```python
client = OpenAI(
    base_url="https://your-endpoint.com/v1/",
    api_key=os.environ.get("YOUR_API_KEY"),
)
```

### Add new file type support

Extend the file reader function in `app.py`:

```python
def read_file(file_path):
    if file_path.endswith(".docx"):
        # Add docx parsing
    ...
```

---

## API Reference (Crusoe Cloud Foundry)

All inference calls use the OpenAI-compatible SDK:

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://managed-inference-api-proxy.crusoecloud.com/v1/",
    api_key="YOUR_CRUSOE_API_KEY",
)

response = client.chat.completions.create(
    model="Qwen/Qwen3-235B-A22B-Instruct-2507",
    messages=[{"role": "user", "content": "Analyze this document..."}],
    temperature=1,
    top_p=0.95,
)
```

```typescript
import OpenAI from 'openai';

const client = new OpenAI({
  baseURL: 'https://managed-inference-api-proxy.crusoecloud.com/v1/',
  apiKey: 'YOUR_CRUSOE_API_KEY',
});

const response = await client.chat.completions.create({
  model: 'Qwen/Qwen3-235B-A22B-Instruct-2507',
  messages: [{ role: 'user', content: 'Analyze this document...' }],
});
```

---

## Dependencies

```
gradio>=6.0.0
openai>=1.30.0
tiktoken>=0.7.0
pdfminer.six>=20221105
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `CRUSOE_API_KEY` | Yes | Your Crusoe Cloud Foundry API key |

---

## License

MIT — fork it, remix it, ship it.

Built by Crusoe AI Developer Relations · [crusoe.ai](https://crusoe.ai)
