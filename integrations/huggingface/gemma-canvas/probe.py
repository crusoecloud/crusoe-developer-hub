"""Probe whether google/gemma-4-31b-it on Crusoe Foundry accepts multimodal input.

Run:  CRUSOE_API_KEY=... python probe.py

Exits 0 on success (vision works), 1 on failure (model is text-only or errored).
"""
from __future__ import annotations

import os
import sys

from openai import OpenAI

# Small public test image (Wikipedia commons PNG transparency demo)
TEST_IMAGE_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/"
    "PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"
)

MODEL = os.getenv("GEMMA_MODEL") or os.getenv("DEBATE_MODEL", "google/gemma-4-31b-it")
BASE_URL = os.getenv(
    "CRUSOE_API_BASE",
    "https://api.inference.crusoecloud.com/v1/",
)

api_key = os.environ.get("CRUSOE_API_KEY")
if not api_key:
    print("ERROR: set CRUSOE_API_KEY", file=sys.stderr)
    sys.exit(2)

client = OpenAI(base_url=BASE_URL, api_key=api_key)

print(f"Probing {MODEL} for vision support via {BASE_URL}…\n")

try:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": TEST_IMAGE_URL}},
                    {"type": "text", "text": "Describe this image in one sentence."},
                ],
            }
        ],
        temperature=0.3,
        max_tokens=120,
    )
    text = resp.choices[0].message.content or ""
    print("SUCCESS — vision accepted. Model output:\n")
    print(text.strip())
    sys.exit(0)
except Exception as exc:  # noqa: BLE001
    print("FAILURE — model rejected multimodal input or errored:")
    print(f"  {type(exc).__name__}: {exc}")
    print("\nIf the error mentions 'image', 'content type', or 'unsupported',")
    print("Gemma 4 on Crusoe is likely text-only. Pivot to Option 2 (Gemma as")
    print("prompt engineer + separate image-gen API).")
    sys.exit(1)
