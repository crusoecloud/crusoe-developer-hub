"""The same client code against Crusoe Managed Inference or your own vLLM server.

Set VLLM_BASE_URL and VLLM_API_KEY (and VLLM_MODEL) to point at either. Defaults target Crusoe.
Makes live requests; Crusoe requests incur charges.
"""

import os

from openai import OpenAI

base_url = os.environ.get("VLLM_BASE_URL", "https://api.inference.crusoecloud.com/v1")
api_key = os.environ.get("VLLM_API_KEY") or os.environ.get("CRUSOE_API_KEY", "unused")
model = os.environ.get("VLLM_MODEL", "zai/GLM-5.3")

client = OpenAI(base_url=base_url, api_key=api_key)

if __name__ == "__main__":
    print(f"endpoint: {base_url}\nmodel:    {model}\n")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Answer in one sentence."},
            {"role": "user", "content": "What does PagedAttention do?"},
        ],
        max_tokens=120,
    )
    print(response.choices[0].message.content)
    print(f"\n[usage] prompt={response.usage.prompt_tokens} completion={response.usage.completion_tokens}")
