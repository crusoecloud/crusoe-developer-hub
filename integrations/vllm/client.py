"""vLLM features on Crusoe Managed Inference through the OpenAI SDK.

Streaming, JSON-schema guided decoding, and the model catalog. Requires CRUSOE_API_KEY.
Makes live requests that incur charges.
"""

import json
import os

from openai import OpenAI

MODEL = "zai/GLM-5.3"

client = OpenAI(
    base_url="https://api.inference.crusoecloud.com/v1",
    api_key=os.environ["CRUSOE_API_KEY"],
)


def stream_answer() -> None:
    print("--- streaming ---")
    with client.chat.completions.stream(
        model=MODEL,
        messages=[{"role": "user", "content": "In two sentences, what is continuous batching?"}],
        max_tokens=200,
    ) as stream:
        for event in stream:
            if event.type == "content.delta":
                print(event.delta, end="", flush=True)
    print()


def guided_json() -> None:
    print("--- guided decoding (json_schema) ---")
    schema = {
        "type": "object",
        "properties": {
            "gpu": {"type": "string"},
            "memory_gb": {"type": "integer"},
            "fits_70b_fp8": {"type": "boolean"},
        },
        "required": ["gpu", "memory_gb", "fits_70b_fp8"],
        "additionalProperties": False,
    }
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Describe the NVIDIA H100 SXM for serving a 70B FP8 model."}],
        response_format={"type": "json_schema", "json_schema": {"name": "gpu_fit", "schema": schema}},
        max_tokens=200,
    )
    print(json.dumps(json.loads(response.choices[0].message.content), indent=2))


def catalog() -> None:
    print("--- models ---")
    for model in sorted(client.models.list(), key=lambda m: m.id):
        print(f"  {model.id}")


if __name__ == "__main__":
    stream_answer()
    guided_json()
    catalog()
