"""aisuite on Crusoe: one prompt, two models, only the model string changes.

Requires CRUSOE_API_KEY in the environment. Makes live requests that incur charges.
"""

import aisuite as ai

client = ai.Client()

MODELS = ["crusoe:zai/GLM-5.3", "crusoe:meta-llama/Llama-3.3-70B-Instruct"]

messages = [
    {"role": "system", "content": "You are a helpful assistant. Answer in one sentence."},
    {"role": "user", "content": "What is the advantage of serverless inference for a new product?"},
]

if __name__ == "__main__":
    for model in MODELS:
        response = client.chat.completions.create(model=model, messages=messages, temperature=0.2)
        print(f"{model}\n  {response.choices[0].message.content}\n")
