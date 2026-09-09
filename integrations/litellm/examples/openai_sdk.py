import os

from openai import OpenAI

client = OpenAI(
    base_url=os.environ.get("GATEWAY", "http://localhost:4000") + "/v1",
    api_key=os.environ.get("LITELLM_MASTER_KEY", "sk-local-dev-master"),
)

response = client.chat.completions.create(
    model="glm-5.2",
    messages=[{"role": "user", "content": "What makes a neocloud different from a hyperscaler? Two sentences."}],
)
print(response.choices[0].message.content)

print("---")

stream = client.chat.completions.create(
    model="glm-5.2",
    messages=[{"role": "user", "content": "Write a haiku about GPUs."}],
    stream=True,
)
for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()
