# Helicone with Crusoe

Log, trace, and cost-track Crusoe Managed Inference requests with [Helicone](https://helicone.ai). Helicone's AI Gateway proxies any OpenAI-compatible endpoint, so Crusoe works today by adding one target header. Native cost mappings for Crusoe models are open upstream ([Helicone/helicone#5777](https://github.com/Helicone/helicone/pull/5777)); once merged, Helicone prices Crusoe requests automatically instead of showing token counts only.

## Prerequisites

- A Crusoe Inference API key from the [Crusoe Console](https://console.crusoecloud.com/) (**Security > Inference API Key**)
- A Helicone API key from [helicone.ai](https://helicone.ai)

## Setup

Point your OpenAI client at Helicone's gateway and name Crusoe as the target:

```python
import os

from openai import OpenAI

client = OpenAI(
    base_url="https://gateway.helicone.ai/v1",
    api_key=os.environ["CRUSOE_API_KEY"],
    default_headers={
        "Helicone-Auth": f"Bearer {os.environ['HELICONE_API_KEY']}",
        "Helicone-Target-Url": "https://api.inference.crusoecloud.com",
        "Helicone-Property-App": "crusoe-demo",
    },
)

response = client.chat.completions.create(
    model="zai/GLM-5.3",
    messages=[{"role": "user", "content": "Hello from Helicone"}],
)
print(response.choices[0].message.content)
```

Every request then appears in the Helicone dashboard with latency, tokens, and the custom property, and `Helicone-Session-Id` headers group multi-step agent runs into one trace.

## Notes

- Crusoe's base path is `/v1`; the gateway appends it, so `Helicone-Target-Url` takes the host only.
- Streaming responses pass through unchanged.
- Until the cost mappings merge, add Crusoe rates from the [pricing page](https://crusoe.ai/cloud/pricing) as a custom cost in Helicone if you need dollar figures.

[Back to integrations](../README.md)
