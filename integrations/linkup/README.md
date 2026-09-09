# Grounded Answers with Crusoe and Linkup

Pair a Crusoe-hosted model with the [Linkup](https://linkup.so) search API to answer questions grounded in live web content. Linkup returns either raw search results or source-backed answers; this recipe uses raw results so the Crusoe model does the reasoning.

## Prerequisites

- A Crusoe Inference API key from the [Crusoe Console](https://console.crusoe.ai/) (**Security > Inference API Key**)
- A Linkup API key from [app.linkup.so](https://app.linkup.so)

```bash
pip install openai linkup-sdk
export CRUSOE_API_KEY="your-crusoe-key"
export LINKUP_API_KEY="your-linkup-key"
```

## Complete example

```python
import json
import os

from linkup import LinkupClient
from openai import OpenAI

crusoe = OpenAI(
    base_url="https://api.inference.crusoecloud.com/v1",
    api_key=os.environ["CRUSOE_API_KEY"],
)
linkup = LinkupClient(api_key=os.environ["LINKUP_API_KEY"])

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web and return sourced snippets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."}
                },
                "required": ["query"],
            },
        },
    }
]


def search_web(query: str) -> str:
    response = linkup.search(
        query=query,
        depth="standard",
        output_type="searchResults",
    )
    return json.dumps(
        [
            {"name": r.name, "url": r.url, "content": r.content}
            for r in response.results[:5]
        ]
    )


def run_agent(question: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are a research assistant. Use search_web for "
            "anything that needs current information. Cite sources by URL.",
        },
        {"role": "user", "content": question},
    ]
    while True:
        response = crusoe.chat.completions.create(
            model="zai/GLM-5.2",
            messages=messages,
            tools=tools,
        )
        message = response.choices[0].message
        messages.append(message)
        if not message.tool_calls:
            return message.content
        for call in message.tool_calls:
            args = json.loads(call.function.arguments)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": search_web(**args),
                }
            )


print(run_agent("What are the latest developments in open-weight LLMs?"))
```

## Notes

- Linkup's `depth="deep"` mode runs an iterative search for harder questions, at higher latency.
- For a one-shot pattern without an agent loop, request `output_type="sourcedAnswer"` from Linkup and use the Crusoe model to refine or expand the sourced answer.
- `zai/GLM-5.2` interleaves reasoning with tool calls; `google/gemma-4-31b-it` is a good pick when you want structured JSON output from the final answer step.
