# Building a Web-Search Agent with Crusoe and Tavily

Pair a Crusoe-hosted model with the [Tavily](https://tavily.com) search API to build an agent that answers questions with live web results. Crusoe's OpenAI-compatible tool calling handles the agent loop; Tavily provides the search tool.

## Prerequisites

- A Crusoe Inference API key from the [Crusoe Console](https://console.crusoe.ai/) (**Security > Inference API Key**)
- A Tavily API key from [app.tavily.com](https://app.tavily.com)

```bash
pip install openai tavily-python
export CRUSOE_API_KEY="your-crusoe-key"
export TAVILY_API_KEY="your-tavily-key"
```

## Complete example

```python
import json
import os

from openai import OpenAI
from tavily import TavilyClient

crusoe = OpenAI(
    base_url="https://api.inference.crusoecloud.com/v1",
    api_key=os.environ["CRUSOE_API_KEY"],
)
tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

tools = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information.",
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


def web_search(query: str) -> str:
    results = tavily.search(query=query, max_results=5)
    return json.dumps(
        [
            {"title": r["title"], "url": r["url"], "content": r["content"]}
            for r in results["results"]
        ]
    )


def run_agent(question: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are a research assistant. Use web_search for "
            "anything that needs current information, then answer with "
            "citations.",
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
                    "content": web_search(**args),
                }
            )


print(run_agent("What happened in AI news this week?"))
```

## How it works

1. The model receives the question and the `web_search` tool definition.
2. When it needs current information, it returns a `tool_calls` response instead of an answer.
3. The loop executes the Tavily search and feeds the results back as a `tool` message.
4. The model composes the final answer from the search results, with citations.

## Notes

- `zai/GLM-5.2` interleaves reasoning with tool calls, which improves multi-step research quality. Swap in `openai/gpt-oss-120b` for controllable reasoning effort or `moonshotai/Kimi-K2.6` for long agentic sessions.
- Tavily's `search` accepts `search_depth="advanced"`, `include_domains`, and `time_range` parameters for tighter queries; see the [Tavily docs](https://docs.tavily.com).
