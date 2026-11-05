"""Pydantic AI on Crusoe: a plain run, a streamed run, and token usage.

Requires CRUSOE_API_KEY in the environment. Makes live requests that incur charges.
"""

import asyncio

from pydantic_ai import Agent

MODEL = "crusoe:zai/GLM-5.3"

agent = Agent(
    MODEL,
    instructions="You are a concise infrastructure engineer. Answer in two sentences.",
)


async def main() -> None:
    # One request, one answer. (Outside an event loop, agent.run_sync(...) does the same.)
    result = await agent.run("Why does GPU utilization matter for inference cost?")
    print(result.output)
    print(f"\n[usage] {result.usage()}")

    # The same question, streamed token by token.
    print("\n--- streaming ---")
    async with agent.run_stream("Give one tip for lowering time to first token.") as stream:
        async for text in stream.stream_text(delta=True):
            print(text, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
