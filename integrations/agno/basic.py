"""Agno on Crusoe: sync, streaming, and async runs.

Requires CRUSOE_API_KEY in the environment. Makes live requests that incur charges.
"""

import asyncio

from agno.agent import Agent

from crusoe_model import crusoe

agent = Agent(model=crusoe("zai/GLM-5.3"), markdown=True)

PROMPT = "Write a two sentence horror story set in a data center at 3 a.m."

if __name__ == "__main__":
    agent.print_response(PROMPT)
    agent.print_response(PROMPT, stream=True)
    asyncio.run(agent.aprint_response(PROMPT, stream=True))
