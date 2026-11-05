"""Strands Agents on Crusoe: two custom @tool functions the agent chains together.

Requires CRUSOE_API_KEY in the environment. Makes live requests that incur charges.
"""

import os

from strands import Agent, tool
from strands.models.openai import OpenAIModel

MODEL_ID = "zai/GLM-5.3"


@tool
def rack_power_kw(gpus: int, watts_per_gpu: int = 700, overhead: float = 1.35) -> float:
    """Estimate the power draw of a rack in kilowatts.

    Args:
        gpus: Number of GPUs in the rack.
        watts_per_gpu: Board power of one GPU in watts.
        overhead: Multiplier for CPUs, memory, networking, and cooling.
    """
    return round(gpus * watts_per_gpu * overhead / 1000, 1)


@tool
def generators_needed(load_kw: float, generator_kw: float, spares: int = 1) -> int:
    """Count the generators that cover a load, with spare units for N+1 redundancy.

    Args:
        load_kw: Total load in kilowatts.
        generator_kw: Rating of one generator in kilowatts.
        spares: Extra units to keep in reserve.
    """
    import math

    return math.ceil(load_kw / generator_kw) + spares


model = OpenAIModel(
    client_args={
        "api_key": os.environ["CRUSOE_API_KEY"],
        "base_url": "https://api.inference.crusoecloud.com/v1/",
    },
    model_id=MODEL_ID,
    params={"max_tokens": 2048, "temperature": 0.1},
)

agent = Agent(
    model=model,
    tools=[rack_power_kw, generators_needed],
    system_prompt=(
        "You size AI infrastructure. Use rack_power_kw for rack estimates and "
        "generators_needed for backup power; never do that arithmetic yourself. "
        "Keep answers under 80 words."
    ),
)

if __name__ == "__main__":
    agent(
        "A hall has 48 racks of 8 GPUs at 1000 W each. What is the hall's power draw, "
        "and how many 2.5 MW generators cover it with one spare?"
    )
