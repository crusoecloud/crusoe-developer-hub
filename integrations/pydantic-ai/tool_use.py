"""Pydantic AI on Crusoe: a tool the model calls, with typed dependencies.

The tool reads a small rate table so the sample runs without a second API. Swap it for a real
lookup and the agent loop stays the same. Requires CRUSOE_API_KEY. Incurs inference charges.
"""

from dataclasses import dataclass

from pydantic_ai import Agent, RunContext

MODEL = "crusoe:zai/GLM-5.3"


@dataclass
class Catalog:
    """Dependencies injected into every tool call."""

    rates_per_million_tokens: dict[str, tuple[float, float]]


catalog = Catalog(
    rates_per_million_tokens={
        "zai/GLM-5.3": (0.40, 1.26),
        "openai/gpt-oss-120b": (0.10, 0.50),
        "meta-llama/Llama-3.3-70B-Instruct": (0.25, 0.75),
    }
)

agent = Agent(
    MODEL,
    deps_type=Catalog,
    instructions=(
        "You help teams estimate inference spend. Use the rate tool for any price; "
        "never guess a number. Show the arithmetic briefly."
    ),
)


@agent.tool
def get_rate(ctx: RunContext[Catalog], model_id: str) -> str:
    """Return the input and output price per million tokens for a model on Crusoe.

    Args:
        model_id: The model id as listed on Crusoe, for example "zai/GLM-5.3".
    """
    rate = ctx.deps.rates_per_million_tokens.get(model_id)
    if rate is None:
        return f"{model_id} is not in the rate table. Known: {sorted(ctx.deps.rates_per_million_tokens)}"
    return f"{model_id}: ${rate[0]:.2f} per 1M input tokens, ${rate[1]:.2f} per 1M output tokens"


if __name__ == "__main__":
    result = agent.run_sync(
        "A support bot sends 2 million input tokens and 400k output tokens a day on gpt-oss-120b. "
        "What does a 30-day month cost?",
        deps=catalog,
    )
    print(result.output)
    print("\n[tool calls]")
    for message in result.all_messages():
        for part in message.parts:
            if part.part_kind == "tool-call":
                print(f"  {part.tool_name}({part.args})")
