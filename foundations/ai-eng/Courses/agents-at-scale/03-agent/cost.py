import tomllib
from pathlib import Path

PRICE = tomllib.loads(Path(__file__).with_name("prices.toml").read_text())


def cost_usd(usage, model: str) -> float:
    p = PRICE.get(model.removeprefix("crusoe:"))
    if not p:
        return 0.0
    uncached = usage.prompt_tokens - usage.cached_tokens
    return (uncached * p["input"] + usage.cached_tokens * p["cached_input"] + usage.completion_tokens * p["output"]) / 1e6
