"""Turn a fine-tuned Crusoe deployment into a decision model: text in, probability per class out.

A decision model doesn't write text, it scores a closed set of options. You get
that from any fine-tuned LLM by reading its first-token logprobs instead of its
output. Because every label starts with a different token (checked in
01_prepare_data.py), and after fine-tuning the rest of the label follows with
~100% certainty,
one `max_tokens=1` request with `top_logprobs=20` returns the model's full
distribution over the labels: one round trip, one generated token, exact.

    export CRUSOE_API_KEY='...'
    python decision_model.py agnews-decider "Stocks rallied after the Fed held rates."
"""
import asyncio
import math
import os
import sys

import httpx

LABELS = ["World", "Sports", "Business", "Technology"]
SYSTEM = (
    "You are a news topic classifier. Classify the news article into exactly one of these "
    "categories: World, Sports, Business, Technology.\n\nRespond with ONLY the category name, nothing else."
)
INFERENCE_URL = os.environ.get("INFERENCE_URL", "https://api.inference.crusoecloud.com/v1")
TRUST_ENV = os.environ.get("TRUST_ENV") == "1"  # skill default is off; set TRUST_ENV=1 when you must egress via a proxy
TOP_LOGPROBS = 20  # vLLM's default --max-logprobs, enforced by Crusoe inference; covers any label set up to 20


def token(env_var: str = "CRUSOE_API_KEY") -> str:
    try:
        return os.environ[env_var]
    except KeyError:
        raise SystemExit(f"export {env_var}='<your Crusoe Inference API key>' first (console.crusoecloud.com/security/inference-api-keys)")


def inference_client() -> httpx.AsyncClient:
    """One shared client: keep-alive connections, so only the first call pays TLS setup."""
    return httpx.AsyncClient(
        base_url=INFERENCE_URL, timeout=60, trust_env=TRUST_ENV,
        headers={"Authorization": f"Bearer {token()}"},
        limits=httpx.Limits(max_connections=64),
    )


def probabilities(top_logprobs: list[dict], labels: list[str] = LABELS, temperature: float = 1.0) -> dict:
    """One position's top-k logprobs -> class probabilities.

    Softmax over the label tokens only, so the answer always sums to 1. `temperature` > 1
    softens an over-confident SFT model (fit it on validation data: see 04_eval.py).
    `unassigned` is the probability mass the model put on anything that is not a label.
    """
    seen = {}  # whitespace variants (" Business", "\nBusiness") are the same decision: add their probabilities
    for t in top_logprobs:
        key = t["token"].strip()
        seen[key] = math.log(math.exp(seen[key]) + math.exp(t["logprob"])) if key in seen else t["logprob"]
    present = [l for l in labels if l in seen]
    if not present:
        return {"choice": None, "confidence": 0.0, "probabilities": dict.fromkeys(labels, 0.0),
                "logprobs": {}, "unassigned": 1.0}
    peak = max(seen[l] for l in present)
    weights = {l: math.exp((seen[l] - peak) / temperature) for l in present}
    total = sum(weights.values())
    probs = {l: weights.get(l, 0.0) / total for l in labels}
    choice = max(probs, key=probs.get)
    return {"choice": choice, "confidence": probs[choice], "probabilities": probs,
            "logprobs": {l: seen[l] for l in present},  # raw, for fitting a calibration temperature
            "unassigned": max(0.0, 1.0 - sum(math.exp(seen[l]) for l in present))}


async def decide(client: httpx.AsyncClient, model: str, article: str, temperature: float = 1.0) -> dict:
    """One request to the deployment -> {choice, confidence, probabilities, unassigned, logprobs}."""
    resp = await client.post("/chat/completions", json={
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": article}],
        "max_tokens": 1,            # we only read the first token's scores, so generate exactly one
        "temperature": 0,
        "logprobs": True,
        "top_logprobs": TOP_LOGPROBS,
        "prompt_cache_key": "decision-model",  # same system prompt every time: let the server reuse its KV cache
        # Qwen3.5 is a thinking model and our SFT data has no thinking turns (see the skill's dataset-format notes)
        "chat_template_kwargs": {"enable_thinking": False},
    })
    resp.raise_for_status()
    return probabilities(resp.json()["choices"][0]["logprobs"]["content"][0]["top_logprobs"], temperature=temperature)


async def decide_many(client: httpx.AsyncClient, model: str, articles: list[str],
                      temperature: float = 1.0, concurrency: int = 16) -> list[dict]:
    """Many decisions concurrently over the shared connection pool; results keep input order."""
    gate = asyncio.Semaphore(concurrency)

    async def one(article):
        async with gate:
            return await decide(client, model, article, temperature)
    return await asyncio.gather(*(one(a) for a in articles))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    model, articles = sys.argv[1], sys.argv[2:]

    async def main():
        async with inference_client() as client:
            for article, out in zip(articles, await decide_many(client, model, articles)):
                ranked = sorted(out["probabilities"].items(), key=lambda kv: -kv[1])
                print(f"{article[:70]!r}\n  -> {out['choice']} ({out['confidence']:.3f})  "
                      + "  ".join(f"{l} {p:.3f}" for l, p in ranked) + f"  [off-label mass {out['unassigned']:.4f}]")
    asyncio.run(main())
