"""SGLang frontend programs on Crusoe Managed Inference.

Multi-turn, fork, batch, and streaming through the built-in Crusoe backend.
Requires CRUSOE_API_KEY in the environment. Makes live requests that incur charges.
"""

import os

import sglang as sgl

MODEL = "zai/GLM-5.3"

if hasattr(sgl, "Crusoe"):
    # sglang >= 0.5.11 ships the Crusoe backend (Linux wheels).
    backend = sgl.Crusoe(MODEL)
else:
    # Older releases, including the last macOS wheels: the same thing through the OpenAI backend.
    backend = sgl.OpenAI(
        MODEL,
        base_url="https://api.inference.crusoecloud.com/v1/",
        api_key=os.environ["CRUSOE_API_KEY"],
    )

sgl.set_default_backend(backend)


@sgl.function
def multi_turn(s, topic):
    s += sgl.system("You are a terse SRE. One sentence per answer.")
    s += sgl.user(f"What is the biggest operational risk of {topic}?")
    s += sgl.assistant(sgl.gen("risk", max_tokens=80))
    s += sgl.user("And the cheapest mitigation?")
    s += sgl.assistant(sgl.gen("mitigation", max_tokens=80))


@sgl.function
def compare(s, workload):
    s += sgl.system("You compare GPU options for a workload. One sentence each.")
    s += sgl.user(f"Workload: {workload}")
    forks = s.fork(3)
    for fork, gpu in zip(forks, ["H100", "B200", "MI355X"]):
        fork += sgl.assistant(f"{gpu}: " + sgl.gen("verdict", max_tokens=60))
    forks.join()
    s["verdicts"] = [f["verdict"] for f in forks]


if __name__ == "__main__":
    print("--- multi-turn ---")
    state = multi_turn.run(topic="air-cooled 100 kW racks")
    print("risk:      ", state["risk"].strip())
    print("mitigation:", state["mitigation"].strip())

    print("\n--- fork (3 branches in parallel) ---")
    state = compare.run(workload="serving a 70B model at 2k requests per second")
    for gpu, verdict in zip(["H100", "B200", "MI355X"], state["verdicts"]):
        print(f"{gpu}: {verdict.strip()}")

    print("\n--- batch ---")
    states = multi_turn.run_batch(
        [{"topic": "liquid cooling"}, {"topic": "a single-region deployment"}],
        progress_bar=False,
    )
    for st in states:
        print("-", st["risk"].strip())

    print("\n--- streaming ---")
    state = multi_turn.run(topic="running inference without rate limits", stream=True)
    for chunk in state.text_iter("risk"):
        print(chunk, end="", flush=True)
    print()
