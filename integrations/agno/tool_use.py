"""Agno on Crusoe: a custom tool the agent calls.

Requires CRUSOE_API_KEY in the environment. Makes live requests that incur charges.
"""

from agno.agent import Agent

from crusoe_model import crusoe

GPU_SPECS = {
    "h100": {"memory_gb": 80, "tdp_w": 700, "fp8_tflops": 3958},
    "b200": {"memory_gb": 192, "tdp_w": 1000, "fp8_tflops": 9000},
    "mi355x": {"memory_gb": 288, "tdp_w": 1400, "fp8_tflops": 10100},
}


def gpu_spec(gpu: str) -> dict:
    """Look up memory, power, and FP8 throughput for a GPU.

    Args:
        gpu: GPU name such as "h100", "b200", or "mi355x".
    """
    return GPU_SPECS.get(gpu.lower(), {"error": f"unknown gpu {gpu}; known: {sorted(GPU_SPECS)}"})


agent = Agent(
    model=crusoe("zai/GLM-5.3"),
    tools=[gpu_spec],
    instructions=["Use gpu_spec for any hardware number.", "Answer in under 80 words."],
    markdown=True,
)

if __name__ == "__main__":
    agent.print_response(
        "A 70B model in FP8 needs about 70 GB of weights plus 30 GB of KV cache. "
        "Which of h100, b200, or mi355x fits it on one GPU, and which draws the least power?",
        stream=True,
    )
