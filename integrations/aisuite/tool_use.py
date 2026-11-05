"""aisuite on Crusoe: automatic tool calling with max_turns.

aisuite reads the function signature and docstring to build the tool schema, runs the tool when
the model asks for it, and feeds the result back. Requires CRUSOE_API_KEY. Incurs charges.
"""

import aisuite as ai

client = ai.Client()


def gpu_availability(region: str, gpu_type: str) -> dict:
    """Check how many GPUs of a type are available in a Crusoe region.

    Args:
        region: Region slug, for example "us-east1" or "eu-iceland1".
        gpu_type: GPU model, for example "h100", "b200", or "mi355x".
    """
    inventory = {
        ("us-east1", "h100"): 512,
        ("us-east1", "b200"): 64,
        ("eu-iceland1", "h100"): 256,
        ("eu-iceland1", "mi355x"): 128,
    }
    return {"region": region, "gpu_type": gpu_type, "available": inventory.get((region, gpu_type), 0)}


if __name__ == "__main__":
    response = client.chat.completions.create(
        model="crusoe:zai/GLM-5.3",
        messages=[
            {
                "role": "user",
                "content": "I need 200 H100s in Europe and 50 B200s in the US. Can Crusoe cover both today?",
            }
        ],
        tools=[gpu_availability],
        max_turns=5,
    )
    print(response.choices[0].message.content)
    for message in response.choices[0].intermediate_messages:
        if getattr(message, "tool_calls", None):
            for call in message.tool_calls:
                print(f"[tool] {call.function.name}({call.function.arguments})")
