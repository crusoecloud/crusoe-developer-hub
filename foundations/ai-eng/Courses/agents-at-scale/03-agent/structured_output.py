import asyncio, sys
from config import settings, client
from schemas import Brief

NOTES = """
Findings gathered earlier in the run, each with the page it came from:
- https://docs.crusoe.ai/serverless-inference/overview : "Serverless Inference lets you run inference on open-weight models through an OpenAI-compatible API without managing infrastructure."
- https://docs.crusoe.ai/serverless-inference/overview : "The base URL and the model identifier are the only two values a client needs to change to move between models."
- https://modelcontextprotocol.io/specification/2026-07-28/changelog : "The protocol core is now stateless. Every request carries its own protocol version, client identity and capabilities."
"""


async def main(topic: str) -> None:
    messages = [
        {"role": "system", "content": "Write a research brief from the findings you are given. Quote only text that appears in the findings."},
        {"role": "user", "content": f"Topic: {topic}\n{NOTES}"},
    ]
    resp = await client.chat.completions.create(
        model=settings.model, messages=messages,
        response_format={"type": "json_schema",
                         "json_schema": {"name": "Brief", "schema": Brief.model_json_schema(), "strict": True}})
    brief = Brief.model_validate_json(resp.choices[0].message.content)
    print(brief.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main(" ".join(sys.argv[1:]) or "running open models through one endpoint"))
