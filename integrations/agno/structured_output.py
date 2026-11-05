"""Agno on Crusoe: a Pydantic output schema, enforced by guided decoding.

Requires CRUSOE_API_KEY in the environment. Makes live requests that incur charges.
"""

from agno.agent import Agent
from pydantic import BaseModel, Field

from crusoe_model import crusoe


class IncidentSummary(BaseModel):
    title: str = Field(description="Under ten words")
    severity: str = Field(description="SEV1, SEV2, or SEV3")
    root_cause: str
    actions: list[str] = Field(description="Up to three follow-up actions")


agent = Agent(
    model=crusoe("google/gemma-4-31b-it"),
    description="You write post-incident summaries for a cloud platform team.",
    output_schema=IncidentSummary,
)

if __name__ == "__main__":
    response = agent.run(
        "At 02:14 UTC a BGP misconfiguration withdrew the inference API's prefix for 11 minutes. "
        "Customers saw connection timeouts. The on-call reverted the change at 02:25."
    )
    print(response.content.model_dump_json(indent=2))
