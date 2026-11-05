"""Pydantic AI on Crusoe: a validated Pydantic model as the agent's output.

Crusoe serves every model with guided decoding, so NativeOutput works across the catalog.
Requires CRUSOE_API_KEY. Incurs inference charges.
"""

from pydantic import BaseModel, Field
from pydantic_ai import Agent, NativeOutput

MODEL = "crusoe:zai/GLM-5.3"


class SiteAssessment(BaseModel):
    """A short go/no-go on a candidate data center site."""

    site: str
    verdict: str = Field(description="GO, CONDITIONAL, or NO_GO")
    score: int = Field(ge=0, le=100, description="Overall suitability, 0 to 100")
    risks: list[str] = Field(description="Up to three risks, each under ten words")


agent = Agent(
    MODEL,
    output_type=NativeOutput(SiteAssessment),
    instructions="You assess sites for a 100 MW AI data center. Be decisive.",
)

if __name__ == "__main__":
    result = agent.run_sync(
        "Narvik, Norway: hydro power at 2.5 cents per kWh, 4 C average temperature, "
        "fiber to Oslo and London, one 420 kV substation with spare capacity."
    )
    assessment = result.output
    print(assessment.model_dump_json(indent=2))
    assert isinstance(assessment, SiteAssessment)
