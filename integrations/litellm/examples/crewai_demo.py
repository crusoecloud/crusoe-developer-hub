import os

from crewai import Agent, Crew, Task
from crewai.llm import LLM

llm = LLM(
    model="openai/glm-5.2",
    base_url=os.environ.get("GATEWAY", "http://localhost:4000") + "/v1",
    api_key=os.environ.get("LITELLM_MASTER_KEY", "sk-local-dev-master"),
)

researcher = Agent(
    role="Research analyst",
    goal="Produce crisp technical summaries",
    backstory="A precise analyst who never pads an answer.",
    llm=llm,
)

task = Task(
    description="Summarize the trade-offs between running LLM inference on a neocloud versus a hyperscaler, in three bullet points.",
    expected_output="Three bullet points.",
    agent=researcher,
)

crew = Crew(agents=[researcher], tasks=[task])
print(crew.kickoff())
