import mlflow
from dotenv import load_dotenv

load_dotenv(override=True)

version = mlflow.genai.register_prompt(
    name="support-assistant",
    template="You are a support assistant for Crusoe Cloud. Answer in one sentence: {{question}}",
    commit_message="Initial version",
)
mlflow.genai.set_prompt_alias("support-assistant", alias="production", version=version.version)

prompt = mlflow.genai.load_prompt("prompts:/support-assistant@production")
print(prompt.format(question="How do I create a workspace?"))
