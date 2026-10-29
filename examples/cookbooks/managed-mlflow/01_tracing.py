import os

import mlflow
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)
mlflow.openai.autolog()

client = OpenAI(api_key=os.environ["CRUSOE_API_KEY"], base_url=os.environ["CRUSOE_API_BASE"])

response = client.chat.completions.create(
    model=os.environ["CRUSOE_MODEL"],
    messages=[
        {"role": "system", "content": "You are a helpful assistant. Answer in one sentence."},
        {"role": "user", "content": "Who is Robinson Crusoe?"},
    ],
)
print(response.choices[0].message.content)
