import os

import mlflow
from dotenv import load_dotenv
from mlflow.genai.scorers import Correctness, scorer
from openai import OpenAI

load_dotenv(override=True)

mlflow.openai.autolog()
client = OpenAI(api_key=os.environ["CRUSOE_API_KEY"], base_url=os.environ["CRUSOE_API_BASE"])


def app(question):
    """The application under test: replace this with your own entry point."""
    response = client.chat.completions.create(
        model=os.environ["CRUSOE_MODEL"],
        messages=[
            {"role": "system", "content": "Answer in one sentence."},
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content


data = [
    {"inputs": {"question": "What is the capital of France?"}, "expectations": {"expected_response": "Paris"}},
    {"inputs": {"question": "What is the capital of Japan?"}, "expectations": {"expected_response": "Tokyo"}},
    {"inputs": {"question": "What is the capital of Kenya?"}, "expectations": {"expected_response": "Nairobi"}},
    {"inputs": {"question": "What is the capital of India?"}, "expectations": {"expected_response": "New Delhi"}},
    {"inputs": {"question": "What is the capital of Australia?"}, "expectations": {"expected_response": "Sydney"}},
]


@scorer
def contains_expected(outputs, expectations):
    return expectations["expected_response"].lower() in outputs.lower()


judge = Correctness(model=f"crusoe:/{os.environ['CRUSOE_MODEL']}")
results = mlflow.genai.evaluate(data=data, predict_fn=app, scorers=[contains_expected, judge])
print(results.metrics)
