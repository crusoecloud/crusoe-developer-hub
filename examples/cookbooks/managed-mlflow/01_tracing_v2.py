import os

import mlflow
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)
mlflow.openai.autolog()

client = OpenAI(api_key=os.environ["CRUSOE_API_KEY"], base_url=os.environ["CRUSOE_API_BASE"])


def ask(question):
    response = client.chat.completions.create(
        model=os.environ["CRUSOE_MODEL"],
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Answer in one sentence."},
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content


@mlflow.trace
def handle_user_request(user_id, session_id, message):
    mlflow.update_current_trace(
        metadata={"mlflow.trace.user": user_id, "mlflow.trace.session": session_id},
        tags={"environment": os.getenv("ENVIRONMENT", "dev"), "service_version": os.getenv("SERVICE_VERSION", "1.0.0")},
    )
    try:
        response = ask(message)
        response_len = str(len(response)) if response is not None else "0"
        mlflow.update_current_trace(tags={"response_length": response_len, "processing_successful": "true"})
        return response
    except Exception as e:
        mlflow.update_current_trace(tags={"error": "true", "error_type": type(e).__name__, "error_message": str(e)})
        raise


print(handle_user_request("user-1", "session-1", "Who is Robinson Crusoe?"))
