"""Example usage of the Crusoe MLflow deployment plugin.

Prerequisites:
    pip install mlflow-crusoe
    export CRUSOE_API_KEY="your-api-key"
"""

import mlflow.deployments


def main():
    # Get the Crusoe deployment client
    client = mlflow.deployments.get_deploy_client("crusoe")

    # Create a deployment
    client.create_deployment(
        name="my-llama",
        model_uri="meta-llama/Llama-3.3-70B-Instruct",
        config={
            "temperature": 0.7,
            "max_tokens": 2048,
        },
    )
    print("Created deployment 'my-llama'")

    # List all deployments
    deployments = client.list_deployments()
    print(f"Active deployments: {[d['name'] for d in deployments]}")

    # Run inference with a simple prompt
    result = client.predict("my-llama", inputs={"prompt": "Explain GPUs in one sentence."})
    print(f"Response: {result['choices'][0]['message']['content']}")

    # Run inference with full chat messages
    result = client.predict(
        "my-llama",
        inputs={
            "messages": [
                {"role": "system", "content": "You are a helpful coding assistant."},
                {"role": "user", "content": "Write a Python hello world."},
            ]
        },
    )
    print(f"Chat response: {result['choices'][0]['message']['content']}")

    # Update the deployment
    client.update_deployment("my-llama", model_uri="deepseek-ai/DeepSeek-V3")
    print("Updated deployment to DeepSeek-V3")

    # Clean up
    client.delete_deployment("my-llama")
    print("Deleted deployment 'my-llama'")


if __name__ == "__main__":
    main()
