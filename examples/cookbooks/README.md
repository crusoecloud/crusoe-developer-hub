# Cookbooks

Cookbooks are short, focused recipes with a concrete result and a small number of steps. Longer task workflows belong in infrastructure, training, or inference; guided event sessions belong in workshop.

| <div align="center">Recipe</div> | <div align="center">Outcome</div> | <div align="center">Requirements</div> |
| --- | --- | --- |
| [Managed MLflow](managed-mlflow/) | Trace an LLM call and a LangChain chain, evaluate outputs with a scorer and an LLM judge, and version prompts on Crusoe Managed MLflow | uv, a Managed MLflow workspace, and a Crusoe API key |
| [NVIDIA Exemplar with Managed MLflow](managed-mlflow-exemplar/) | Run an NVIDIA exemplar-performance benchmark on Slurm or Kubernetes and stream its params, metrics and tags to Crusoe Managed MLflow with three environment variables | A Crusoe Slurm or Kubernetes cluster with NVIDIA GPUs, a Managed MLflow workspace, a Crusoe API key, and a Hugging Face token with gated model access |


[All examples](../README.md)
