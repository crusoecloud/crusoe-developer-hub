# Inference examples

Call a model or build a model-serving workflow: chat, streaming, tool calling, structured output, inference evaluation, and reproducible benchmarking. Organize contributions by task, with the relevant API or serving infrastructure documented inside each example.

| <div align="center">Example</div> | <div align="center">Outcome</div> | <div align="center">Requirements</div> |
| --- | --- | --- |
| [MLPerf Inference v6.1 on AMD MI355X](mlperf-inference-amd-mi355x-v6.1/) | Reproduce Crusoe's MLPerf Inference v6.1 submissions for gpt-oss-120b and DeepSeek-R1 at up to 512-GPU (64-node) scale on Crusoe Managed Kubernetes, and validate the results with the official submission checker | Crusoe Managed Kubernetes cluster with MI355X nodes, `kubectl` and `helm`, a container registry, and a Hugging Face token for gated models |

The MLPerf example is a complete, self-contained directory with its own Apache-2.0 license. It scales down to a single node for validation before the full run. Read its README before applying anything: the manifests create a namespace, a shared persistent volume, and Jobs that occupy entire MI355X nodes, and those resources incur charges until cleaned up.

Additional entry points live in Integrations or the Solutions Library:

| <div align="center">Task</div> | <div align="center">Entry point</div> | <div align="center">What it contains</div> |
| --- | --- | --- |
| Set up an API key and make a first request | [Crusoe inference quickstart](https://docs.crusoecloud.com/serverless-inference/index.html) | Official instructions for API key setup and a chat completion |
| Call Crusoe through a framework | [LangChain](../../integrations/langchain/) | Python package, usage examples, and tests |
| Route requests through a local gateway | [LiteLLM](../../integrations/litellm/) | Docker Compose setup and request examples |
| Deploy a serving stack | [KServe solution](https://github.com/crusoecloud/solutions-library/tree/main/crusoe-kserve-example) | Serving configuration and deployment tooling in the Solutions Library |

Use the selected README's prerequisites and credentials, and choose a model available to your account. API calls and deployed infrastructure can incur charges. These entries retain their primary home because they focus on a tool connection or a broader deployment solution.

[All examples](../README.md)
