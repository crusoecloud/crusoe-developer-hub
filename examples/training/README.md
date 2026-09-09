# Training examples

Prepare data, train or fine-tune a model, and evaluate the result. Organize examples around the task and expected outcome rather than the underlying Crusoe service.

| <div align="center">Example</div> | <div align="center">Outcome</div> | <div align="center">Requirements</div> |
| --- | --- | --- |
| [PII redaction](pii-redaction/) | Prepare synthetic data, fine-tune a model, deploy it, and evaluate redaction against a baseline | Python environment, Crusoe Inference API key, and access to the training and deployment services used in the notebook; no local GPU |

The PII notebook contains executable code, helper functions, and a credentials template. Follow its README to configure it and inspect the steps that launch jobs or deployments before running them. Training, inference, and deployed resources can incur charges; the README explains cleanup and validation limits.

For the mechanics, start with [quantization and LoRA foundations](../../foundations/post-training/). For distributed training infrastructure, see the [TorchTitan solution](https://github.com/crusoecloud/solutions-library/tree/main/torchtitan-llama3_1-kubernetes-pytorchjob).

[All examples](../README.md)
