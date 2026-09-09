# Foundations

Learn the concepts behind AI applications and infrastructure. Foundations explain how things work and why particular approaches help. They can include runnable notebooks and teaching exercises, such as implementing quantization or benchmarking a kernel.

Choose [examples](../examples/) when you want to complete a specific task, such as fine-tuning a PII redaction model. An example can link to a foundation for background instead of repeating the lesson.

| <div align="center">Track</div> | <div align="center">What you can find here</div> | <div align="center">Status and prerequisites</div> |
|---|---|---|
| [GPU engineering](./gpu-engineering/) | Triton kernels for vector addition, SiLU, fused SwiGLU, masking, and rotary positional embeddings, with correctness checks and benchmarks | Available. Python, `uv`, and an NVIDIA GPU; start with the track's setup instructions |
| [Post-training](./post-training/) | Two notebooks explaining quantization, LoRA, and QLoRA through PyTorch implementations and a model fine-tune | Available. Python, `uv`, and an NVIDIA GPU for the full sequence; no Crusoe API key needed |
| [AI engineering](./ai-eng/) | Planned lessons on RAG, tool calling, agents, agent loops, graph-based workflows, agent harnesses, and evaluation | Planned. Scope only; no lessons or runnable exercises yet |
| [Cloud](./cloud/) | Planned lessons on storage, customer-managed keys (CMK), Slurm, and related infrastructure concepts | Planned. Scope only; no lessons or runnable exercises yet |

Start with [vector addition](./gpu-engineering/01_vector_add/v0_basic.py) to learn a GPU kernel, or [quantization](./post-training/01-into-lora-fine-tuning/01_quantization.ipynb) to understand model memory. Follow the corresponding track README to set up its environment. Environments are local to each track, not shared across the repository.

[Back to the hub](../README.md)
