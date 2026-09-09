# Post-training Foundations

Understand quantization, LoRA, and QLoRA by implementing the core ideas in PyTorch, then using `transformers`, `peft`, and `bitsandbytes` to fine-tune a model. These notebooks teach the mechanics of post-training through runnable exercises.

## Start here

| Lesson | What it covers |
|---|---|
| [Quantization](./01-into-lora-fine-tuning/01_quantization.ipynb) | Numeric precision, quantization and dequantization, granularity, NF4, and loading a model in 4 bit |
| [LoRA and QLoRA](./01-into-lora-fine-tuning/02_lora_and_qlora.ipynb) | A LoRA layer built from scratch, its effect on a pretrained model, and a QLoRA fine-tune using standard libraries |

Work through them in order. See the [lesson README](./01-into-lora-fine-tuning/) for details and further reading.

To apply the ideas to a specific task, continue to the [PII redaction example](../../examples/training/pii-redaction/). That walkthrough prepares data, runs managed fine-tuning, deploys the model, and evaluates it. It has its own setup and credentials instructions.

## Prerequisites

- Python 3.12 and [`uv`](https://docs.astral.sh/uv/).
- An NVIDIA GPU and a working CUDA-compatible PyTorch environment for the full notebook sequence. The original walkthrough was developed on an NVIDIA L40S. The early LoRA exercises can run on CPU, but the quantization and QLoRA cells use CUDA.
- Internet access and disk space to download model weights, a dataset, and Python dependencies.
- No Crusoe API key or environment variables are required by these foundational notebooks. A Crusoe account is needed only if you choose to provision a Crusoe VM.

### Optional: use a GPU VM on Crusoe

If you need a GPU machine, sign in to the [Crusoe Console](https://console.crusoecloud.com/) and follow the [VM creation guide](https://docs.crusoecloud.com/quickstart/creating-a-vm/index.html). Choose a GPU instance and an image with NVIDIA drivers, add your SSH public key, and allow space on the boot disk for model downloads. The original guide suggested at least 50 GB of disk space.

For an Ubuntu image, connect using the instance's public IP:

```bash
ssh ubuntu@<PUBLIC_IP>
```

See [Accessing your VMs](https://docs.crusoecloud.com/compute/virtual-machines/accessing-vms/index.html) for connection details. A cloud VM and its storage can incur charges; review and delete resources you no longer need in the console after saving your work.

## Set up the track environment

Create the virtual environment in `foundations/post-training`. It is shared by the two notebooks in this track, not by every notebook in the repository.

From a new checkout:

```bash
git clone https://github.com/crusoecloud/crusoe-developer-hub.git
cd crusoe-developer-hub/foundations/post-training
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m ipykernel install --user --name fine-tuning-lora --display-name "fine-tuning-lora (.venv)"
```

If you already cloned the repository, start by changing into `foundations/post-training`, then create the environment and install the requirements.

## Run the notebooks

From `foundations/post-training`:

```bash
cd 01-into-lora-fine-tuning
../.venv/bin/jupyter lab 01_quantization.ipynb
```

Select **fine-tuning-lora (.venv)** in your notebook editor. After the first notebook, open `02_lora_and_qlora.ipynb` in the same environment.

The notebooks display quantization error, memory comparisons, LoRA parameter counts, and training results. Values depend on hardware and the run; use the explanations and assertions in each notebook to interpret them.

## Cleanup

Stop Jupyter when finished. Training writes local artifacts under `01-into-lora-fine-tuning/outputs/`; keep any results you need before removing generated files. Model and dataset downloads can remain in your Hugging Face cache. If you created a cloud VM, closing Jupyter does not remove that VM or its attached storage.

[All foundations](../README.md) | [Training examples](../../examples/training/)
