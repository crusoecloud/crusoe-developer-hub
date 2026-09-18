# Upload a large fine-tuning dataset in parts

Upload a multi-gigabyte training file to Crusoe with the multipart path in the OpenAI-compatible Uploads API, then hand the assembled file to a fine-tuning job. This example implements the approach from the blog post *Making Room for Bigger Fine-Tuning Datasets*: create an upload session, send the parts in parallel, complete with an ordered part list and an MD5 checksum, and poll until Crusoe returns a ready-to-use file id.

<div align="center"><img src="assets/multipart-flow.png" alt="A training file split into parts on your machine, sent in parallel into an upload session on Crusoe, completed with ordered part ids and a checksum, and assembled into one file id" width="900"></div>

You point it at a training file you already have — up to 16 GiB. It splits the file, uploads the parts concurrently with per-part retries, and saves progress after every part, so an interrupted run resumes where it left off instead of starting over. This example is about the upload API. For a full fine-tune, deploy, and evaluate walkthrough, see [PII redaction](../pii-redaction/).

## Prerequisites

- A Crusoe Inference API key, created in [Inference API keys](https://console.crusoecloud.com/security/inference-api-keys). The file id it produces can be passed straight to Serverless Fine-Tuning.
- Python 3.10 or newer and, for the notebook workflow, a Jupyter client such as JupyterLab (installed for you on first use). No GPU is involved.
- A training file to upload, for example a chat-format `.jsonl`. Nothing is downloaded for you.

## How to get started

Clone the repository and create an environment for this example:

```bash
git clone https://github.com/crusoecloud/crusoe-developer-hub.git
cd crusoe-developer-hub/examples/training/multipart-dataset-upload

python3 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run the `make` commands below from inside this activated environment. They use whatever `python3` resolves to; override with `make run PYTHON=python3.12` if you need a specific interpreter.

Provide your Crusoe key either by exporting it (`export CRUSOE_API_KEY=…`) or by leaving it unset — the run prompts for it, masked, when the variable is missing. Then run it as a notebook, the primary way to follow the steps:

```bash
make notebook           # converts run.py to a notebook and opens it in JupyterLab
```

`make notebook` generates the notebook from [`run.py`](run.py) on demand (installing `jupytext` and `jupyterlab` the first time) and removes the generated file when you close it, so no notebook is checked into the repo. Prefer the terminal? `make run` walks the same steps as a plain CLI.

Either way, the tool prompts you for the file to upload (with path autocompletion), then for the part size and the number of parallel workers, each with a sensible default. It creates the upload session, sends the parts, completes with the whole-file MD5, and polls until the assembled file id is ready. Press Ctrl-C at any point and re-run to resume; progress is saved after every landed part. The assembled file persists in your account until you delete it, and its id is printed at the end for use in a fine-tuning job. You can also see it under [Datasets](https://console.crusoecloud.com/foundry/datasets) in the Crusoe Console.

## Development

Helpers live in [`src/`](src/): `client.py` wraps the OpenAI SDK plus the REST status endpoints, `uploader.py` splits, sends, and completes, and `state.py` drives resume. The example ships a test suite:

```bash
make test               # run the pytest suite (installs deps on first use)
make lint               # ruff
```

## References

- [Uploads API reference](https://docs.crusoecloud.com/api/managed-ai/#tag/Uploads) and [Fine-tuning API reference](https://docs.crusoecloud.com/api/managed-ai/#tag/Fine-tuning)
- [Serverless Fine-Tuning documentation](https://docs.crusoecloud.com/serverless-fine-tuning/overview)
- [OpenAI Uploads API reference](https://platform.openai.com/docs/api-reference/uploads), the shape Crusoe's API matches
