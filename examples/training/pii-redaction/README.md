# Fine-tune a PII redaction model

Fine-tune Qwen3 8B to find and mask personally identifiable information in synthetic financial documents, deploy the checkpoint on Crusoe, and compare its entity extraction scores with a Llama 3.3 70B baseline. The [notebook](finetune-deploy-inference.ipynb) covers dataset preparation, Serverless Fine-Tuning, a dedicated Self-Serve Deployment, inference, evaluation, and cleanup.

The expected result is a model that returns a JSON object containing `redacted_text` and an `entities` list, plus a measured comparison on held-out documents. This is a task-focused example. For quantization and LoRA concepts, start with [post-training foundations](../../../foundations/post-training/).

![Workflow from synthetic financial documents through fine-tuning and deployment to redacted output](assets/pii-redaction-pipeline.svg)

## Prerequisites

- Python 3.12 with `venv` and `pip`, a browser for JupyterLab, and internet access to Crusoe and Hugging Face. Training and serving run on Crusoe; a local GPU is not required.
- A Crusoe account with access to Serverless Fine-Tuning, Self-Serve Deployments, and Serverless Inference. The notebook lists the model registry before resolving `Qwen/Qwen3-8B`; the evaluation also calls `meta-llama/Llama-3.3-70B-Instruct`.
- Access to the [synthetic PII dataset](https://huggingface.co/datasets/gretelai/synthetic_pii_finance_multilingual). Configure a Hugging Face token if authentication is required.
- Familiarity with running notebook cells and basic Python. You create the dedicated deployment manually in the Crusoe Console during the walkthrough.

## Infrastructure, credentials, and configuration

This example creates uploaded training and validation files, a fine-tuning job with model checkpoints, and a dedicated deployment. It also uses a shared inference endpoint for the evaluation baseline.

Copy [`.env.example`](.env.example) to `.env` and replace its placeholder values:

| Variable | Purpose |
|---|---|
| `CRUSOE_API_KEY` | Required Crusoe Inference API key. Create one in [Inference API keys](https://console.crusoecloud.com/security/inference-api-keys). |
| `HF_TOKEN` | Hugging Face token with access to the dataset, when required. Replace the placeholder with your token, or remove the assignment for anonymous access. |

The notebook loads `.env` with `python-dotenv`. It sends management requests to `https://api.intelligence.crusoecloud.com/v1` and inference requests to `https://api.inference.crusoecloud.com/v1`. These URLs are notebook constants.

After creating the deployment in Step 8, replace the existing example value of `DEPLOYMENT_ALIAS` with your deployment's alias before running inference. The alias is configured in the notebook, not in `.env`.

## Setup and execution

From the repository root:

```bash
cd examples/training/pii-redaction
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
# Edit .env with your credentials before starting the notebook.
python -m jupyterlab finetune-deploy-inference.ipynb
```

Use the Python kernel from this environment and keep the notebook's working directory at `examples/training/pii-redaction` so its `utils` import and relative data paths resolve. This example has its own [requirements file](requirements.txt); the foundational notebooks' GPU dependencies are not needed.

Run the cells in order and pause at the deployment instructions:

1. **Configure and inspect available models.** Step 4 initializes the API client and resolves the base model's registry ID.
2. **Prepare and validate the dataset.** Step 5 runs [`data/prepare_dataset.py`](data/prepare_dataset.py) with the notebook's Python interpreter. It filters short English documents and writes `data/train.jsonl` (2,500 rows), `data/validation.jsonl` (150), and `data/test.jsonl` (100). Check that local validation reports all checks passed before uploading the training and validation files.
3. **Fine-tune and inspect checkpoints.** Steps 6 and 7 submit a two-epoch LoRA job, poll its status, display loss curves, and list checkpoints. Submitting the job uses paid resources. The optional adapter download cell has `DOWNLOAD_ADAPTER = True` by default; set it to `False` to skip saving a ZIP under `outputs/`.
4. **Create a deployment.** Follow Step 8 in the [Deployments console](https://console.crusoecloud.com/foundry/deployments), choose a checkpoint and optimization profile, and set the replica count to one. Wait until it is ready, then update `DEPLOYMENT_ALIAS` in the notebook.
5. **Run inference and evaluation.** Steps 9 and 10 demonstrate regular and streaming requests, then compare both models on up to `EVAL_N = 100` held-out documents. This evaluation makes up to 200 completion requests; the final inspection cell makes two additional requests. Lower `EVAL_N` for a smaller comparison.
6. **Clean up.** Complete Step 11 and the instructions below before leaving the walkthrough.

[`utils.py`](utils.py) contains the shared task prompt, dataset validation, prediction parsing, entity scoring, and plotting helpers. Dataset preparation and inference use the same system prompt.

## Expected output

- Three local JSONL files with the row counts above, followed by validation messages.
- Uploaded file IDs, a fine-tuning job ID, job status and events, loss plots, checkpoint details, and a registered fine-tuned model ID.
- Redacted demonstration text with placeholders such as `[NAME]` and `[EMAIL]`, plus the extracted entity types and values.
- An evaluation table and chart reporting detection precision, recall and F1, typed F1, and unparseable response counts. Scores depend on the training run and model responses; the example does not assert a target score.
- An adapter ZIP in `outputs/` if the download cell is enabled.

## Costs and cleanup

Fine-tuning, the dedicated deployment, and baseline inference can incur charges. The notebook describes training charges per processed token, dedicated deployment charges per GPU-hour while deployed, and shared inference charges per token. Review the current [fine-tuning](https://docs.crusoecloud.com/serverless-fine-tuning/overview) and [deployment](https://docs.crusoecloud.com/self-serve-deployments/overview) documentation and the Console before creating resources. Closing Jupyter does not delete a deployment or cancel submitted work.

1. Delete the dedicated deployment in the [Deployments console](https://console.crusoecloud.com/foundry/deployments), using its three-dot menu and **Delete**. The notebook does not perform this deletion through code.
2. In Step 11, set `CLEAN_UP_FILES = True` and run the cell to delete the uploaded training and validation files. Its default is `False`. The cell needs the `train_file` and `val_file` objects from this notebook session; retain their printed IDs if you plan to clean up later.
3. Fine-tuned checkpoints remain in the model registry after this cleanup. Local JSONL files, downloaded adapters, the Hugging Face download cache, `.env`, and `.venv` also remain; remove them when no longer needed. Keep credentials and generated data out of Git.

## Validation

Restructuring checks on September 8, 2026 verified local links and assets, helper-script and notebook-cell syntax, and dependency import coverage. The move preserved executable notebook cells and recorded outputs; it changed the setup path and gave the example its own environment instructions.

Dependencies were not installed and the notebook was not rerun against Crusoe during these checks. Live model availability, account access, training, deployment, and cleanup still need verification in your environment. Recorded notebook outputs come from an earlier run and are not a promise of identical results.

## References

- [Serverless Fine-Tuning documentation](https://docs.crusoecloud.com/serverless-fine-tuning/overview) and [API reference](https://docs.crusoecloud.com/api/managed-ai/#tag/Fine-tuning)
- [Self-Serve Deployments documentation](https://docs.crusoecloud.com/self-serve-deployments/overview)
- [Serverless Fine-Tuning launch blog](https://www.crusoe.ai/resources/blog/crusoe-introduces-serverless-fine-tuning) and [Self-Serve Deployments launch blog](https://www.crusoe.ai/resources/blog/crusoe-self-serve-deployments)
- [Qwen/Qwen3-8B model](https://huggingface.co/Qwen/Qwen3-8B)
- [Training examples](../)
