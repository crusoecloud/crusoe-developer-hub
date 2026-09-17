# Sample from a saved fine-tuning checkpoint

Generate text from a fine-tuning checkpoint you have already trained on Crusoe Tinker. The [script](sample.py) uses the SDK's REST client to look up a named checkpoint on a training session, opens a sampling client against it, and runs one generation with the Qwen3 chat template.

This example is about the sampling surface. If you have not yet produced a checkpoint, start with a fine-tuning walkthrough first.

## Prerequisites

- A Crusoe API key with access to Serverless Fine-Tuning.
- Python 3.11 or newer with `venv` and `pip`. No GPU is involved locally; generation runs on Crusoe.
- The session ID of the training run that produced the checkpoint (`tsess_...`), the name you passed to `save_weights_for_sampler(name=...)` when the checkpoint was created, and the base model the training run was placed on (e.g. `Qwen/Qwen3.8-27B`).

## Setup

From the repository root:

```bash
cd examples/training/crusoe-tinker/sampling
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

```bash
python sample.py \
  --api-key "$CRUSOE_API_KEY" \
  --session-id tsess_65eff00a82b74e9b9e20b59915da4429 \
  --checkpoint-name step-100 \
  --base-model Qwen/Qwen3.8-27B
```

| Argument | Meaning |
|---|---|
| `--api-key` | Your Crusoe API key. |
| `--session-id` | The training session that produced the checkpoint (`tsess_...`). |
| `--checkpoint-name` | The name you passed to `save_weights_for_sampler` when the checkpoint was created. Matched against the checkpoint's `tinker_path`. |
| `--base-model` | The HuggingFace id of the base model the training run was placed on (e.g. `Qwen/Qwen3.8-27B`). Used both to place the sampler and to load the matching tokenizer. |

The script prints the decoded response for one hardcoded prompt. To try a different prompt, adjust `PROMPT` at the top of `sample.py`.


## References

- [Tinker SDK on PyPI](https://pypi.org/project/tinker/)
- [crusoe-tinker on PyPI](https://pypi.org/project/crusoe-tinker/)
