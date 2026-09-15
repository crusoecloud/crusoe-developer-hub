# Upload a large fine-tuning dataset in parts

Upload a multi-gigabyte training file to Crusoe with the multipart path in the OpenAI-compatible Uploads API, then hand the assembled file to a fine-tuning job. The [notebook](multipart-upload.ipynb) is the companion to the blog post *Making Room for Bigger Fine-Tuning Datasets*. It builds a 1.2 GB chat-format JSONL file from a public dataset, uploads it in 128 MiB parts in parallel, checks which parts have landed, completes the upload with a checksum, and polls until Crusoe returns a ready-to-use file id.

<div align="center"><img src="assets/multipart-flow.png" alt="A training file split into parts on your machine, sent in parallel into an upload session on Crusoe, completed with ordered part ids and a checksum, and assembled into one file id" width="900"></div>

This example is about the API. For a full fine-tune, deploy, and evaluate walkthrough, see [PII redaction](../pii-redaction/).

## Prerequisites

- Python 3.12, a browser for JupyterLab, and internet access to Crusoe and Hugging Face. No GPU is involved.
- A Crusoe Inference API key, created in [Inference API keys](https://console.crusoecloud.com/security/inference-api-keys). The optional last step submits a fine-tuning job and needs access to Serverless Fine-Tuning.
- About 4 GB of free disk for the dataset cache and the training file, and about 1 GB of free RAM for eight parts in flight.

## Run it

From the repository root:

```bash
cd examples/training/multipart-dataset-upload
python3.12 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
python -m jupyterlab multipart-upload.ipynb
```

Paste your key into `CRUSOE_API_KEY` in the setup cell, then run the cells in order. Everything the notebook needs is defined inside it; there are no imports from elsewhere in the repository. Three constants control the transfer: `PART_SIZE` (128 MiB, Crusoe's ceiling), `WORKERS` (eight parallel uploads), and `MAX_ROWS` (unset for the full split; set an integer for a smaller trial file).

The notebook walks through:

1. **Build the file.** Download the `train_sft` split of [HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k), about 208,000 conversations, and write `data/train.jsonl`.
2. **Create the upload.** Declare the filename, purpose `fine-tune`, byte count, and MIME type. The session stays open for two hours.
3. **Add the parts.** Read each 128 MiB part by offset, send it from a thread pool, and re-send only a part that fails. Then query the parts list and the pending uploads list to see the session from the outside.
4. **Complete and poll.** Pass the ordered part ids and the file's MD5. Crusoe returns `pending` immediately and assembles in the background; poll the upload until it reports `completed` and read the file id.
5. **Use, cancel, clean up.** The fine-tuning job call sits behind `SUBMIT_FINE_TUNING_JOB = False`. A throwaway session shows cancellation, and `DELETE_UPLOADED_FILE = True` removes the assembled file.

## Expected output

With the full split: 207,865 conversations, a 1,185 MiB file, and 10 parts. The recorded run in the notebook moved the file in 38 seconds at about 32 MiB/s with eight workers, and Crusoe assembled it in about 17 seconds. Transfer time depends on your link. The parts list may still show a few parts right after completion; they are released within a minute or so.

## Costs and cleanup

The assembled file counts against your storage and persists until deleted. The optional fine-tuning job over 208,000 conversations is a real training run billed per training token; review the [Serverless Fine-Tuning documentation](https://docs.crusoecloud.com/serverless-fine-tuning/overview) before enabling it, and cancel unwanted jobs from the [fine-tuning console](https://console.crusoecloud.com/foundry/fine-tuning). Remove `data/train.jsonl`, the Hugging Face cache under `~/.cache/huggingface`, and `.venv` when no longer needed. `data/` is ignored by this folder's `.gitignore`; keep your API key out of Git.

## References

- [Uploads API reference](https://docs.crusoecloud.com/api/managed-ai/#tag/Uploads) and [Fine-tuning API reference](https://docs.crusoecloud.com/api/managed-ai/#tag/Fine-tuning)
- [Serverless Fine-Tuning documentation](https://docs.crusoecloud.com/serverless-fine-tuning/overview)
- [OpenAI Uploads API reference](https://platform.openai.com/docs/api-reference/uploads), the shape Crusoe's API matches
- [Training examples](../)
