# Upload a large fine-tuning dataset in parts

Upload a multi-gigabyte training file to Crusoe with the multipart path in the OpenAI-compatible Uploads API, then use the assembled file in a fine-tuning job. The [notebook](multipart-upload.ipynb) is the companion to the blog post *Making Room for Bigger Fine-Tuning Datasets*. It builds a 1.2 GB chat-format JSONL file from a public dataset, uploads it in 128 MiB parts in parallel, checks which parts have landed, completes the upload with a checksum, and polls until Crusoe has assembled the parts into a ready-to-use file id.

The expected result is a file in your Crusoe account that any fine-tuning job can reference by id, produced without hitting the per-request size limit of a single file upload. This example is about the API. For a full fine-tune, deploy, and evaluate walkthrough, see [PII redaction](../pii-redaction/).

## Prerequisites

- Python 3.12 with `venv` and `pip`, a browser for JupyterLab, and internet access to Crusoe and Hugging Face. Nothing here needs a GPU.
- A Crusoe account with an Inference API key. The optional last step submits a fine-tuning job and needs access to Serverless Fine-Tuning.
- About 4 GB of free disk for the dataset download cache and the generated training file, and roughly 1 GB of free RAM for eight parts in flight.
- The `openai` package at version 1.37 or newer, which is where the Uploads API arrived. The requirements file pins a compatible range.

## Credentials and configuration

Copy [`.env.example`](.env.example) to `.env` and replace the placeholder value:

| <div align="center">Variable</div> | <div align="center">Purpose</div> |
| --- | --- |
| `CRUSOE_API_KEY` | Required Crusoe Inference API key. Create one in [Inference API keys](https://console.crusoecloud.com/security/inference-api-keys). |
| `HF_TOKEN` | Optional Hugging Face token. The default dataset is public; set this only if you switch to a gated dataset. |

The notebook loads `.env` with `python-dotenv` and sends every request to `https://api.intelligence.crusoecloud.com/v1`. Three constants at the top of the notebook control the transfer: `PART_SIZE` (128 MiB, Crusoe's ceiling), `WORKERS` (eight parallel part uploads), and `MAX_ROWS` (unset, so the full split is used; set an integer for a smaller trial file).

## Setup and execution

From the repository root:

```bash
cd examples/training/multipart-dataset-upload
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
# Edit .env with your key before starting the notebook.
python -m jupyterlab multipart-upload.ipynb
```

Use the Python kernel from this environment and keep the working directory at `examples/training/multipart-dataset-upload` so the `data/` path resolves. The example is self-contained: every helper is defined in the notebook and there are no imports from elsewhere in the repository.

Run the cells in order:

1. **Setup.** Load the key, create the OpenAI client against Crusoe, and set the part size and worker count.
2. **Build a large training file.** Download the `train_sft` split of [HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) and write `data/train.jsonl`, one `{"messages": [...]}` object per line. The split has about 208,000 conversations and the file is about 1.2 GB.
3. **Create the upload.** Declare the filename, purpose `fine-tune`, byte count, and MIME type. The session is open for two hours.
4. **Add the parts in parallel.** Read each 128 MiB part by offset, send it from a thread pool, and re-send only a part that fails. Compute the file's MD5 for the completion call. Then query the parts list and the pending uploads list to see the session from the outside.
5. **Complete the upload.** Pass the ordered part ids and the checksum. Crusoe returns `pending` immediately and assembles in the background.
6. **Poll until the file is ready.** Retrieve the upload until its status leaves `pending`, read the file id, and confirm it with the regular files endpoint.
7. **Use the file in a fine-tuning job.** The job submission is behind `SUBMIT_FINE_TUNING_JOB = False`. Set it to `True` only when you intend to train on the full dataset.
8. **Cancel and clean up.** Open and cancel a throwaway session to see self-cleaning in action, then optionally delete the assembled file with `DELETE_UPLOADED_FILE = True`.

## Expected output

- 207,865 conversations, a file of 1,185 MiB, and 10 parts when the full split is used.
- An upload id with status `pending` and a two hour session.
- One line per part as it lands, followed by the elapsed time and throughput. Transfer time depends on your link; our run moved the file in 56 seconds at about 21 MiB/s with eight workers. The parts list should then report 10 of 10 parts received and one pending session.
- `pending` after the completion call, then `completed` from the poll loop with a file id and the same byte count you declared. Assembly took about 20 seconds in our run. The parts list may still show a few parts right after completion; they are released within a minute or so.
- A skipped fine-tuning cell and a cancelled throwaway session.

## Limits and lifetimes

| <div align="center">Limit or lifetime</div> | <div align="center">Crusoe</div> |
| --- | --- |
| Max size per upload | 16 GiB |
| Max part size | 128 MiB |
| Upload session lifetime | 2 hours |
| Part lifetime | Tied to the session; cleaned up on completion, cancel, or expiry |
| Resulting file expiry | Persists by default; optional `expires_after` from 1 hour to 30 days |

If a session expires mid-upload, its parts expire with it; start a new upload. Pick the part size for your link rather than for the server: large parts on a fast, stable link, smaller parts on a slow or flaky one, since the retry unit is a single part.

## Costs and cleanup

The upload itself and the assembled file are not GPU resources, but the file counts against your storage and persists until deleted. Submitting the optional fine-tuning job over 208,000 conversations is a real training run billed per training token; review the [Serverless Fine-Tuning documentation](https://docs.crusoecloud.com/serverless-fine-tuning/overview) before enabling it.

1. Set `DELETE_UPLOADED_FILE = True` in the last cell to delete the assembled file, or keep it for later jobs. Retain the printed file id if you plan to clean up from another session.
2. Any fine-tuning job you submitted keeps running after you close the notebook; cancel it from the [fine-tuning console](https://console.crusoecloud.com/foundry/fine-tuning) if you do not want it.
3. Remove `data/train.jsonl`, the Hugging Face cache under `~/.cache/huggingface`, `.env`, and `.venv` when no longer needed. Keep credentials and generated data out of Git; `data/` is ignored by this folder's `.gitignore`.

## References

- [Uploads API reference](https://docs.crusoecloud.com/api/managed-ai/#tag/Uploads) and [Fine-tuning API reference](https://docs.crusoecloud.com/api/managed-ai/#tag/Fine-tuning)
- [Serverless Fine-Tuning documentation](https://docs.crusoecloud.com/serverless-fine-tuning/overview)
- [OpenAI Uploads API reference](https://platform.openai.com/docs/api-reference/uploads), for the shape Crusoe's API matches
- [Crusoe Foundry console](https://console.crusoecloud.com/foundry)
- [Training examples](../)
