# Fine Tuning on Crusoe with Crusoe Mill

**Crusoe Mill** is Crusoe's Tinker-compatible API for fine-tuning and reinforcement learning.
It speaks the same client surface and wire protocol as Tinker, so a loop written against one
runs against the other.

Write the training loop yourself, in Python, on your own machine. Crusoe runs the GPU work
behind an API: forward and backward passes, optimizer steps, and generation. There is no
cluster to provision and no local GPU involved.

This is the opposite trade from [Serverless Fine-Tuning](../../../foundations/post-training/),
where you hand over a dataset and get a checkpoint back. Here you keep the loop: your data,
your reward function, your evaluation, your stopping rule. Reach for it when the recipe is the
thing you are working on: reinforcement learning, a custom reward, an unusual curriculum.

| Example | Description |
| --- | --- |
| [Hello World](01_hello_crusoe_mill.ipynb) | A very simple getting started example. Sessions and the client hierarchy, waiting out provisioning, sampling, reading a response, and chat templates. |
| [SFT](02_sft_banking77.ipynb) | Supervised fine-tuning on Banking77: building a training client, masking loss to the answer tokens, the `forward_backward` / `optim_step` loop, publishing an adapter, and measuring accuracy before and after. |
| [GSM8K](03_grpo_gsm8k.ipynb) | Reinforcement learning from a verifiable reward, with GRPO: group-relative advantages, sampling a fresh adapter every step, and why the advantage is the mask. |
| [Sampling a checkpoint](04_sample_checkpoint.ipynb) | Listing the checkpoints saved across your sessions with a session that declares no models, then serving one of them next to its base model in a new session. |

## Supported models

Crusoe Mill currently supports these base models, for both training and sampling. Pass the
name exactly as written wherever the SDK takes a `base_model`.

| Model | Used in |
| --- | --- |
| `Qwen/Qwen3.8-27B` | 01 |
| `Qwen/Qwen3.5-9B` | 02, and 04 when serving 02's adapter |
| `Qwen/Qwen3.5-9B-Base` | 03 |

## Prerequisites

- An **Intelligence API key**, created in the [Crusoe Console](https://console.crusoecloud.com/).
  Crusoe Mill is in limited preview, so talk to your Crusoe contact if your key does not
  yet have access.
- **Python 3.12 or 3.13** and a Jupyter client such as VS Code or JupyterLab.
- No GPU, and no CUDA. Everything local here is CPU-only.

## How to get started

Clone the repository and open the first notebook:

```bash
git clone https://github.com/crusoecloud/crusoe-developer-hub.git
cd crusoe-developer-hub/examples/training/crusoe-mill

jupyter lab 01_hello_crusoe_mill.ipynb      # or open the folder in VS Code
```

### The environment

Everything the notebooks need is pinned in `requirements.txt`. The recommended setup builds a
virtual environment and registers it as a kernel, so the notebooks run against a known-good
set of versions:

```bash
python3 -m venv .venv
. .venv/bin/activate                          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m ipykernel install --user --name crusoe-mill --display-name "Crusoe Mill"
```

Then pick the **Crusoe Mill** kernel in Jupyter or VS Code.

If you would rather not build one, the notebooks' first cell runs
`pip install -r requirements.txt` into whichever kernel you already have, so they also work
from a plain kernel. Either way it is the same list. The pins matter, because an unpinned
`transformers` can change tokenisation and an unpinned `datasets` can change the data.

`crusoe-mill` is in limited preview and not on a public package index, so its wheel ships in
`sdk/` and `requirements.txt` points at it by relative path.
No GPU or CUDA is needed locally; `transformers` is used for its tokenisers only, so PyTorch
is not installed and you can ignore its "PyTorch was not found" notice.

### The data

The notebooks build their datasets from the public upstream sources via `prepare_data.py`,
each pinned to a commit. No dataset is stored in this repository. The data cell of each notebook
runs it, and each file is checked against a recorded md5, so a moved upstream pin fails loudly
instead of quietly changing a score. Files already in a data directory are never
overwritten, so to train on your own data, point `DATA` in the notebook's data cell at a
directory of your own files in the same shape. To build or verify the example data yourself:

```bash
python prepare_data.py              # notebook 03's GSM8K data
python prepare_data.py --banking77  # notebook 02's Banking77 splits
python prepare_data.py --check      # verify what is on disk, build nothing
```

The notebooks prompt for your API key if `CRUSOE_API_KEY` is not set, so nothing secret is
saved into the notebook file.

**Run them from this directory.** `requirements.txt`, `prepare_data.py`, `sdk/` and `data/`
are all referenced by relative path, so the notebook's working directory has to be its own
folder. In VS Code, if the first cell cannot find `sdk/`, set *Jupyter: Notebook File Root*
to `${fileDirname}`.

If an import of `crusoe_mill` still fails after the install cell ran, the install and the
kernel are in different environments. This tells you which interpreter the notebook is actually
using:

```python
import sys; print(sys.executable)
```

## Cost and cleanup

A session holds GPUs from the moment you construct the `ServiceClient` until it ends, so
**the notebooks end with `client.close()` and you should run that cell**. Until the session
goes, its model stays resident and the next run you start may sit waiting on capacity your own
last run is still holding. If your script exits without closing, the cluster spins down on its
own about 15 minutes later. While a session is still alive you can re-attach to it by passing
its id as `existing_session_id` in `ServiceConfig`, which skips provisioning.

Expect constructing the `ServiceClient` to be slow: it provisions dedicated GPUs for your
session and blocks until they are ready. That usually takes several minutes, and longer when
capacity is busy. The SDK prints progress once a minute while it waits, and
`wait_for_ready=False` returns immediately if you would rather take the wait later.

## References

- [Crusoe Mill documentation](https://docs.crusoecloud.com/) on the Crusoe docs site
- [Serverless Fine-Tuning](https://docs.crusoecloud.com/serverless-fine-tuning/overview), for
  supervised fine-tuning without writing a loop
- [Quantization and LoRA foundations](../../../foundations/post-training/), for the mechanics
  underneath LoRA training

[All training examples](../README.md)
