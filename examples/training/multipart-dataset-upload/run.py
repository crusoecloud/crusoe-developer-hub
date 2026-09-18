# %% [markdown]
# # Multipart dataset upload - interactive multipart upload of a large training file
#
# Walks the four-step Uploads API end to end:
#
# 1. Create an upload session and declare the file's size.
# 2. Add parts: split the file into chunks and upload them in parallel.
# 3. Complete the upload with the ordered part ids and an md5 checksum.
# 4. Poll until Crusoe assembles the parts into one verified file.
#
# The run is resumable: state is saved after every part, so you can Ctrl-C and
# re-run to pick up where you left off. Set `CRUSOE_API_KEY` in the environment,
# or leave it unset and the run prompts for it (masked, works in the terminal and
# in Jupyter).
#
# - **CLI**: `python3 run.py`
# - **Jupyter**: convert this percent-formatted file to a notebook with jupytext.

# %%
from __future__ import annotations

import os

from src import local_file, parameters, runtime
from src.client import UploadsClient
from src.state import State
from src.uploader import Uploader


def require_api_key() -> str:
    # Prefer the environment (exported shells, CI); otherwise prompt for it, so no
    # secret file is needed and nothing is echoed to the screen or a notebook cell.
    key = os.environ.get("CRUSOE_API_KEY", "").strip()
    if not key:
        key = runtime.password("Crusoe API key").strip()
    if not key:
        runtime.abort("error: no Crusoe API key provided (set CRUSOE_API_KEY or enter it when prompted).")
    return key


# %% [markdown]
# ## Authentication and API client
#
# Read the Crusoe API key (from `CRUSOE_API_KEY`, or a masked prompt if it is unset)
# and build the Uploads API client. The client wraps the OpenAI SDK for the canonical
# upload calls and httpx for Crusoe's REST status endpoints. A quick list call up front
# confirms the key and network before the upload begins.

# %%
api_key = require_api_key()
uploader = Uploader(UploadsClient(api_key))  # owns the client for the rest of the run
uploader.check_connection()

# %% [markdown]
# ## Load and resolve saved state
#
# Load any saved state file (a pure read: no prompts, no network), then resume_or_start()
# runs the interactive resume/reset decision - continue an interrupted upload, start
# fresh, or adopt an already-finished prior run.

# %%
state = State.load(api_key)
state.resume_or_start(uploader.client)

# %% [markdown]
# ## Re-show a finished prior run?
#
# If resume_or_start() adopted an already-finished prior run (the user declined a new
# upload), print its result and exit. Otherwise fall through to the upload flow.

# %%
if state.session.completed:
    print(f"\n  this upload already completed: file {state.session.file_id}")
    state.print_summary()
    uploader.close()
    runtime.abort("  nothing to upload; the previous upload already completed.", status=0)

print("  resuming the previous upload" if state.resuming else "  starting a new upload")

# %% [markdown]
# ## Step 1: choose a file
#
# Pick the file with an autocompleting path prompt. Then hold one file descriptor
# for the whole run and start the md5 on a background thread, so it overlaps the
# setup and the part uploads.

# %%
print("\n--- Step 1: choose a file ---")
file = local_file.LocalFile.choose(state) if state.should_pick_file() else local_file.LocalFile.reopen(state)
file.start_md5(state)

# %% [markdown]
# ## Step 2: upload parameters
#
# Each prompt has a sane default (press Enter to accept). On a resumed run the saved
# values are reused and nothing is asked.
#
# - **Purpose** — what the file is for: `fine-tune` (default) or `batch`. It is recorded
#   on the resulting file and determines where it can be used.
# - **Part size (MiB)** — the file is split into chunks of this size. Default 64, range
#   1-128. Bigger parts mean fewer requests; smaller parts retry and resume at a finer
#   grain over a flaky link.
# - **Workers** — how many parts upload in parallel. Default 8, range 1-64. Higher can be
#   faster until it saturates your uplink.
#
# The file itself can be up to 16 GiB; the part count is derived from its size and the
# part size.

# %%
print("\n--- Step 2: upload parameters ---")
if state.should_choose_params():
    parameters.choose_params(state)
else:
    state.print_selected_params()

# %% [markdown]
# ## Step 3: create the upload session

# %%
print("\n--- Step 3: create upload session ---")
uploader.create_upload(state)

# %% [markdown]
# ## Step 4: upload the parts in parallel
#
# Only the parts that never landed are sent on a resume.

# %%
print("\n--- Step 4: upload parts ---")
uploader.upload_parts(state, file)

# %% [markdown]
# ## Step 5: complete and assemble
#
# Join the background md5, verify the file did not change on disk, complete with
# the ordered part ids + md5, then poll until the file id is ready.

# %%
print("\n--- Step 5: complete + assemble ---")
file.finish_md5(state)
file.verify_unchanged()
uploader.complete_upload(state)

# %% [markdown]
# ## Summary

# %%
state.print_summary()
file.close()
uploader.close()
