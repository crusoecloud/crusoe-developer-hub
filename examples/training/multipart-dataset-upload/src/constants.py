"""Non-configurable constants. The API key comes from the environment; the
resumable state file is managed by ``src/state.py``."""

BASE_URL = "https://api.intelligence.crusoecloud.com/v1"

MiB = 1024 * 1024
GiB = 1024 * 1024 * 1024

# Crusoe Uploads API ceilings, above OpenAI's 8 GB / 64 MB.
MAX_UPLOAD_BYTES = 16 * GiB
MAX_PART_BYTES = 128 * MiB

# Interactive defaults (all overridable). Part size is a link decision, not a
# speed decision: assembly is independent of it, so smaller = finer retry units.
DEFAULT_PART_BYTES = 64 * MiB
# Parallelism is where the speedup comes from (parts upload independently).
DEFAULT_WORKERS = 8
MAX_WORKERS = 64

# File purposes the gateway supports; fine-tune leads (the common case).
PURPOSES = ["fine-tune", "batch"]
DEFAULT_PURPOSE = "fine-tune"

# Upload-session statuses. `failed` is a Crusoe extension for an upload whose
# asynchronous server-side assembly failed.
PENDING_STATUS = "pending"
COMPLETED_STATUS = "completed"
FAILED_STATUS = "failed"
EXPIRED_STATUS = "expired"
CANCELLED_STATUS = "cancelled"

# Retry tuning for every Uploads API call (parts, create, complete): a transient
# blip costs one retry, not the run. 4xx other than 429 are not retried (retrying a
# client error cannot help).
RETRY_MAX_ATTEMPTS = 5
RETRY_BACKOFF_BASE_SECONDS = 1.0
HTTP_TOO_MANY_REQUESTS = 429
HTTP_CONFLICT = 409  # complete() on a session that already left UPLOADING

# Server-side assembly polling (after `complete`, status is `pending` until
# assembly finishes and the file id is minted).
ASSEMBLY_POLL_INTERVAL_SECONDS = 5
ASSEMBLY_TIMEOUT_SECONDS = 30 * 60

# httpx timeout for a single request. Generous, since one request can carry a
# 128 MiB part over a slow link.
HTTP_TIMEOUT_SECONDS = 600

# md5 is read from disk in a single sequential pass with this block size.
MD5_READ_BLOCK_BYTES = 8 * MiB

# Tolerance when comparing a resumed file's mtime against the saved one.
MTIME_TOLERANCE_SECONDS = 1e-6
