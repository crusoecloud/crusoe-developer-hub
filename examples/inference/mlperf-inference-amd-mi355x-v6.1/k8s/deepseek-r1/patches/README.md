# DeepSeek-R1 patches and warmup overlays

The head/worker pods apply these from **`/shared/patches-ds/`** at start (stage them with
`k8s/common/02-stage-pod.yaml`). Which files each scenario needs:

| File | In this repo? | Used by | Purpose |
|---|---|---|---|
| `harden-offline-sut.py` | yes | offline (106) | ZMQ registration hardening, offline head |
| `harden-worker-offline.py` | yes | offline (107) | worker-side REGISTER retry, offline |
| `harden-registration.py` | yes | server (102/104) | ZMQ registration hardening, server head (shared with gpt-oss) |
| `harden-worker-register.py` | yes | server (103) | worker-side REGISTER retry, server |
| `enable-deepseek-warmup.py` | yes | server-warmup (102/104) | **our fix** - injects a `deepseek-r1` entry into the harness warmup sample map so every replica warms before timing; collapses p99 TTFT so Server passes |
| `server_mn_warmup.yaml` | **no - you create it** | server-warmup (102/104) | overlay config with warmup enabled (see below) |
| `user_mi355x_mn_warmup.conf` | **no - you create it** | server-warmup (102/104) | overlay `user.conf` with the warmup-run target_qps / min_duration |

## Creating the two warmup overlays

We do not re-host these two files because they are **AMD's** MI355X tuning config (AITER/HIPBLASLT
tuning-file paths, MoRI-EP, MXFP4 ASM kernels, the hand-tuned decode-batch list, the TP8/EP8/DP8 layout -
AMD's engineering). AMD publishes that config itself, and **our change to it is a single line**, so building
the overlays from AMD's published v6.1 DeepSeek submission is trivial:

```bash
A=closed/AMD/src/deepseek-r1      # path inside the published mlcommons/submissions_inference_v6.1 repo
# server overlay = AMD's server config with warmup turned ON (it ships enable_warmup: False):
sed 's/enable_warmup: *False/enable_warmup: True/' "$A/server_mi355x_mn.yaml" > server_mn_warmup.yaml
# user.conf overlay = AMD's user.conf unchanged (it already uses a long Server.min_duration so the warmed
# engines are what get timed, not the cold start):
cp "$A/user_mi355x_mn.conf" user_mi355x_mn_warmup.conf
```

Stage both to `/shared/patches-ds/` alongside the `.py` patches. That one `enable_warmup: True` flip pairs
with our shipped `enable-deepseek-warmup.py` (which injects the missing `deepseek-r1` warmup sample into the
harness) - together they are the Server-VALID fix. The prefill-delayer is already enabled in AMD's base
config. Offline runs do **not** need these overlays.
