# Methodology - running MLPerf Inference Kubernetes-native at 512 GPU

## Why Kubernetes

MLPerf Inference's reference harnesses (`mlcommons/inference`) are built to run as a Docker container on a
single machine - bare metal or one VM. That model is a poor fit for a 512-GPU run across 64 nodes, and it
isn't how most production inference on Crusoe is operated. Rather than hand-orchestrate 64 SSH sessions, we
re-expressed the benchmark as **Kubernetes-native manifests**, which gives us three things for free:

- **Scaling** - the same worker `Job` runs at `parallelism=1` or `parallelism=64` by changing one argument.
- **Observability** - pod logs, events, and resource metrics through the standard Kubernetes surface.
- **Fault tolerance** - a crashed replica is a restarted pod, not a dead run; stragglers are visible and evictable.

Everything below runs on a Crusoe Managed Kubernetes (CMK) cluster of MI355X nodes.

## The distributed SUT (system-under-test)

MLPerf's LoadGen must see a single SUT. At 512 GPUs we build that SUT as a **ZMQ-connected head + workers**:

```
                 ┌────────────────────────┐
   LoadGen  ───► │  HEAD  (1 pod)          │   binds :12345 (dispatch)  +  :12346 (results)
                 │  dispatch + LoadGen     │
                 └───────────┬────────────┘
                             │  tokenized prompts / generated tokens (ZMQ over RoCE/Ethernet)
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                     ▼
   WORKER pod 0         WORKER pod 1   ...    WORKER pod 63
   8 MI355X GPUs        8 MI355X GPUs         8 MI355X GPUs
   per node             per node              per node
```

Parallelism *within* each 8-GPU node is model-specific (see below); the cross-node picture above is the same
either way.

- **Dedicated head:** one pod does *only* dispatch + LoadGen (no local model), so the head is never a compute
  bottleneck. Each of the 64 worker pods owns a node's 8 GPUs; how those 8 GPUs are used is model-specific:
    - **gpt-oss-120b:** 8 independent single-GPU replicas (tensor-parallel size 1) -> 512 replicas. The 120B
      model fits in one GPU at MXFP4, so no sharding is needed.
    - **DeepSeek-R1:** one replica sharded across all 8 GPUs (TP8 + EP8 + DP-attention 8) -> 64 replicas. The
      671B MoE is far too large for a single 288 GB GPU, so it must shard - but only *within* the node, over XGMI.
  Either way `device_count = 8*N` counts GPUs (512 at 64 nodes).
- **No cross-node collectives.** The only inter-node traffic is tokenized input/output **streams** over ZMQ on
  the RoCE Ethernet fabric. There is no NVLink/InfiniBand all-reduce - inference at this scale is
  embarrassingly parallel, so commodity Ethernet is sufficient. This is a key result: you do not need an
  exotic collective fabric to serve at 512-GPU scale.
- **Launchers** (`launch-mn*.sh`) pick a free MI355X node for the head, read its internal IP, `sed`-template
  `${HEAD_NODE}`/`${HEAD_IP}`/`${DEVICE_COUNT}`/`${TARGET_QPS}`/… into the head + worker manifests, and apply them.

## Scenarios

- **Offline** measures raw batch throughput with no latency constraint - `device_count = 8*N`, a high
  `target_qps` so LoadGen issues enough samples to fill the 20-minute minimum-duration window.
- **Server** measures throughput under a p99 latency SLA (TTFT + TPOT). `target_qps` is tuned to the highest
  value that still passes the SLA - driving it too high creates a scheduling backlog and blows p99. For
  gpt-oss the sustainable point is `target_qps 4000`.

## The two engineering problems 512-GPU scale exposed (and our fixes)

1. **ZMQ registration under a 512-worker burst.** When 512 replicas connect to the head within seconds, the
   stock SUT drops or mis-orders registrations and the run never reaches `device_count`. Our `harden-*.py`
   patches add `ROUTER_HANDOVER`, larger accept backlogs, a dedup + poll-timeout registration loop on the head,
   and a worker-side REGISTER retry - so all 512 attach deterministically.
2. **DeepSeek Server cold-start.** The stock harness's warmup path self-disabled for DeepSeek-R1, so the first
   server queries paid full engine cold-start and the p99 TTFT (~130 s) blew the SLA → INVALID. Adding a
   `deepseek-r1` entry to the warmup encoded-samples map (`enable-deepseek-warmup.py`) warms every replica
   before timing starts, collapsing p99 TTFT to ~1.7 s and making the Server run VALID.

Both live in `k8s/<model>/patches/` and are applied to the shared PVC so head/worker pods pick them up at start.

## Accuracy and compliance

Accuracy is **scenario-independent** (each sample is issued once), so one AccuracyOnly run per model validates
all scenarios. Compliance audit runs (gpt-oss: TEST07 + TEST09; DeepSeek/LLMs: TEST06) re-run the workload with
an `audit.config` and a verification script. The packaging manifests assemble the results into the MLCommons
directory layout, truncate the accuracy logs, and run the official `submission_checker` - a submission is only
real when it ends with `SUMMARY: submission looks OK`.

## Reproducing at your own scale

Start small to de-risk the pipeline, then scale: `N=1` (8 GPUs) → `N=8` (64) → `N=64` (512). Accuracy and
compliance are scale-independent, so you can confirm *correctness* at `N=1` and only spend the full fleet on
the throughput numbers. See the top-level [README](../README.md) for the exact per-model commands.
