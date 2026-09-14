#!/usr/bin/env python3
# WORKER-side hardening for large-fleet (>=64 node / 512 GPU) ZMQ OFFLINE completion.
# Applied in-place to distributed_sync_offline.py before the worker starts.
#
# ROOT CAUSE of the 64-node hang: each worker sends exactly ONE None exit-sentinel
# (run() -> self.sender.send_pyobj(None); break -> self.close()) and the sender DEALER is
# created with LINGER=0. LINGER=0 DISCARDS any not-yet-flushed message when the socket
# closes. At >=64 nodes, one worker races close() ahead of the sentinel flush, its None is
# dropped, and the head's recv_outputs (which counts Nones down to 0) blocks forever at
# 63/64 with all GPUs idle. Give the sentinel a bounded window to flush before close.
# (The head keeps its results ROUTER open until every None arrives, so a 30s linger never
# blocks the worker beyond the head's own teardown.) SNDHWM=0 additionally prevents the
# outbound giant range-result / sentinel from being dropped under the end-of-run burst.
# Idempotent + anchor-checked: safe to run every launch; a no-op if already applied.
import sys
F = "/lab-mlperf-inference/code/harness_llm/backends/vllm/zmq/distributed_sync_offline.py"
s = open(F).read()
def apply(old, new, label, required=True):
    global s
    if new.split("\n")[1].strip() in s and old not in s:
        print(f"[skip] {label}: already applied"); return
    if old not in s:
        if required: sys.exit(f"[FAIL] {label}: anchor not found")
        print(f"[skip] {label}: anchor absent"); return
    s = s.replace(old, new, 1); print(f"[ok] {label}")

# results-sender DEALER: SNDHWM=0 (before connect) + LINGER long enough to flush the final
# None exit-sentinel. Anchor is unique to the sender (the receiver uses self.receiver.*).
apply(
    "        self.sender = self.context.socket(zmq.DEALER)\n"
    "        self.sender.setsockopt(zmq.IDENTITY, self.identity)\n"
    "        self.sender.connect(f\"tcp://{self.address}:{int(self.port) + 1}\")\n"
    "        self.sender.setsockopt(zmq.LINGER, 0)",
    "        self.sender = self.context.socket(zmq.DEALER)\n"
    "        self.sender.setsockopt(zmq.IDENTITY, self.identity)\n"
    "        self.sender.setsockopt(zmq.SNDHWM, 0)\n"
    "        self.sender.connect(f\"tcp://{self.address}:{int(self.port) + 1}\")\n"
    "        # LINGER>0: the final None exit-sentinel is sent immediately before close();\n"
    "        # LINGER=0 DISCARDS it if unflushed, losing one node's None at >=64 nodes and\n"
    "        # hanging the head's recv_outputs forever (63/64, GPUs idle). Flush up to 30s.\n"
    "        self.sender.setsockopt(zmq.LINGER, 30000)",
    "worker results-sender LINGER + SNDHWM")

open(F, "w").write(s); print("OFFLINE WORKER PATCH COMPLETE")
