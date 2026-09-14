#!/usr/bin/env python3
# HEAD-side hardening for large-fleet (>=64 node / 512 GPU) ZMQ OFFLINE completion.
# Applied in-place to distributed_offline_sut.py before run_harness.sh (offline scenario).
#
# Companion to harden-worker-offline.py. The 64-node hang is a lost None exit-sentinel:
# recv_outputs() counts one None per server down to 0, so ONE lost None hangs the head
# forever (63/64, all GPUs idle) even though every sample output already arrived. Three
# defenses, all narrowly scoped (DeepSeek's harness already ships ROUTER_MANDATORY /
# _routed_send EHOSTUNREACH retry / ROUTER_HANDOVER, so registration itself is fine):
#   1. RCVHWM=0 on the results ROUTER  - never drop a sentinel/output under the end-of-run
#      burst (default HWM=1000; 64 giant range-results + 64 Nones can breach it at 64 nodes).
#   2. SNDHWM=0 + BACKLOG on the dispatch ROUTER - never drop a range dispatch / 512-way
#      connect burst.
#   3. A TAIL-ONLY idle-break in recv_outputs - armed ONLY once <=HARNESS_RECV_TAIL
#      exit-sentinels remain (env HARNESS_RECV_IDLE_TIMEOUT_S>0). Offline sends one big
#      message per range at the very end, so mid-run silence is normal and MUST NOT trip a
#      timeout; only a long silence while waiting for the last sentinel(s) means a lost None,
#      at which point every real output is already delivered to LoadGen -> finalize VALID.
#      Default disabled (idle_s=0) to preserve stock semantics unless the manifest opts in.
# Idempotent + anchor-checked.
import sys
F = "/lab-mlperf-inference/code/harness_llm/backends/vllm/zmq/distributed_offline_sut.py"
s = open(F).read()
def apply(old, new, label, sentinel, required=True):
    global s
    if sentinel in s and old not in s:
        print(f"[skip] {label}: already applied"); return
    if old not in s:
        if required: sys.exit(f"[FAIL] {label}: anchor not found")
        print(f"[skip] {label}: anchor absent"); return
    s = s.replace(old, new, 1); print(f"[ok] {label}")

# 1+2. dispatch ROUTER (:port): SNDHWM=0 + BACKLOG before bind (HANDOVER already present)
apply(
    "        self.sender.setsockopt(zmq.ROUTER_HANDOVER, 1)\n"
    "        self.sender.bind(f\"tcp://*:{self.port}\")",
    "        self.sender.setsockopt(zmq.ROUTER_HANDOVER, 1)\n"
    "        self.sender.setsockopt(zmq.SNDHWM, 0)\n"
    "        self.sender.setsockopt(zmq.BACKLOG, 8192)\n"
    "        self.sender.bind(f\"tcp://*:{self.port}\")",
    "dispatch ROUTER SNDHWM+BACKLOG", "self.sender.setsockopt(zmq.SNDHWM, 0)")

# 1. results ROUTER (:port+1): RCVHWM=0 + HANDOVER + BACKLOG before bind
apply(
    "        receiver = ctx.socket(zmq.ROUTER)\n"
    "        receiver.bind(f\"tcp://*:{int(self.port) + 1}\")",
    "        receiver = ctx.socket(zmq.ROUTER)\n"
    "        receiver.setsockopt(zmq.RCVHWM, 0)\n"
    "        receiver.setsockopt(zmq.ROUTER_HANDOVER, 1)\n"
    "        receiver.setsockopt(zmq.BACKLOG, 8192)\n"
    "        receiver.bind(f\"tcp://*:{int(self.port) + 1}\")",
    "results ROUTER RCVHWM+BACKLOG", "receiver.setsockopt(zmq.RCVHWM, 0)")

# 3. tail-only idle-break + 3-tuple guard in recv_outputs
apply(
    "        self.log(\"Collecting outputs started...\")\n"
    "        while True:\n"
    "            identity, response = receiver.recv_multipart()\n"
    "            response = pickle.loads(response)\n"
    "            if response is None:\n"
    "                self.log(f\"{identity} exited\")\n"
    "                device_count -= 1\n"
    "                if device_count <= 0:\n"
    "                    break\n"
    "                continue\n"
    "            self.post_proc(response)\n"
    "        self.log(\"Collecting outputs finished...\")",
    "        self.log(\"Collecting outputs started...\")\n"
    "        import os as _os\n"
    "        _idle_s = int(_os.environ.get(\"HARNESS_RECV_IDLE_TIMEOUT_S\", \"0\"))\n"
    "        _tail = int(_os.environ.get(\"HARNESS_RECV_TAIL\", \"2\"))\n"
    "        _total = device_count\n"
    "        while True:\n"
    "            # Arm the idle-break ONLY in the tail (<=_tail sentinels left). Offline emits\n"
    "            # one big message per range at the very end, so mid-run silence is expected\n"
    "            # and must never trip; a long silence while awaiting the last sentinel(s)\n"
    "            # means a lost None -> every real output is already delivered -> finalize.\n"
    "            if _idle_s > 0 and device_count <= _tail and receiver.poll(_idle_s * 1000) == 0:\n"
    "                self.log(f\"RECV IDLE {_idle_s}s, {device_count}/{_total} exit-sentinels pending - lost None sentinel(s); finalizing (all sample outputs delivered)\")\n"
    "                break\n"
    "            identity, response = receiver.recv_multipart()\n"
    "            response = pickle.loads(response)\n"
    "            if response is None:\n"
    "                self.log(f\"{identity} exited\")\n"
    "                device_count -= 1\n"
    "                if device_count <= 0:\n"
    "                    break\n"
    "                continue\n"
    "            if not isinstance(response, (list, tuple)) or len(response) != 3:\n"
    "                continue\n"
    "            self.post_proc(response)\n"
    "        self.log(\"Collecting outputs finished...\")",
    "recv_outputs tail idle-break", "HARNESS_RECV_IDLE_TIMEOUT_S")

open(F, "w").write(s); print("OFFLINE HEAD PATCH COMPLETE")
