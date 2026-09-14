#!/usr/bin/env python3
# HEAD-side hardening for large-fleet (>=64 node / 512 GPU) ZMQ OFFLINE registration.
# Applied in-place to distributed_offline_sut.py before run_harness.sh (offline scenario).
#
# The offline SUT's register_servers() (a) had no dedup and asserted the identity was
# unseen, and (b) acks are only sent AFTER all instance_count servers register
# (wait_for_servers_ready). With the worker REGISTER-retry patch, every early-arriving
# worker resends its [identity, perf] every ~15s for the whole warmup window, so the head
# MUST dedup or the assert fires. We also add a poll-timeout register loop (visible
# progress, never hangs silently), ROUTER_HANDOVER + a large accept backlog on both
# ROUTERs (512-way connect burst), and a skip-guard in recv_outputs so a stray late
# REGISTER ([identity, perf], len 2) that lands on the results socket after registration
# does not crash post_proc (which unpacks a 3-tuple). Conservative on purpose: no
# unbounded HWM / high io_threads (those correlated with a native heap-corruption crash
# in the server SUT at serving-start).
import sys
F = "/lab-mlperf-inference/code/harness_llm/backends/vllm/zmq/distributed_offline_sut.py"
s = open(F).read()
def apply(old, new, label, required=True):
    global s
    if old not in s:
        if required: sys.exit(f"[FAIL] {label}: anchor not found")
        print(f"[skip] {label}: anchor absent"); return
    s = s.replace(old, new, 1); print(f"[ok] {label}")

# dispatch ROUTER (:port): HANDOVER + large backlog (before bind)
apply(
    "        self.sender.setsockopt(zmq.ROUTER_MANDATORY, 1)\n"
    "        self.sender.bind(f\"tcp://*:{self.port}\")",
    "        self.sender.setsockopt(zmq.ROUTER_MANDATORY, 1)\n"
    "        self.sender.setsockopt(zmq.ROUTER_HANDOVER, 1)\n"
    "        self.sender.setsockopt(zmq.BACKLOG, 8192)\n"
    "        self.sender.bind(f\"tcp://*:{self.port}\")",
    "dispatch ROUTER tuning")

# results/registration ROUTER (:port+1): HANDOVER + large backlog (before bind)
apply(
    "        receiver = ctx.socket(zmq.ROUTER)\n"
    "        receiver.bind(f\"tcp://*:{int(self.port) + 1}\")",
    "        receiver = ctx.socket(zmq.ROUTER)\n"
    "        receiver.setsockopt(zmq.ROUTER_HANDOVER, 1)\n"
    "        receiver.setsockopt(zmq.BACKLOG, 8192)\n"
    "        receiver.bind(f\"tcp://*:{int(self.port) + 1}\")",
    "receiver ROUTER tuning")

# dedup + poll-timeout register loop (idempotent to worker re-sends of [identity, perf])
old_reg = (
    "    def register_servers(self, socket):\n"
    "        for i in range(0, self.instance_count):\n"
    "            self.log(f\"{i=}/{self.instance_count} Wait for server register message\")\n"
    "            identity, data = socket.recv_multipart()\n"
    "            data, perf = pickle.loads(data)\n"
    "            self.log(f\"{i=}/{self.instance_count} Got new server with {data=} {perf=}\")\n"
    "            assert identity == data\n"
    "            assert identity not in self.server_mapping\n"
    "            self.servers[i] = ServerInfo(identity, perf)\n"
    "            self.server_mapping[identity] = i")
new_reg = (
    "    def register_servers(self, socket):\n"
    "        while len(self.server_mapping) < self.instance_count:\n"
    "            if socket.poll(60000) == 0:\n"
    "                self.log(f\"REG WAIT registered={len(self.server_mapping)}/{self.instance_count} (60s idle)\")\n"
    "                continue\n"
    "            identity, data = socket.recv_multipart()\n"
    "            data, perf = pickle.loads(data)\n"
    "            assert identity == data\n"
    "            if identity in self.server_mapping:\n"
    "                continue\n"
    "            i = len(self.server_mapping)\n"
    "            self.servers[i] = ServerInfo(identity, perf)\n"
    "            self.server_mapping[identity] = i\n"
    "            self.log(f\"registered={len(self.server_mapping)}/{self.instance_count} Got new server {identity!r} perf={perf}\")")
apply(old_reg, new_reg, "register_servers dedup+timeout")

# recv_outputs: skip a stray late REGISTER ([identity, perf], len 2) that lands on the
# results socket after registration - post_proc expects a 3-tuple (start, end, tokens).
apply(
    "                continue\n"
    "            self.post_proc(response)",
    "                continue\n"
    "            if not isinstance(response, (list, tuple)) or len(response) != 3:\n"
    "                continue\n"
    "            self.post_proc(response)",
    "recv_outputs skip stray REGISTER")

open(F, "w").write(s); print("OFFLINE HEAD PATCH COMPLETE")
