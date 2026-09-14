#!/usr/bin/env python3
# HEAD-side hardening for large-fleet (>=64 node / 512 GPU) ZMQ registration.
# Applied in-place to distributed_server_sut.py before run_harness.sh.
#
# 512-way fan-in exposed two problems: (1) re-warmed workers reconnect with a duplicate ZMQ
# identity and, without ROUTER_HANDOVER, the ROUTER silently drops their frames; (2) some
# REGISTERs are lost outright under the connection burst (default RCVHWM/backlog/io-threads).
# register_servers() had no dedup and no timeout, so any loss stalled it forever (~494/512).
# Fixes: ROUTER_HANDOVER + unbounded HWM + large accept backlog + more IO threads on BOTH
# ROUTERs, and a dedup/poll-timeout register loop. Pairs with the worker REGISTER-retry patch.
import sys
F = "/lab-mlperf-inference/code/harness_llm/backends/vllm/zmq/distributed_server_sut.py"
s = open(F).read()
def apply(old, new, label, required=True):
    global s
    if old not in s:
        if required: sys.exit(f"[FAIL] {label}: anchor not found")
        print(f"[skip] {label}: anchor absent"); return
    s = s.replace(old, new, 1); print(f"[ok] {label}")

# more ZMQ IO threads on both contexts (512 peers)
apply("        self.context = zmq.Context()\n",
      "        self.context = zmq.Context(io_threads=8)\n", "dispatch ctx io_threads")
apply("        ctx = zmq.Context()\n        receiver = ctx.socket(zmq.ROUTER)",
      "        ctx = zmq.Context(io_threads=8)\n        receiver = ctx.socket(zmq.ROUTER)", "receiver ctx io_threads")

# dispatch ROUTER (:port): HANDOVER + unbounded HWM + large backlog (before bind)
apply("        self.sender.setsockopt(zmq.ROUTER_MANDATORY, 1)\n        self.sender.bind(",
      "        self.sender.setsockopt(zmq.ROUTER_MANDATORY, 1)\n"
      "        self.sender.setsockopt(zmq.ROUTER_HANDOVER, 1)\n"
      "        self.sender.setsockopt(zmq.SNDHWM, 0)\n"
      "        self.sender.setsockopt(zmq.RCVHWM, 0)\n"
      "        self.sender.setsockopt(zmq.BACKLOG, 8192)\n"
      "        self.sender.bind(", "dispatch ROUTER tuning")

# results/registration ROUTER (:port+1): HANDOVER + unbounded HWM + large backlog (before bind)
apply("        receiver = ctx.socket(zmq.ROUTER)\n        receiver.bind(",
      "        receiver = ctx.socket(zmq.ROUTER)\n"
      "        receiver.setsockopt(zmq.ROUTER_HANDOVER, 1)\n"
      "        receiver.setsockopt(zmq.RCVHWM, 0)\n"
      "        receiver.setsockopt(zmq.SNDHWM, 0)\n"
      "        receiver.setsockopt(zmq.BACKLOG, 8192)\n"
      "        receiver.bind(", "receiver ROUTER tuning")

# dedup + poll-timeout register loop (idempotent to worker re-sends)
old_reg = (
    "    def register_servers(self, socket):\n"
    "        for i in range(0, self.instance_count):\n"
    '            self.log(f"{i=}/{self.instance_count} Wait for server register message")\n'
    "            identity, data = socket.recv_multipart()\n"
    "            data = pickle.loads(data)\n"
    '            self.log(f"{i=}/{self.instance_count} Got new server with {identity=}")\n'
    "            assert identity == data\n"
    "            assert identity not in self.servers\n"
    "            self.servers[i] = QueryInfo(identity=identity)\n"
    "            self.server_mapping[identity] = i")
new_reg = (
    "    def register_servers(self, socket):\n"
    "        seen = {}\n"
    "        while len(seen) < self.instance_count:\n"
    "            if socket.poll(60000) == 0:\n"
    '                self.log(f"REG WAIT registered={len(seen)}/{self.instance_count} (60s idle)")\n'
    "                continue\n"
    "            identity, data = socket.recv_multipart()\n"
    "            data = pickle.loads(data)\n"
    "            assert identity == data\n"
    "            if identity in self.server_mapping:\n"
    '                continue\n'
    "            i = len(seen)\n"
    "            seen[identity] = i\n"
    "            self.servers[i] = QueryInfo(identity=identity)\n"
    "            self.server_mapping[identity] = i\n"
    '            self.log(f"registered={len(seen)}/{self.instance_count} Got new server {identity!r}")')
apply(old_reg, new_reg, "register_servers dedup+timeout")

# recv_outputs: ignore stray late-REGISTER messages (a worker's retry that lands on the
# results socket just after registration completes) - they are a bare identity, not a
# [sample_id, token_ids] response, and would crash post_proc with "int not iterable".
apply(
    "            self.post_proc(response, self.server_mapping[identity])",
    "            if not isinstance(response, (list, tuple)) or len(response) != 2:\n"
    "                continue\n"
    "            if identity not in self.server_mapping:\n"
    "                continue\n"
    "            self.post_proc(response, self.server_mapping[identity])",
    "recv_outputs skip stray REGISTER")

open(F, "w").write(s); print("HEAD PATCH COMPLETE")
