#!/usr/bin/env python3
# WORKER-side hardening for large-fleet (512 GPU) ZMQ registration.
# Applied in-place to distributed_async_server.py before the worker starts.
#
# The worker sends its REGISTER (identity) once on the results DEALER, then blocks waiting
# for the head's ack. At 512-way fan-in a REGISTER can be lost -> the head never counts that
# worker and never acks it -> both sides hang. Fix: keep re-sending REGISTER every 2s until
# the ack arrives (main thread sets _ack_received). The head dedups repeats, so retries are
# safe, and a lost REGISTER simply gets resent. Re-sends stop the instant the ack lands
# (which only happens after the head has registered all N), so no stray identity ever reaches
# the results-collection loop.
import sys
F = "/lab-mlperf-inference/code/harness_llm/backends/vllm/zmq/distributed_async_server.py"
s = open(F).read()
def apply(old, new, label):
    global s
    if old not in s: sys.exit(f"[FAIL] {label}: anchor not found")
    s = s.replace(old, new, 1); print(f"[ok] {label}")

# 1) init the ack flag just before the sender thread starts
apply(
    "        self.sender_thread = threading.Thread(target=self.sender_loop, daemon=True)\n"
    "        self.sender_thread.start()",
    "        self._ack_received = False\n"
    "        self.sender_thread = threading.Thread(target=self.sender_loop, daemon=True)\n"
    "        self.sender_thread.start()",
    "init _ack_received")

# 2) set the flag once the head's ack arrives
apply(
    "        ack = self.receiver.recv_pyobj()\n"
    '        self.log(f"Got {ack=}")',
    "        ack = self.receiver.recv_pyobj()\n"
    "        self._ack_received = True\n"
    '        self.log(f"Got {ack=}")',
    "set _ack_received on ack")

# 3) re-send REGISTER until acked (between the initial send and the send_queue loop)
apply(
    '        self.log("Sender loop started...")\n'
    "        while True:\n"
    "            data = self.send_queue.get()",
    '        self.log("Sender loop started...")\n'
    "        import time as _t\n"
    "        while not getattr(self, '_ack_received', False):\n"
    "            _t.sleep(2)\n"
    "            if getattr(self, '_ack_received', False):\n"
    "                break\n"
    "            try:\n"
    "                sender.send_pyobj(self.identity)\n"
    "            except Exception:\n"
    "                pass\n"
    "        while True:\n"
    "            data = self.send_queue.get()",
    "REGISTER retry-until-acked")

open(F, "w").write(s); print("WORKER PATCH COMPLETE")
