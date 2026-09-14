#!/usr/bin/env python3
# WORKER-side hardening for large-fleet (>=64 node / 512 GPU) ZMQ OFFLINE registration.
# Applied in-place to distributed_sync_offline.py before launching the worker.
#
# The offline worker sends its [identity, perf] REGISTER exactly once, then blocks on a
# single recv for the ack. At 512-way fan-in a REGISTER lost in the connect burst hangs
# that worker (and thus the whole all-512 barrier) forever. Fix: re-send [identity, perf]
# every 15s until the ack is actually received. NOTE offline acks are only sent by the
# head after ALL servers register (wait_for_servers_ready), so an early worker legitimately
# waits minutes here and WILL re-send many times - that is expected; the head-side
# register_servers dedup (harden-offline-sut.py) absorbs the duplicates.
import sys
F = "/lab-mlperf-inference/code/harness_llm/backends/vllm/zmq/distributed_sync_offline.py"
s = open(F).read()

old = (
    "        self.sender.send_pyobj([self.identity, perf])\n"
    "\n"
    "        ack = self.receiver.recv_pyobj()")
new = (
    "        self.sender.send_pyobj([self.identity, perf])\n"
    "\n"
    "        while not self.receiver.poll(15000):\n"
    "            self.log(\"REGISTER ack not yet received (15s) - resending REGISTER\")\n"
    "            self.sender.send_pyobj([self.identity, perf])\n"
    "        ack = self.receiver.recv_pyobj()")

if old not in s:
    sys.exit("[FAIL] REGISTER-retry: anchor not found")
s = s.replace(old, new, 1)
open(F, "w").write(s)
print("[ok] REGISTER retry-until-acked (offline worker)")
print("OFFLINE WORKER PATCH COMPLETE")
