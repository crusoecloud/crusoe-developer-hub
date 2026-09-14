#!/usr/bin/env bash
# launch-mn-deepseek.sh - start a DeepSeek-R1 v6.1 multi-node run (1 head + N-1 workers) in one shot.
# Adapted from the gpt-oss launch-mn scripts; DeepSeek uses the same ZMQ head/worker mechanism.
#
#   bash launch-mn-deepseek.sh <N> [HEAD_NODE] [TARGET_QPS]
#     SCENARIO=offline (default) | server     # picks 106/107 (offline) or 100/101 (server) manifests
#
#   N          total nodes (2..64). device_count = 8*N, workers = N-1. One 8-GPU SGLang replica/node.
#   TARGET_QPS default = AMD guidance: 0.9 * single_node_qps * N  (leave 10% headroom so the
#     single-head keeps Server latency VALID). offline 0.9*13.2=11.88/node ; server 0.9*10.9=9.81/node.
#     e.g. offline 2 nodes -> round(2*11.88)=24 ; server 2 nodes -> round(2*9.81)=20.
#
# Prereq once (only needed for large fleets, >=64 nodes): stage the ZMQ hardening patches to the PVC
# at /shared/patches-ds/ AFTER verifying their anchors match THIS image's harness_llm copy:
#   kubectl -n mlperf cp patches/harden-offline-sut.py    <pod>:/shared/patches-ds/
#   kubectl -n mlperf cp patches/harden-worker-offline.py <pod>:/shared/patches-ds/
#
# Watch:    kubectl -n mlperf logs -f job/ds-mn-<scenario>-n<N>-head
# Teardown: kubectl -n mlperf delete job -l run=ds-mn-<scenario>-n<N>
set -euo pipefail
cd "$(dirname "$0")"
NS=mlperf
SEL='crusoe.ai/accelerator=amd-mi355x-288gb-roce'
PHASE=${PHASE:-all}                 # all | head | workers
SCENARIO=${SCENARIO:-offline}       # offline | server
MODE=${MODE:-perf}                  # perf | accuracy  (accuracy -> test_mode=accuracy, distinct -acc run)
WARMUP=${WARMUP:-0}                 # server only: 1 -> enable AMD dsr_code6_1 warmup (102/103 manifests)
TEST06=${TEST06:-0}                 # server+warmup only: 1 -> TEST06 compliance head (104 + 103 workers)
if [ "$MODE" = accuracy ]; then TEST_MODE=accuracy; SUFFIX="-acc"; else TEST_MODE=performance; SUFFIX=""; fi
[ "$WARMUP" = "1" ] && SUFFIX="-wu${SUFFIX}"
[ "$TEST06" = "1" ] && SUFFIX="-test06${SUFFIX}"

N=${1:?usage: launch-mn-deepseek.sh <N> [HEAD_NODE] [TARGET_QPS]}
HEAD_NODE=${2:-}
TARGET_QPS=${3:-}
RUN="ds-mn-${SCENARIO}${SUFFIX}-n${N}"
OUT="/shared/results/${RUN}"
# DEDICATED_HEAD=1: run a pure-SUT head (no co-located worker) + N REMOTE workers = N+1 nodes.
# Needed at large scale (>=64): the head's heavy result post-proc starves a co-located worker's
# SGLang scheduler -> its watchdog aborts -> lost range result -> head hangs 63/64. A dedicated
# head keeps all N replicas on their own uncontended nodes. device_count = 8*N either way.
DEDICATED_HEAD=${DEDICATED_HEAD:-0}
if [ "$DEDICATED_HEAD" = "1" ]; then WORKERS=$N; LOCAL_WORKER=0; else WORKERS=$((N - 1)); LOCAL_WORKER=1; fi
DEVICE_COUNT=$((8 * N))
if [ -z "$TARGET_QPS" ]; then
  if [ "$SCENARIO" = server ]; then TARGET_QPS=$(python3 -c "print(round($N*9.81))"); else TARGET_QPS=$(python3 -c "print(round($N*11.88))"); fi
fi
if [ "$SCENARIO" = server ]; then
  if [ "$TEST06" = "1" ]; then HEAD_TMPL=104-ds-mn-server-test06-head.yaml; WORK_TMPL=103-ds-mn-server-warmup-workers.yaml
  elif [ "$WARMUP" = "1" ]; then HEAD_TMPL=102-ds-mn-server-warmup-head.yaml; WORK_TMPL=103-ds-mn-server-warmup-workers.yaml
  else HEAD_TMPL=100-ds-mn-server-head.yaml; WORK_TMPL=101-ds-mn-server-workers.yaml; fi
else
  if [ "$TEST06" = "1" ]; then HEAD_TMPL=108-ds-mn-offline-test06-head.yaml; WORK_TMPL=109-ds-mn-offline-test06-workers.yaml
  else HEAD_TMPL=106-ds-mn-offline-head.yaml; WORK_TMPL=107-ds-mn-offline-workers.yaml; fi
fi

# resolve head node (a head already scheduled for this RUN wins)
existing=$(kubectl -n "$NS" get pod -l "run=${RUN},role=head" -o jsonpath='{.items[0].spec.nodeName}' 2>/dev/null || true)
if [ -n "$existing" ]; then HEAD_NODE="$existing"
elif [ -z "$HEAD_NODE" ]; then
  used=$(kubectl -n "$NS" get pods -o jsonpath='{range .items[*]}{.spec.nodeName}{"\n"}{end}' 2>/dev/null | sort -u)
  for n in $(kubectl get nodes -l "$SEL" -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}'); do
    [ "$(kubectl get node "$n" -o jsonpath='{.spec.unschedulable}')" = "true" ] && continue
    echo "$used" | grep -qx "$n" && continue
    HEAD_NODE="$n"; break
  done
fi
[ -z "$HEAD_NODE" ] && { echo "no free head node found"; exit 1; }
HEAD_IP=$(kubectl get node "$HEAD_NODE" -o jsonpath='{.status.addresses[?(@.type=="InternalIP")].address}')

render() {
  sed -e "s|\${RUN}|${RUN}|g" -e "s|\${HEAD_NODE}|${HEAD_NODE}|g" -e "s|\${HEAD_IP}|${HEAD_IP}|g" \
      -e "s|\${DEVICE_COUNT}|${DEVICE_COUNT}|g" -e "s|\${TARGET_QPS}|${TARGET_QPS}|g" \
      -e "s|\${OUT}|${OUT}|g" -e "s|\${WORKERS}|${WORKERS}|g" -e "s|\${TEST_MODE}|${TEST_MODE}|g" \
      -e "s|__LOCAL_WORKER__|${LOCAL_WORKER}|g" "$1"
}

echo ">> ${RUN} [phase=${PHASE}]: head=${HEAD_NODE} (${HEAD_IP}) workers=${WORKERS} device_count=${DEVICE_COUNT} target_qps=${TARGET_QPS} dedicated_head=${DEDICATED_HEAD}"
if [ "$PHASE" = all ] || [ "$PHASE" = head ]; then
  render "$HEAD_TMPL" | kubectl apply -f -
  until kubectl -n "$NS" get pod -l "run=${RUN},role=head" -o name 2>/dev/null | grep -q pod; do sleep 1; done
fi
if { [ "$PHASE" = all ] || [ "$PHASE" = workers ]; } && [ "$WORKERS" -gt 0 ]; then
  render "$WORK_TMPL" | kubectl apply -f -
fi
echo ">> ${RUN} [phase=${PHASE}] done. watch: kubectl -n ${NS} logs -f job/${RUN}-head"
