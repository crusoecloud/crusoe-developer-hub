#!/usr/bin/env bash
# launch-mn-offline.sh - start a gpt-oss-120b v6.1 multi-node OFFLINE run (1 head + N-1
# workers) in one shot. Offline sibling of launch-mn.sh.
#
#   bash launch-mn-offline.sh <N> [HEAD_NODE] [EXPECTED_QPS]
#
#   N            total nodes (2..64). device_count = 8*N, workers = N-1.
#   HEAD_NODE    optional; default = first free schedulable MI355X node.
#   EXPECTED_QPS optional; default = round(N * 85). This sizes samples_per_query
#                (offline_expected_qps); the MN user.conf sets Offline.min_duration=600s
#                and LoadGen replays the 6396-sample pool to fill it. 85 ≈ single-node
#                Offline samples/s (115,197 tok/s ÷ ~1360 tok/sample).
#
# Prereq once: stage the offline patches to the PVC:
#   kubectl -n mlperf cp patches/harden-offline-sut.py    <pod>:/shared/patches/
#   kubectl -n mlperf cp patches/harden-worker-offline.py <pod>:/shared/patches/
#
# Watch:    kubectl -n mlperf logs -f job/mn-off-n<N>-head
# Result:   /shared/results/mn-off-n<N>/head/mlperf_log_summary.txt
# Teardown: kubectl -n mlperf delete job -l run=mn-off-n<N>
set -euo pipefail
cd "$(dirname "$0")"
NS=mlperf
SEL='crusoe.ai/accelerator=amd-mi355x-288gb-roce'
PHASE=${PHASE:-all}     # all | head | workers

N=${1:?usage: launch-mn-offline.sh <N> [HEAD_NODE] [EXPECTED_QPS]}
HEAD_NODE=${2:-}
TARGET_QPS=${3:-}
RUN="mn-off-n${N}"
OUT="/shared/results/${RUN}"
# DEDICATED_HEAD=1: pure-SUT head (no co-located worker) + N REMOTE worker nodes = N+1 nodes.
# Mirrors k8s-ds/launch-mn-deepseek.sh; device_count = 8*N either way (the N worker nodes
# supply all 8*N single-GPU vLLM replicas; the head contributes none when dedicated).
DEDICATED_HEAD=${DEDICATED_HEAD:-0}
if [ "$DEDICATED_HEAD" = "1" ]; then WORKERS=$N; LOCAL_WORKER=0; else WORKERS=$((N - 1)); LOCAL_WORKER=1; fi
DEVICE_COUNT=$((8 * N))
[ -z "$TARGET_QPS" ] && TARGET_QPS=$(python3 -c "print(round($N*85))")

# resolve the head node (a head already scheduled for this RUN wins)
existing=$(kubectl -n "$NS" get pod -l "run=${RUN},role=head" -o jsonpath='{.items[0].spec.nodeName}' 2>/dev/null || true)
if [ -n "$existing" ]; then
  HEAD_NODE="$existing"
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
      -e "s|__LOCAL_WORKER__|${LOCAL_WORKER}|g" \
      -e "s|\${OUT}|${OUT}|g" -e "s|\${WORKERS}|${WORKERS}|g" "$1"
}

echo ">> ${RUN} [phase=${PHASE}]: head=${HEAD_NODE} (${HEAD_IP})  workers=${WORKERS}  device_count=${DEVICE_COUNT}  expected_qps=${TARGET_QPS}"
if [ "$PHASE" = all ] || [ "$PHASE" = head ]; then
  render 106-mn-gptoss-offline-head.yaml | kubectl apply -f -
  until kubectl -n "$NS" get pod -l "run=${RUN},role=head" -o name 2>/dev/null | grep -q pod; do sleep 1; done
fi
if { [ "$PHASE" = all ] || [ "$PHASE" = workers ]; } && [ "$WORKERS" -gt 0 ]; then
  render 107-mn-gptoss-offline-workers.yaml | kubectl apply -f -
fi
echo ">> ${RUN} [phase=${PHASE}] done. watch: kubectl -n ${NS} logs -f job/${RUN}-head"
