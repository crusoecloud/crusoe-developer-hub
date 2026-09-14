#!/usr/bin/env bash
# launch-mn.sh - start a gpt-oss-120b v6.1 multi-node run (1 head + N-1 workers) in one shot.
#
#   bash launch-mn.sh <N> [HEAD_NODE] [TARGET_QPS]
#
#   N          total nodes (2..64). device_count = 8*N, workers = N-1.
#   HEAD_NODE  optional; default = first free schedulable MI355X node.
#   TARGET_QPS optional; default = round(N * 81 * 0.9)  (81 = v6.1 single-node Server qps).
#
# Workers are ONE Indexed Job (parallelism = N-1); each pod grabs a whole node's 8 GPUs, so
# Kubernetes auto-spreads one per free node - no per-worker YAML. The head is nodeName-pinned
# so the scheduler reserves its 8 GPUs the moment its pod object exists; we apply the head and
# wait for that pod before applying workers, so workers never steal the head node.
#
# Concurrent runs (e.g. a 2/8/16/32 scaling sweep): reserve every head first, then release
# workers, so no run's workers grab another run's head:
#     for N in 2 8 16 32; do PHASE=head    bash launch-mn.sh $N; done
#     for N in 2 8 16 32; do PHASE=workers bash launch-mn.sh $N; done
#
# Watch:    kubectl -n mlperf logs -f job/mn-n<N>-head
# Result:   /shared/results/mn-n<N>/head/mlperf_log_summary.txt
# Teardown: kubectl -n mlperf delete job -l run=mn-n<N>
set -euo pipefail
cd "$(dirname "$0")"
NS=mlperf
SEL='crusoe.ai/accelerator=amd-mi355x-288gb-roce'
PHASE=${PHASE:-all}     # all | head | workers
CONFIG_NAME=${CONFIG_NAME:-server_mi355x}   # server_mi355x | offline_mi355x

N=${1:?usage: launch-mn.sh <N> [HEAD_NODE] [TARGET_QPS]}
HEAD_NODE=${2:-}
TARGET_QPS=${3:-}
RUN="mn-n${N}"
OUT="/shared/results/${RUN}"
# DEDICATED_HEAD=1: pure-SUT head (no co-located worker) + N REMOTE worker nodes = N+1 nodes.
# Mirrors k8s-ds/launch-mn-deepseek.sh; device_count = 8*N either way (the N worker nodes
# supply all 8*N single-GPU vLLM replicas; the head contributes none when dedicated).
DEDICATED_HEAD=${DEDICATED_HEAD:-0}
if [ "$DEDICATED_HEAD" = "1" ]; then WORKERS=$N; LOCAL_WORKER=0; else WORKERS=$((N - 1)); LOCAL_WORKER=1; fi
DEVICE_COUNT=$((8 * N))
[ -z "$TARGET_QPS" ] && TARGET_QPS=$(python3 -c "print(round($N*81*0.9))")

# resolve the head node (deterministic: a head already scheduled for this RUN wins, so the
# 'workers' phase reuses the same head the 'head' phase picked)
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

render() {  # sed-render a template (portable; only replaces our named vars, leaves $(...) etc.)
  sed -e "s|\${RUN}|${RUN}|g" -e "s|\${HEAD_NODE}|${HEAD_NODE}|g" -e "s|\${HEAD_IP}|${HEAD_IP}|g" \
      -e "s|\${DEVICE_COUNT}|${DEVICE_COUNT}|g" -e "s|\${TARGET_QPS}|${TARGET_QPS}|g" \
      -e "s|__LOCAL_WORKER__|${LOCAL_WORKER}|g" \
      -e "s|\${OUT}|${OUT}|g" -e "s|\${WORKERS}|${WORKERS}|g" -e "s|\${CONFIG_NAME}|${CONFIG_NAME}|g" "$1"
}

echo ">> ${RUN} [phase=${PHASE}]: head=${HEAD_NODE} (${HEAD_IP})  workers=${WORKERS}  device_count=${DEVICE_COUNT}  target_qps=${TARGET_QPS}"
if [ "$PHASE" = all ] || [ "$PHASE" = head ]; then
  render 100-mn-gptoss-head.yaml | kubectl apply -f -
  until kubectl -n "$NS" get pod -l "run=${RUN},role=head" -o name 2>/dev/null | grep -q pod; do sleep 1; done
fi
if { [ "$PHASE" = all ] || [ "$PHASE" = workers ]; } && [ "$WORKERS" -gt 0 ]; then
  render 101-mn-gptoss-workers.yaml | kubectl apply -f -
fi
echo ">> ${RUN} [phase=${PHASE}] done. watch: kubectl -n ${NS} logs -f job/${RUN}-head"
