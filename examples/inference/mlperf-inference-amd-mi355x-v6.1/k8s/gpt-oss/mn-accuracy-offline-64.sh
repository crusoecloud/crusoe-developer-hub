#!/usr/bin/env bash
# 64-node (512 GPU) gpt-oss v6.1 OFFLINE ACCURACY run over the ZMQ distributed offline SUT,
# then score exact_match. Head runs run_harness --config-name offline_mi355x --backend zmq
# test_mode=accuracy (device_count=512); workers = 63-pod offline Indexed Job (107 template).
# Uses the offline patches (harden-offline-sut.py + harden-worker-offline.py). Eval
# (check_gptoss_accuracy_scores.sh) is baked into the head after the run.
set -euo pipefail
NS=mlperf; SEL='crusoe.ai/accelerator=amd-mi355x-288gb-roce'
# DEVICE_COUNT overridable: accuracy is scale-independent (AccuracyOnly processes each sample
# once regardless of GPU count), so we over-provision 64 worker nodes (512 GPUs) but can set a
# lower barrier (e.g. 504) to tolerate a few dead vLLM replicas without hanging the 512-hard SUT.
RUN=mn-off-n64-acc; OUT=/shared/results/$RUN; DEVICE_COUNT=${DEVICE_COUNT:-512}
# DEDICATED_HEAD=1: pure-SUT head + 64 remote worker pods (65 nodes); no co-located worker.
DEDICATED_HEAD=${DEDICATED_HEAD:-0}
if [ "$DEDICATED_HEAD" = "1" ]; then WORKERS=64; LOCAL_WORKER=0; else WORKERS=63; LOCAL_WORKER=1; fi
K="$(cd "$(dirname "$0")" && pwd)"

# pick a free head node
used=$(kubectl -n "$NS" get pods -o jsonpath='{range .items[*]}{.spec.nodeName}{"\n"}{end}' 2>/dev/null | sort -u)
HEAD_NODE=""
for n in $(kubectl get nodes -l "$SEL" -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}'); do
  [ "$(kubectl get node "$n" -o jsonpath='{.spec.unschedulable}')" = "true" ] && continue
  echo "$used" | grep -qx "$n" && continue
  HEAD_NODE="$n"; break
done
HEAD_IP=$(kubectl get node "$HEAD_NODE" -o jsonpath='{.status.addresses[?(@.type=="InternalIP")].address}')
echo ">> $RUN: head=$HEAD_NODE ($HEAD_IP) device_count=$DEVICE_COUNT workers=$WORKERS"

# ---- offline accuracy head job ----
cat <<YAML | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata: { name: ${RUN}-head, namespace: mlperf, labels: { app: mlperf-mn-gptoss-offline, role: head, run: ${RUN} } }
spec:
  backoffLimit: 0
  ttlSecondsAfterFinished: 21600
  template:
    metadata: { labels: { app: mlperf-mn-gptoss-offline, role: head, run: ${RUN} } }
    spec:
      restartPolicy: Never
      hostNetwork: true
      hostIPC: true
      imagePullSecrets: [{ name: regcred }]
      nodeName: ${HEAD_NODE}
      containers:
        - name: head
          image: <your-registry>/<your-project>/gptoss_vllm_v0.22.0
          securityContext: { privileged: true }
          env:
            - { name: MODEL_OVERRIDE_PATH, value: "/model/gpt-oss-120b/fp4_quantized" }
          command: ["/bin/bash","-lc"]
          args:
            - |
              set -x
              ulimit -Sn 524288 2>/dev/null || true
              echo "fd limit now: \$(ulimit -Sn)"
              HEAD_IP=${HEAD_IP}; OUT=${OUT}; M=/shared/models/gpt-oss-120b-b5c939d
              rm -rf /model/gpt-oss-120b; mkdir -p /model/gpt-oss-120b /data "\$OUT/head"
              rm -f "\$OUT/head"/mlperf_log_*.txt "\$OUT/head"/mlperf_log_*.json "\$OUT/head-sut.log" "\$OUT"/worker-*.log 2>/dev/null || true
              for f in "\$M"/*; do ln -sf "\$f" /model/gpt-oss-120b/; done
              ln -sfn "\$M" /model/gpt-oss-120b/fp4_quantized
              ln -sfn /shared/datasets/gpt-oss-120b/raw/gpt-oss-dataset /data/gptoss-dataset
              ln -sfn /shared/v61-cache/gptoss-mxfp4 /root/.cache 2>/dev/null || true
              [ -f /shared/patches/harden-offline-sut.py ] && python3 /shared/patches/harden-offline-sut.py || echo "WARN offline head patch missing"
              cd /lab-mlperf-inference/code
              # local offline worker for this node's 8 GPUs (offline worker ignores dataset_path).
              # DEDICATED head (LOCAL_WORKER=0): skip it - all ${DEVICE_COUNT} GPUs are remote workers.
              if [ "${LOCAL_WORKER}" = "1" ]; then
                python3 harness_llm/backends/vllm/zmq/distributed_sync_offline.py --config-path gpt-oss-120b/ --config-name offline_mi355x \
                  node_id=\$(hostname) headnode_address=\${HEAD_IP}:12345 > "\$OUT/worker-head.log" 2>&1 &
                sleep 5
              else
                echo "DEDICATED head: no co-located worker"
              fi
              echo "===== ${RUN}: run_harness OFFLINE ACCURACY (device_count=${DEVICE_COUNT}) ====="
              bash run_harness.sh --config-path gpt-oss-120b/ --config-name offline_mi355x --backend zmq \
                test_mode=accuracy port=12345 harness_config.device_count=${DEVICE_COUNT} \
                harness_config.dataset_path=/data/gptoss-dataset/acc/acc_eval_ref.parquet \
                harness_config.output_log_dir="\$OUT/head" 2>&1 | tee "\$OUT/head-sut.log" | tail -30
              echo "===== ACCURACY EVAL (check_gptoss_accuracy_scores.sh) ====="
              MODEL_OVERRIDE_PATH=/model/gpt-oss-120b/fp4_quantized bash scripts/check_gptoss_accuracy_scores.sh \
                "\$OUT/head/mlperf_log_accuracy.json" 2>&1 | tee "\$OUT/acc-eval.log" | tail -30
              echo "===== EXACT_MATCH ====="; grep -iE "exact_match|FINAL|Accuracy|score" "\$OUT/acc-eval.log" 2>/dev/null | tail
          resources: { requests: { amd.com/gpu: 8, cpu: "180", memory: 1800Gi }, limits: { amd.com/gpu: 8 } }
          volumeMounts: [{ name: shared, mountPath: /shared }, { name: dshm, mountPath: /dev/shm }]
      volumes:
        - { name: shared, persistentVolumeClaim: { claimName: mlperf-shared } }
        - { name: dshm, emptyDir: { medium: Memory, sizeLimit: 512Gi } }
YAML

# wait for head pod to exist (reserves its GPUs), then release workers
until kubectl -n "$NS" get pod -l "run=${RUN},role=head" -o name 2>/dev/null | grep -q pod; do sleep 1; done

# ---- offline workers: render 107 template ----
sed -e "s|\${RUN}|${RUN}|g" -e "s|\${HEAD_IP}|${HEAD_IP}|g" -e "s|\${WORKERS}|${WORKERS}|g" -e "s|\${OUT}|${OUT}|g" \
    "$K/107-mn-gptoss-offline-workers.yaml" | kubectl apply -f -
echo ">> launched ${RUN} offline accuracy run. head=${HEAD_NODE}"
