#!/usr/bin/env bash
# 64-node (512 GPU) gpt-oss v6.1 COMPLIANCE over the ZMQ distributed offline SUT, dedicated head.
#   bash mn-compliance-64.sh <TEST07|TEST09>
#
# TEST07 = accuracy-in-perf over the 990-sample gpqa compliance set, threshold 60.698.
# TEST09 = mean output-token length in [1150.38, 1406.02] over the 6396-sample perf set.
#
# Each test is one fleet launch (head + 64 offline workers); the offline distributed SUT serves
# exactly one run_harness cycle, so TEST07 and TEST09 are separate runs. Matches the DeepSeek
# 64-node TEST06 precedent (compliance at full submitted scale). Verification output lands in
# $OUT/compliance-out/<TEST>/gpt-oss-120b - Phase F copies it under each scenario's compliance dir.
set -euo pipefail
NS=mlperf; SEL='crusoe.ai/accelerator=amd-mi355x-288gb-roce'
K="$(cd "$(dirname "$0")" && pwd)"
TEST=${1:?usage: mn-compliance-64.sh <TEST07|TEST09>}
case "$TEST" in
  TEST07) DS=acc_eval_compliance_gpqa.parquet; ACCDS=acc_eval_compliance_gpqa.parquet; NSAMP=990;;
  TEST09) DS=perf_eval_ref.parquet;            ACCDS=acc_eval_ref.parquet;            NSAMP=6396;;
  *) echo "unknown test $TEST"; exit 1;;
esac
# DEVICE_COUNT overridable: TEST07/TEST09 verify harness BEHAVIOR (accuracy-in-perf / output-len),
# which is scale-independent - over-provision 64 worker nodes but allow a lower barrier (e.g. 496)
# to tolerate a few dead vLLM replicas without hanging the hard SUT barrier.
RUN=mn-comp-$(echo "$TEST" | tr A-Z a-z)-n64; OUT=/shared/results/$RUN; DEVICE_COUNT=${DEVICE_COUNT:-512}; WORKERS=64

# pick a free head node
used=$(kubectl -n "$NS" get pods -o jsonpath='{range .items[*]}{.spec.nodeName}{"\n"}{end}' 2>/dev/null | sort -u)
HEAD_NODE=""
for n in $(kubectl get nodes -l "$SEL" -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}'); do
  [ "$(kubectl get node "$n" -o jsonpath='{.spec.unschedulable}')" = "true" ] && continue
  echo "$used" | grep -qx "$n" && continue
  HEAD_NODE="$n"; break
done
HEAD_IP=$(kubectl get node "$HEAD_NODE" -o jsonpath='{.status.addresses[?(@.type=="InternalIP")].address}')
echo ">> $RUN [$TEST]: head=$HEAD_NODE ($HEAD_IP) device_count=$DEVICE_COUNT workers=$WORKERS samples=$NSAMP"

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
              HEAD_IP=${HEAD_IP}; OUT=${OUT}; M=/shared/models/gpt-oss-120b-b5c939d
              DSROOT=/shared/datasets/gpt-oss-120b/raw/gpt-oss-dataset
              T=${TEST}; TDIR="\$OUT/head"; VOUT="\$OUT/compliance-out"
              rm -rf /model/gpt-oss-120b; mkdir -p /model/gpt-oss-120b /data/gpt-oss-120b "\$TDIR" "\$VOUT"
              rm -f "\$TDIR"/mlperf_log_*.txt "\$TDIR"/mlperf_log_*.json "\$OUT/head-sut.log" "\$OUT"/worker-*.log 2>/dev/null || true
              for f in "\$M"/*; do ln -sf "\$f" /model/gpt-oss-120b/; done
              ln -sfn "\$M" /model/gpt-oss-120b/fp4_quantized
              ln -sf \$DSROOT/acc/acc_eval_ref.parquet             /data/gpt-oss-120b/acc_eval_ref.parquet
              ln -sf \$DSROOT/acc/acc_eval_compliance_gpqa.parquet /data/gpt-oss-120b/acc_eval_compliance_gpqa.parquet
              ln -sf \$DSROOT/perf/perf_eval_ref.parquet           /data/gpt-oss-120b/perf_eval_ref.parquet
              ln -sfn /shared/v61-cache/gptoss-mxfp4 /root/.cache 2>/dev/null || true
              [ -f /shared/patches/harden-offline-sut.py ] && python3 /shared/patches/harden-offline-sut.py || echo "WARN offline head patch missing"
              # fetch compliance tools once (run_verification.py + audit.config per test)
              for X in TEST07 TEST09; do
                D=/shared/mlcommons/compliance/\$X; mkdir -p "\$D/gpt-oss-120b"
                [ -s "\$D/run_verification.py" ] || curl -fsSL "https://raw.githubusercontent.com/mlcommons/inference/master/compliance/\$X/run_verification.py" -o "\$D/run_verification.py"
                [ -s "\$D/gpt-oss-120b/audit.config" ] || curl -fsSL "https://raw.githubusercontent.com/mlcommons/inference/master/compliance/\$X/gpt-oss-120b/audit.config" -o "\$D/gpt-oss-120b/audit.config"
              done
              cd /lab-mlperf-inference/code
              # DEDICATED head: no co-located worker; all ${DEVICE_COUNT} GPUs are remote workers.
              echo "DEDICATED compliance head [\$T]: no co-located worker"
              # audit.config must be in cwd for LoadGen to pick it up
              cp /shared/mlcommons/compliance/\$T/gpt-oss-120b/audit.config ./audit.config
              echo "===== ${RUN}: run_harness \$T perf pass (device_count=${DEVICE_COUNT}, samples=${NSAMP}) ====="
              bash run_harness.sh --config-path gpt-oss-120b/ --config-name offline_mi355x --backend zmq \
                test_mode=performance port=12345 harness_config.device_count=${DEVICE_COUNT} \
                harness_config.dataset_path=/data/gpt-oss-120b/${DS} \
                harness_config.accuracy_dataset_path=/data/gpt-oss-120b/${ACCDS} \
                harness_config.total_sample_count=${NSAMP} \
                harness_config.user_conf_path=/lab-mlperf-inference/code/gpt-oss-120b/user_mi355x_mn.conf \
                harness_config.output_log_dir="\$TDIR" 2>&1 | tee "\$OUT/head-sut.log" | tail -30
              rm -f ./audit.config
              echo "===== ${RUN}: run_verification.py \$T ====="
              if [ "\$T" = "TEST07" ]; then
                python3 /shared/mlcommons/compliance/TEST07/run_verification.py \
                  -c "\$TDIR" -o "\$VOUT" \
                  --accuracy-script "python3 /lab-mlperf-inference/mlperf_inference/language/gpt-oss-120b/eval_mlperf_accuracy.py --reference-data /data/gpt-oss-120b/acc_eval_compliance_gpqa.parquet --mlperf-log {accuracy_log} --tokenizer /model/gpt-oss-120b/fp4_quantized --output-file /tmp/t07_eval.json" \
                  --score-pattern "'exact_match':\s*([\d.]+)" \
                  --audit-config /shared/mlcommons/compliance/TEST07/gpt-oss-120b/audit.config \
                  --accuracy-threshold 60.698 2>&1 | tee "\$OUT/verify-\$T.log" | tail -20 || echo "TEST07 verify: see log"
              else
                python3 /shared/mlcommons/compliance/TEST09/run_verification.py \
                  -c "\$TDIR" -o "\$VOUT" \
                  --audit-config /shared/mlcommons/compliance/TEST09/gpt-oss-120b/audit.config \
                  --min-output-tokens 1150.38 --max-output-tokens 1406.02 2>&1 | tee "\$OUT/verify-\$T.log" | tail -20 || echo "TEST09 verify: confirm args vs README"
              fi
              echo "===== \$T verification output ====="; find "\$VOUT" -maxdepth 3 -type f 2>/dev/null
              grep -iE 'PASS|FAIL|First token|exact_match|output|mean' "\$OUT/verify-\$T.log" 2>/dev/null | tail -12
          resources: { requests: { amd.com/gpu: 8, cpu: "180", memory: 1800Gi }, limits: { amd.com/gpu: 8 } }
          volumeMounts: [{ name: shared, mountPath: /shared }, { name: dshm, mountPath: /dev/shm }]
      volumes:
        - { name: shared, persistentVolumeClaim: { claimName: mlperf-shared } }
        - { name: dshm, emptyDir: { medium: Memory, sizeLimit: 512Gi } }
YAML

until kubectl -n "$NS" get pod -l "run=${RUN},role=head" -o name 2>/dev/null | grep -q pod; do sleep 1; done
sed -e "s|\${RUN}|${RUN}|g" -e "s|\${HEAD_IP}|${HEAD_IP}|g" -e "s|\${WORKERS}|${WORKERS}|g" -e "s|\${OUT}|${OUT}|g" \
    "$K/107-mn-gptoss-offline-workers.yaml" | kubectl apply -f -
echo ">> launched ${RUN} [$TEST]. watch: kubectl -n ${NS} logs -f job/${RUN}-head"
