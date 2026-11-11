#!/usr/bin/env bash
# Check that your laptop and your kubeconfig are ready for the workshop.
#
#   ./check.sh          (login.sh runs it for you)
#
# Uses KUBECONFIG if it is set, else the kubeconfig login.sh saved (~/.sglang-workshop/kubeconfig).
# One line per check, then OK or "ask an instructor". Runs only read-only commands with
# your own kubeconfig. Works with the bash that ships with macOS (3.2), Linux and WSL.
set -u

PASS=0; WARN=0; FAIL=0
ok()   { printf '  [ ok ]  %s\n' "$1"; PASS=$((PASS + 1)); }
warn() { printf '  [warn]  %s\n' "$1"; WARN=$((WARN + 1)); }
bad()  { printf '  [FAIL]  %s\n' "$1"; FAIL=$((FAIL + 1)); }

echo "Workshop check"

# --- local tools ---------------------------------------------------------------
if command -v kubectl >/dev/null 2>&1; then
  ok "kubectl $(kubectl version --client 2>/dev/null | awk '/Client Version/{print $3}')"
else
  bad "kubectl not found: install it from https://kubernetes.io/docs/tasks/tools/"
fi
if command -v uv >/dev/null 2>&1; then
  ok "$(uv --version 2>/dev/null)"
else
  bad "uv not found: install it from https://docs.astral.sh/uv/getting-started/installation/"
fi

# --- kubeconfig ----------------------------------------------------------------
# KUBECONFIG wins; else login.sh's kubeconfig. Never fall back to ~/.kube/config: that may be
# another cluster (work, home lab).
SAVED="$HOME/.sglang-workshop/kubeconfig"
KC_OK=1
if [ -z "${KUBECONFIG:-}" ]; then
  if [ -f "$SAVED" ]; then
    export KUBECONFIG="$SAVED"
    ok "kubeconfig: ~/.sglang-workshop/kubeconfig (saved by login.sh)"
  else
    bad "not logged in: run ./login.sh <passcode> (the passcode is on your desk card)"; KC_OK=0
  fi
elif [ ! -f "$KUBECONFIG" ]; then
  bad "KUBECONFIG points at a file that does not exist: $KUBECONFIG"; KC_OK=0
else
  ok "kubeconfig: $KUBECONFIG (from KUBECONFIG)"
fi

if [ "$KC_OK" = 1 ]; then
NS=$(kubectl config view --minify -o jsonpath='{..namespace}' 2>/dev/null)
if [[ "$NS" =~ ^ws-[0-9]{2,}$ ]]; then  # ws-NN, two or more digits (ws-01 ... ws-100)
  ok "namespace from your kubeconfig: $NS"
elif [ -z "$NS" ]; then
  bad "your kubeconfig has no namespace; ask an instructor"
else
  bad "namespace '$NS' is not a workshop namespace (ws-NN); is KUBECONFIG your workshop kubeconfig?"
fi

# --- access to the cluster -----------------------------------------------------
# If the credential is refused or the cluster is unreachable, the other checks would only repeat it.
REACH=1
if out=$(kubectl get pods 2>&1); then
  ok "kubectl get pods works ($( [ -z "$(kubectl get pods -o name 2>/dev/null)" ] && echo 'no pods yet' || echo "$(kubectl get pods -o name 2>/dev/null | wc -l | tr -d ' ') pod(s)"))"
else
  case "$out" in
    *Unauthorized*|*"must be logged in"*) bad "Unauthorized: this desk's access has been revoked or has expired; ask an instructor for a spare card"; REACH=0 ;;
    *"no such host"*|*"i/o timeout"*|*"connection refused"*|*"Unable to connect"*) bad "cannot reach the cluster API server (network): $(echo "$out" | tail -1)"; REACH=0 ;;
    *) bad "kubectl get pods failed: $(echo "$out" | tail -1)" ;;
  esac
fi

if [ "$REACH" = 1 ]; then
for what in "create llminferenceservices.serving.kserve.io" "create pods/exec" "create pods/portforward"; do
  # shellcheck disable=SC2086
  ans=$(kubectl auth can-i $what 2>/dev/null)
  if [ "$ans" = "yes" ]; then ok "can $what"; else bad "cannot $what (got '${ans:-no answer}')"; fi
done

if out=$(kubectl get nodes 2>&1); then
  bad "kubectl get nodes works: this looks like an admin kubeconfig, not your workshop one"
else
  case "$out" in
    *Forbidden*|*forbidden*) ok "kubectl get nodes is Forbidden (this is your participant kubeconfig)" ;;
    *) warn "kubectl get nodes failed for another reason: $(echo "$out" | tail -1)" ;;
  esac
fi

GPU=$(kubectl get resourcequota gpu-quota -o jsonpath='{.status.used.requests\.nvidia\.com/gpu}/{.status.hard.requests\.nvidia\.com/gpu}' 2>/dev/null)
case "$GPU" in
  0/1) ok "GPU quota: 0/1 in use (your GPU is free)" ;;
  1/1) warn "GPU quota: 1/1 in use. Fine if it is your qwen3 service; otherwise delete the extra pod (kubectl get pods)" ;;
  "")  bad "cannot read your quota (kubectl get resourcequota gpu-quota)" ;;
  *)   warn "GPU quota: $GPU" ;;
esac
else
  printf '  [skip]  the other cluster checks (fix the line above first)\n'
fi

else
  printf '  [skip]  cluster checks (log in first: ./login.sh <passcode>)\n'
fi

# --- local ports ---------------------------------------------------------------
LOCAL_PORT=${WORKSHOP_LOCAL_PORT:-8001}
CHAT_PORT=${WORKSHOP_CHAT_PORT:-8080}
port_free() { ! (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }
check_port() { # port, what, variable that sets it
  if port_free "$1"; then ok "port $1 free: $2"; else warn "port $1 busy ($2); the notebook will pick another port automatically (or set $3)"; fi
}
check_port "$LOCAL_PORT" "notebook port-forward" WORKSHOP_LOCAL_PORT
check_port "$((LOCAL_PORT + 1))" "chat UI port-forward, WORKSHOP_LOCAL_PORT + 1" WORKSHOP_LOCAL_PORT
check_port "$CHAT_PORT" "chat page" WORKSHOP_CHAT_PORT

echo
if [ "$FAIL" -eq 0 ]; then
  w=""; [ "$WARN" -gt 0 ] && w=" ($WARN warning(s) above)"
  echo "OK: $PASS checks passed$w. Next: uv run jupyter lab, open notebooks/01_Deploy.ipynb and run the deploy cell."
  exit 0
else
  echo "$FAIL check(s) failed: ask an instructor (show them this output)."
  exit 1
fi
