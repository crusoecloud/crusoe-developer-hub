#!/usr/bin/env bash
# Log in to the workshop with the passcode on your desk card.
#
#   ./login.sh <passcode>                   download the login bundle and unlock your desk
#   ./login.sh --bundle <file> <passcode>   same, from a copy of the bundle (ask the help desk)
#   ./login.sh --file <kubeconfig>          a kubeconfig handed to you directly
#
# Saves your kubeconfig to ~/.sglang-workshop/kubeconfig (the notebooks and check.sh use it when
# KUBECONFIG is not set), then runs check.sh. Safe to run again. Works with the bash that ships
# with macOS (3.2), Linux and WSL. The work is done by workshop_utils/login.py in the uv environment.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(dirname "$HERE")
PY="$ROOT/.venv/bin/python"
SAVED="$HOME/.sglang-workshop/kubeconfig"

die() { echo "login failed: $1" >&2; exit "${2:-6}"; }

command -v uv >/dev/null 2>&1 ||
  die "uv not found: install it from https://docs.astral.sh/uv/getting-started/installation/, then run uv sync"
if [ ! -x "$PY" ] || ! "$PY" -c 'import cryptography, yaml' >/dev/null 2>&1; then
  die "the workshop environment is not set up: run 'uv sync' in $HERE (it needs the internet once), then ./login.sh again" 7
fi

PYTHONUNBUFFERED=1 PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}" "$PY" -m workshop_utils.login "$@" || exit $?
case " $* " in *" -h "* | *" --help "*) exit 0 ;; esac

echo
# Without KUBECONFIG, check.sh uses the kubeconfig just saved (and says so).
(unset KUBECONFIG; "$HERE/check.sh")
rc=$?
echo
if [ -n "${KUBECONFIG:-}" ] && [ "$KUBECONFIG" != "$SAVED" ]; then
  echo "Note: KUBECONFIG is set to $KUBECONFIG in this terminal, so the notebooks would use that file."
  echo "      Run: unset KUBECONFIG   (or: export KUBECONFIG=$SAVED)"
fi
echo "For kubectl in a terminal: export KUBECONFIG=~/.sglang-workshop/kubeconfig"
exit $rc
