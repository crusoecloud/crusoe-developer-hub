"""Thin ``kubectl`` wrappers.

Nothing here talks to the Kubernetes API directly; every call shells out to
``kubectl`` so what the notebook does is exactly what you would type. A
``runner`` can be injected for tests.

Which cluster: ``KUBECONFIG`` if it is set, else the kubeconfig ``login.sh`` saved
(:data:`WORKSHOP_KUBECONFIG`). Importing this module sets ``KUBECONFIG`` to that file when it is
unset, so every ``kubectl`` started from the notebook (including ``!kubectl`` cells) uses it.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import time
from collections.abc import Callable, MutableMapping
from pathlib import Path
from typing import Any

#: Where ``login.sh`` saves the participant's kubeconfig.
WORKSHOP_KUBECONFIG = Path.home() / ".sglang-workshop" / "kubeconfig"


def use_workshop_kubeconfig(
    environ: MutableMapping[str, str] = os.environ, path: Path | None = None
) -> str | None:
    """Set ``KUBECONFIG`` to login.sh's kubeconfig if it is unset (or empty) and that file exists.

    ``KUBECONFIG`` always wins when it is set. Returns the ``KUBECONFIG`` in effect (``None``: none,
    kubectl would fall back to ``~/.kube/config``).
    """
    path = path or WORKSHOP_KUBECONFIG
    if not environ.get("KUBECONFIG") and path.is_file():
        environ["KUBECONFIG"] = str(path)
    return environ.get("KUBECONFIG") or None


use_workshop_kubeconfig()

Runner = Callable[[list[str], str | None], str]


def _subprocess_runner(cmd: list[str], stdin: str | None) -> str:
    proc = subprocess.run(cmd, input=stdin, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)}\n{proc.stderr.strip()}")
    return proc.stdout


#: Replace in tests: ``k8s.RUNNER = fake``.
RUNNER: Runner = _subprocess_runner


#: kubectl errors that mean the API server did not answer this time (Wi-Fi, a proxy, a busy
#: control plane), not that the request was wrong. Forbidden, NotFound, quota and validation
#: errors never match, so they are never retried.
TRANSIENT_ERRORS = (
    "Unable to connect to the server",
    "TLS handshake timeout",
    "i/o timeout",
    "no such host",  # a DNS hiccup on the laptop's network
    "connection reset by peer",
    "connection refused",
    "context deadline exceeded",
    "http2: client connection lost",
    "unexpected EOF",
    ": EOF",
    "the server is currently unable to handle the request",
    "error: Timeout occurred",  # kubectl exec's stream timed out
)

#: Verbs that are safe to repeat: reads, and writes that end in the same state however often
#: they run. ``create``, ``replace`` and ``delete`` are not here; ``exec`` only on request.
RETRY_VERBS = frozenset({"get", "logs", "describe", "config", "auth", "apply", "annotate"})

#: Attempts per call for a transient error, and the wait before each retry (doubling).
ATTEMPTS = 3
BACKOFF_S = 2.0

#: Replace in tests to skip the waits; NOTICE prints the "retrying" line.
SLEEP: Callable[[float], None] = time.sleep
NOTICE: Callable[[str], None] = print


def is_transient(exc: BaseException) -> bool:
    """True if ``exc`` is a kubectl failure worth retrying (see :data:`TRANSIENT_ERRORS`)."""
    return isinstance(exc, RuntimeError) and any(m in str(exc) for m in TRANSIENT_ERRORS)


def kubectl(*args: str, stdin: str | None = None, retry: bool | None = None) -> str:
    """Run ``kubectl`` with the given arguments and return stdout.

    A read or an idempotent write (:data:`RETRY_VERBS`; ``retry=True`` for anything else, such
    as an ``exec`` that only reads) is retried up to :data:`ATTEMPTS` times when the network
    fails (:data:`TRANSIENT_ERRORS`), with a short wait in between. Any other error is raised at once.
    """
    cmd = ["kubectl", *args]
    if retry is None:
        retry = bool(args) and args[0] in RETRY_VERBS
    attempts = ATTEMPTS if retry else 1
    for i in range(1, attempts + 1):
        try:
            return RUNNER(cmd, stdin)
        except RuntimeError as exc:
            if i == attempts or not is_transient(exc):
                raise
            NOTICE(f"network hiccup, retrying ({i + 1}/{attempts})…")
            SLEEP(BACKOFF_S * 2 ** (i - 1))
    raise AssertionError("unreachable")


#: Namespace used when neither WORKSHOP_NAMESPACE nor the kubeconfig context names one.
DEFAULT_NAMESPACE = "sglang-workshop"

#: Accelerator label per vendor on the shared workshop cluster (participants cannot list nodes).
WORKSHOP_ACCELERATORS = {"nvidia": "nvidia-h200-141gb-sxm-ib", "amd": "amd-mi355x-288gb-roce"}


def current_namespace() -> str:
    """The namespace to deploy into.

    ``WORKSHOP_NAMESPACE`` if set, else the namespace of the current kubeconfig
    context (a participant kubeconfig sets ``ws-NN``), else ``sglang-workshop``.
    """
    env = os.environ.get("WORKSHOP_NAMESPACE")
    if env:
        return env
    try:
        ns = kubectl("config", "view", "--minify", "-o", "jsonpath={..namespace}").strip()
    except RuntimeError:
        ns = ""
    return ns or DEFAULT_NAMESPACE


class NoNamespaceError(RuntimeError):
    """The participant's kubeconfig names no namespace."""


def participant_namespace() -> str:
    """The participant's ``ws-NN`` namespace; stops with a clear message if there is none.

    Like :func:`current_namespace`, but the ``sglang-workshop`` fallback is never right for a
    participant (their kubeconfig cannot use it), so it raises instead.
    """
    ns = current_namespace()
    if ns == DEFAULT_NAMESPACE and not os.environ.get("WORKSHOP_NAMESPACE"):
        raise NoNamespaceError(
            "your kubeconfig has no namespace: run ./login.sh <passcode> in the workshop folder "
            "(or check that KUBECONFIG points at your workshop kubeconfig), then restart the kernel"
        )
    return ns


def workshop_accelerator(vendor: str) -> str:
    """The ``crusoe.ai/accelerator`` label to deploy onto, without listing nodes.

    ``WORKSHOP_ACCELERATOR`` if set, else the workshop cluster's pool for ``vendor``.
    Admins with node access can use :func:`detect_accelerator` instead.
    """
    env = os.environ.get("WORKSHOP_ACCELERATOR")
    if env:
        return env
    if vendor not in WORKSHOP_ACCELERATORS:
        raise KeyError(f"unknown vendor {vendor!r}; choose one of {sorted(WORKSHOP_ACCELERATORS)}")
    return WORKSHOP_ACCELERATORS[vendor]


def apply(manifest_yaml: str) -> str:
    """``kubectl apply -f -``"""
    return kubectl("apply", "-f", "-", stdin=manifest_yaml).strip()


def delete_llmisvc(name: str, namespace: str) -> str:
    return kubectl("delete", "llminferenceservice", name, "-n", namespace, "--ignore-not-found").strip()


def workload_deployment(name: str) -> str:
    """Name of the Deployment KServe v0.19 creates for an LLMInferenceService."""
    return f"{name}-kserve"


def workload_selector(name: str) -> str:
    """Label selector that matches exactly the serving pods of an LLMInferenceService."""
    return f"app.kubernetes.io/name={name},kserve.io/component=workload"


def get_llmisvc(name: str, namespace: str) -> dict[str, Any]:
    return json.loads(kubectl("get", "llminferenceservice", name, "-n", namespace, "-o", "json"))


def workload_pods(name: str, namespace: str) -> list[dict[str, Any]]:
    """Serving pods of an LLMInferenceService (the router and scheduler pods are excluded)."""
    out = json.loads(kubectl("get", "pods", "-n", namespace, "-l", workload_selector(name), "-o", "json"))
    return list(out.get("items", []))


def _pod_phase(pod: dict[str, Any]) -> str:
    phase = pod.get("status", {}).get("phase", "?")
    for cs in pod.get("status", {}).get("initContainerStatuses", []) or []:
        if cs.get("state", {}).get("running"):
            return f"Init:{cs['name']}"
    for cs in pod.get("status", {}).get("containerStatuses", []) or []:
        waiting = cs.get("state", {}).get("waiting")
        if waiting:
            return waiting.get("reason", "Waiting")
    ready = all(cs.get("ready") for cs in pod.get("status", {}).get("containerStatuses", []) or [])
    return "Ready" if phase == "Running" and ready else phase


#: Shown while an LLMInferenceService exists but KServe has not created its pod yet.
KSERVE_QUEUE_MSG = (
    "waiting for KServe to create your pod (deploys are processed one at a time; "
    "this can take a few minutes when everyone deploys together)"
)


def status_line(name: str, namespace: str) -> str:
    """One line you can print in a loop: service readiness plus each pod's state."""
    try:
        svc = get_llmisvc(name, namespace)
    except RuntimeError as exc:
        return f"{name}: not found ({str(exc).strip().splitlines()[-1]})"
    conds = {c["type"]: c for c in svc.get("status", {}).get("conditions", []) or []}
    ready = conds.get("Ready", {}).get("status", "Unknown")
    reason = conds.get("Ready", {}).get("reason", "")
    pods = workload_pods(name, namespace)
    if not pods and ready != "True":
        return f"{name}: {KSERVE_QUEUE_MSG}"
    states = ", ".join(f"{p['metadata']['name'].split('-')[-1]}={_pod_phase(p)}" for p in pods)
    return f"{name}: Ready={ready} {reason}  pods[{states}]"


def is_ready(name: str, namespace: str) -> bool:
    svc = get_llmisvc(name, namespace)
    conds = svc.get("status", {}).get("conditions", []) or []
    return any(c["type"] == "Ready" and c["status"] == "True" for c in conds)


def wait_ready(
    name: str,
    namespace: str,
    timeout_s: int = 1800,
    poll_s: int = 15,
    log: Callable[[str], None] = print,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    """Poll until the service reports Ready. Prints one status line per poll and a final Ready line.

    Use this for a first deploy. After a change, use ``service.redeploy``: the service can
    report Ready while the old pod still serves the old configuration.
    """
    start = time.monotonic()
    while True:
        line = status_line(name, namespace)
        elapsed = int(time.monotonic() - start)
        log(f"[{elapsed:4d}s] {line}")
        try:
            if is_ready(name, namespace):
                log(f"[{int(time.monotonic() - start):4d}s] {name}: Ready")
                return True
        except RuntimeError:
            pass
        if elapsed >= timeout_s:
            log(f"gave up after {timeout_s}s; run logs() and events() to see why")
            return False
        sleep(poll_s)


def replica_failures(name: str, namespace: str) -> list[str]:
    """``ReplicaFailure`` messages on the service's ReplicaSets, e.g. "exceeded quota" when a
    second GPU pod cannot be created. Empty when nothing is failing."""
    out = json.loads(
        kubectl("get", "replicasets", "-n", namespace, "-l", workload_selector(name), "-o", "json")
    )
    msgs = []
    for rs in out.get("items", []):
        for c in rs.get("status", {}).get("conditions", []) or []:
            if c.get("type") == "ReplicaFailure" and c.get("status") == "True":
                msgs.append(c.get("message", c.get("reason", "ReplicaFailure")))
    return msgs


def annotate_llmisvc(name: str, namespace: str, annotation: str) -> str:
    """``kubectl annotate llminferenceservice``; ``key=value`` sets, ``key-`` removes."""
    return kubectl(
        "annotate", "llminferenceservice", name, "-n", namespace, annotation, "--overwrite"
    ).strip()


def logs(name: str, namespace: str, tail: int = 40, container: str = "main") -> str:
    """Last lines of the SGLang server log."""
    return kubectl("logs", "-n", namespace, "-l", workload_selector(name), "-c", container, f"--tail={tail}")


def events(namespace: str, name: str | None = None) -> str:
    out = kubectl("get", "events", "-n", namespace, "--sort-by=.lastTimestamp")
    if name:
        out = "\n".join(line for line in out.splitlines() if name in line or line.startswith("LAST"))
    return out


def exec_in_pod(
    name: str, namespace: str, cmd: list[str], container: str = "main", retry: bool = False
) -> str:
    """Run ``cmd`` in the serving pod. ``retry=True`` only for a command that is safe to run twice
    (it only reads, or does nothing the second time); it lets a network hiccup be retried."""
    pods = workload_pods(name, namespace)
    if not pods:
        raise RuntimeError(f"no serving pod for {name} in {namespace}")
    pod = pods[0]["metadata"]["name"]
    return kubectl("exec", "-n", namespace, pod, "-c", container, "--", *cmd, retry=retry)


def node_gpus() -> list[dict[str, Any]]:
    """Each node's accelerator label and allocatable GPU count (both vendors)."""
    nodes = json.loads(kubectl("get", "nodes", "-o", "json"))
    rows = []
    for n in nodes.get("items", []):
        alloc = n.get("status", {}).get("allocatable", {})
        rows.append(
            {
                "node": n["metadata"]["name"],
                "accelerator": n["metadata"].get("labels", {}).get("crusoe.ai/accelerator", ""),
                "nvidia.com/gpu": int(alloc.get("nvidia.com/gpu", 0)),
                "amd.com/gpu": int(alloc.get("amd.com/gpu", 0)),
            }
        )
    return rows


def detect_accelerator(vendor: str) -> str | None:
    """The ``crusoe.ai/accelerator`` label of the first node with allocatable GPUs of ``vendor``, or None."""
    resource = {"nvidia": "nvidia.com/gpu", "amd": "amd.com/gpu"}[vendor]
    for row in node_gpus():
        if row[resource] > 0 and row["accelerator"]:
            return row["accelerator"]
    return None


def exists_llmisvc(name: str, namespace: str) -> bool:
    """Whether an LLMInferenceService with this name exists."""
    names = kubectl("get", "llminferenceservice", "-n", namespace, "-o", "jsonpath={.items[*].metadata.name}")
    return name in names.split()


#: How far past the requested local port to look for a free one.
PORT_SPAN = 20


def port_free(port: int, host: str = "127.0.0.1") -> bool:
    """True if nothing listens on ``host:port`` (we can bind it ourselves)."""
    with socket.socket() as s:
        try:
            s.bind((host, port))
        except OSError:
            return False
    return True


def free_ports(start: int, span: int = PORT_SPAN):
    """The free ports from ``start`` to ``start + span``, in order (checked lazily, one by one)."""
    return (p for p in range(start, start + span + 1) if port_free(p))


def no_free_port(start: int, what: str, variable: str, span: int = PORT_SPAN) -> RuntimeError:
    return RuntimeError(
        f"{what}: ports {start} to {start + span} on this laptop are all in use by other programs. "
        f"Set {variable}=<another port> (e.g. {start + 1000}) and restart, or ask an instructor."
    )


class PortForward:
    """``kubectl port-forward`` to the workload Service, as a context manager.

    >>> with PortForward("qwen3", "sglang-workshop") as url:
    ...     requests.get(f"{url}/v1/models")

    The local port is 8001, or ``WORKSHOP_LOCAL_PORT`` if set. If another program on your laptop
    holds it, the next free port (up to :data:`PORT_SPAN` higher) is used instead, and ``log`` says so.
    """

    def __init__(
        self,
        name: str,
        namespace: str,
        local_port: int | None = None,
        remote_port: int = 8000,
        log: Callable[[str], None] = print,
    ):
        self.service = f"svc/{name}-kserve-workload-svc"
        self.namespace = namespace
        self.local_port = local_port or int(os.environ.get("WORKSHOP_LOCAL_PORT", "8001"))
        self.remote_port = remote_port
        self.log = log
        self.proc: subprocess.Popen | None = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.local_port}"

    def __enter__(self) -> str:
        # A port another program holds would answer the readiness check below, and every request
        # would go to that program instead of your server; so only ever start on a port we could bind.
        wanted = self.local_port
        for port in free_ports(wanted):
            self.local_port = port
            if self._start():
                if port != wanted:
                    self.log(f"port {wanted} busy, using {port} instead")
                return self.url
        raise no_free_port(wanted, "port-forward", "WORKSHOP_LOCAL_PORT")

    def _start(self) -> bool:
        """Start kubectl on :attr:`local_port`; False if another program took the port meanwhile."""
        self.proc = subprocess.Popen(
            [
                "kubectl",
                "port-forward",
                "-n",
                self.namespace,
                self.service,
                f"{self.local_port}:{self.remote_port}",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(60):
            if self.proc.poll() is not None:
                err = self.proc.stderr.read()
                if "address already in use" in err:
                    return False
                raise RuntimeError(f"port-forward exited: {err}")
            with socket.socket() as s:
                s.settimeout(0.5)
                if s.connect_ex(("127.0.0.1", self.local_port)) == 0:
                    return True
            time.sleep(0.5)
        self.__exit__()
        raise TimeoutError("port-forward did not open the local port in 30s")

    def __exit__(self, *exc: object) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
