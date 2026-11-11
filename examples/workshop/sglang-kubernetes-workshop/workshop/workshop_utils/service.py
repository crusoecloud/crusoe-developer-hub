"""Deploy and redeploy the participant's ``LLMInferenceService``.

On the shared workshop cluster every participant has exactly one GPU. KServe updates its
Deployment with a rolling update, which starts the new pod before stopping the old one, so a
plain ``kubectl apply`` of a changed config never completes: the new pod cannot get a second
GPU, the old pod keeps serving the old config, and the service still says Ready.

:func:`redeploy` therefore follows the stop-annotation route: stop the service, wait until its
pod is gone, apply the new config, remove the stop annotation, and wait for the NEW pod (its
arguments match the new config) to be Ready.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from . import k8s
from .leaderboard import config_diff
from .manifest import ServingConfig, config_from_container, render, sglang_args, to_yaml
from .profiles import Profile, profile

#: KServe annotation that deletes the workload Deployment but keeps the LLMInferenceService.
STOP_ANNOTATION = "serving.kserve.io/stop"

#: Measured on the workshop cluster with pre-pulled images (188-227 s).
REDEPLOY_ETA_S = 200
#: First deploy on a warm node with a short KServe queue (160-202 s).
DEPLOY_ETA_S = 180


class RedeployError(RuntimeError):
    """The redeploy cannot finish on its own; the message says what to do."""


#: The participant's service name: the notebooks, the chat UI and the kubectl-only manifest agree on it.
PARTICIPANT_SERVICE = "qwen3"


def participant_profile(vendor: str = "nvidia") -> Profile:
    """The vendor profile pinned to the workshop cluster's GPU pool (no node listing needed)."""
    return profile(vendor, accelerator=k8s.workshop_accelerator(vendor))


def participant_config(namespace: str | None = None, **changes: Any) -> ServingConfig:
    """The participant's ServingConfig: service ``qwen3`` in their namespace, no Gateway route
    (port-forward only), model and flags from ServingConfig unless overridden in ``changes``."""
    ns = namespace or k8s.participant_namespace()
    return ServingConfig(name=PARTICIPANT_SERVICE, namespace=ns, gateway=False, **changes)


def manifest_yaml(prof: Profile, cfg: ServingConfig) -> str:
    return to_yaml(render(prof, cfg))


def deploy(prof: Profile, cfg: ServingConfig, log: Callable[[str], None] = print) -> str:
    """Apply the service and return immediately (``kubectl apply``). Follow with ``wait_ready``."""
    out = k8s.apply(manifest_yaml(prof, cfg))
    log(out)
    log(
        f"Deploy requested in {cfg.namespace}. KServe creates the pod, the model downloads (~1 min) and "
        f"loads (~1.5 min): about {DEPLOY_ETA_S // 60} min when the queue is short, longer when everyone "
        "deploys at once."
    )
    return out


def _is_new_ready(pod: dict[str, Any], args: list[str]) -> bool:
    """Ready, not terminating, and running exactly the new server arguments."""
    if pod.get("metadata", {}).get("deletionTimestamp"):
        return False
    main = next((c for c in pod.get("spec", {}).get("containers", []) if c.get("name") == "main"), None)
    if main is None or list(main.get("args", [])) != list(args):
        return False
    conds = pod.get("status", {}).get("conditions", []) or []
    return any(c.get("type") == "Ready" and c.get("status") == "True" for c in conds)


def _last_line(exc: BaseException) -> str:
    return str(exc).strip().splitlines()[-1]


def _pods_or_retry(name: str, namespace: str, log: Callable[[str], None]) -> list[dict[str, Any]]:
    """The workload pods; on a transient error, report it and answer "still there" so the caller polls again."""
    try:
        return k8s.workload_pods(name, namespace)
    except RuntimeError as exc:
        if not k8s.is_transient(exc):
            raise
        log(f"{name}: no answer from the cluster ({_last_line(exc)}); trying again")
        return [{}]


def _quota_failure(name: str, namespace: str) -> str | None:
    for msg in k8s.replica_failures(name, namespace):
        if "exceeded quota" in msg:
            return msg
    return None


def wait_new_pod(
    cfg: ServingConfig,
    timeout_s: int = 1800,
    poll_s: int = 10,
    eta_s: int = REDEPLOY_ETA_S,
    log: Callable[[str], None] = print,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> str:
    """Wait for a Ready pod running ``cfg``'s arguments; return its name.

    Raises :class:`RedeployError` if KServe cannot create the pod because of the quota
    (usually another GPU pod is still running in the namespace), or on timeout.
    """
    name, ns, args = cfg.name, cfg.namespace, sglang_args(cfg)
    start = clock()
    while True:
        elapsed = int(clock() - start)
        try:
            pods = k8s.workload_pods(name, ns)
            failure = None if any(_is_new_ready(p, args) for p in pods) else _quota_failure(name, ns)
        except RuntimeError as exc:
            if not k8s.is_transient(exc) or elapsed >= timeout_s:
                raise
            log(f"[{elapsed:4d}s] {name}: no answer from the cluster ({_last_line(exc)}); trying again")
            sleep(poll_s)
            continue
        new = [p for p in pods if _is_new_ready(p, args)]
        if new:
            pod = new[0]["metadata"]["name"]
            log(f"[{elapsed:4d}s] {name}: Ready ({pod})")
            return pod
        if failure:
            raise RedeployError(
                f"KServe cannot start your pod: {failure}\n"
                "Another pod in your namespace is holding your GPU. List them with `kubectl get pods`, "
                "delete the extra one (`kubectl delete pod <name>`), and the deploy continues by itself."
            )
        if not pods:
            state = k8s.KSERVE_QUEUE_MSG
        else:
            state = ", ".join(f"{p['metadata']['name'].split('-')[-1]}={k8s._pod_phase(p)}" for p in pods)
        left = max(0, eta_s - elapsed)
        if not left:
            hint = "taking longer than usual"
        elif left >= 60:
            hint = f"about {left // 60} min {left % 60:02d} s left"
        else:
            hint = f"about {left} s left"
        log(f"[{elapsed:4d}s] {name}: {state}  ({hint})")
        if elapsed >= timeout_s:
            raise RedeployError(f"{name} not Ready after {timeout_s}s; check `kubectl get pods` and the logs")
        sleep(poll_s)


def redeploy(
    prof: Profile,
    cfg: ServingConfig,
    timeout_s: int = 1800,
    poll_s: int = 10,
    log: Callable[[str], None] = print,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> str:
    """Replace the running service with ``cfg`` and wait until the new pod is Ready.

    1. annotate ``serving.kserve.io/stop=true``  2. wait until no workload pod is left
    3. apply the new config  4. remove the annotation (``kubectl apply`` would not)
    5. wait for the NEW pod: Ready, not terminating, arguments equal ``sglang_args(cfg)``
    6. if KServe reports "exceeded quota", say so plainly instead of waiting

    Returns the new pod's name. If the service does not exist yet, this is a first deploy.
    """
    name, ns = cfg.name, cfg.namespace
    start = clock()
    if not k8s.exists_llmisvc(name, ns):
        log(f"{name} does not exist yet in {ns}; deploying it")
        deploy(prof, cfg, log=log)
        return wait_new_pod(cfg, timeout_s, poll_s, DEPLOY_ETA_S, log, sleep, clock)

    log(f"1/4 stopping {name} (frees your GPU; the service object, its status and history stay)")
    k8s.annotate_llmisvc(name, ns, f"{STOP_ANNOTATION}=true")
    while _pods_or_retry(name, ns, log):
        if clock() - start > 300:
            raise RedeployError(f"{name}'s pod did not stop within 5 minutes; check `kubectl get pods`")
        sleep(poll_s / 2)
    log(f"2/4 old pod gone after {int(clock() - start)}s; applying the new configuration")
    k8s.apply(manifest_yaml(prof, cfg))
    log("3/4 starting it again")
    k8s.annotate_llmisvc(name, ns, f"{STOP_ANNOTATION}-")
    log(
        f"4/4 waiting for the new pod (a redeploy takes about {REDEPLOY_ETA_S // 60} min {REDEPLOY_ETA_S % 60} s)"
    )
    pod = wait_new_pod(
        cfg, timeout_s, poll_s, max(0, REDEPLOY_ETA_S - int(clock() - start)), log, sleep, clock
    )
    log(f"redeployed in {int(clock() - start)}s")
    return pod


@dataclass(frozen=True)
class Current:
    """What is deployed right now, read from the cluster (works after a kernel restart)."""

    cfg: ServingConfig | None  # None: no service yet
    state: str  # "running", "starting", "stopped" or "not deployed"
    pod: str | None = None

    def changes(self) -> dict[str, Any]:
        """ServingConfig fields that differ from the baseline; empty means the baseline."""
        return config_diff(self.cfg) if self.cfg is not None else {}

    def describe(self) -> str:
        if self.cfg is None:
            return f"{PARTICIPANT_SERVICE} is not deployed yet: run the Start cells of 01_Deploy first"
        ch = self.changes()
        what = "the baseline" if not ch else ", ".join(f"{k}={v}" for k, v in ch.items())
        where = f" (pod {self.pod})" if self.pod else ""
        hint = {
            "starting": "; it is starting, so wait before benchmarking",
            "stopped": "; it is stopped (an interrupted redeploy): the next experiment cell starts it again",
        }.get(self.state, "")
        return f"{self.cfg.name} in {self.cfg.namespace}: {self.state}{where}, config: {what}{hint}"


def current_config(namespace: str | None = None, name: str = PARTICIPANT_SERVICE) -> Current:
    """Read the service's configuration from the cluster.

    The arguments of the Ready pod if there is one (that is what is actually serving),
    otherwise the service's spec (what KServe will start).
    """
    ns = namespace or k8s.participant_namespace()
    if not k8s.exists_llmisvc(name, ns):
        return Current(None, "not deployed")
    svc = k8s.get_llmisvc(name, ns)
    uri = svc.get("spec", {}).get("model", {}).get("uri")
    for pod in k8s.workload_pods(name, ns):
        main = next((c for c in pod["spec"]["containers"] if c.get("name") == "main"), None)
        if main and _is_new_ready(pod, list(main.get("args", []))):
            cfg = participant_config(ns, **config_from_container(main, uri))
            return Current(cfg, "running", pod["metadata"]["name"])
    main = next(c for c in svc["spec"]["template"]["containers"] if c.get("name") == "main")
    cfg = participant_config(ns, **config_from_container(main, uri))
    stopped = svc.get("metadata", {}).get("annotations", {}).get(STOP_ANNOTATION) == "true"
    return Current(cfg, "stopped" if stopped else "starting")


def switch_to(prof: Profile, cfg: ServingConfig, log: Callable[[str], None] = print, **kw: Any) -> str:
    """Make ``cfg`` the running config: redeploy, unless a Ready pod already runs exactly it
    (re-running a cell after a kernel restart does not cost another 3 minutes)."""
    now = current_config(cfg.namespace, cfg.name)
    if now.state == "running" and now.cfg is not None and sglang_args(now.cfg) == sglang_args(cfg):
        log(f"{cfg.name} already runs this configuration ({now.pod}); no redeploy needed")
        return now.pod or ""
    return redeploy(prof, cfg, log=log, **kw)
