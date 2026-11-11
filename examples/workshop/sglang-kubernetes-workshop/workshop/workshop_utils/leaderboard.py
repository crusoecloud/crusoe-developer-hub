"""Submit a benchmark result to the workshop leaderboard.

Only results from the standard benchmark (``bench.standard_bench``, parameters in
``bench.STANDARD``) are ranked; each entry records its parameters and a ``standard`` flag.

The result is stored in the participant's own namespace as the ConfigMap
``leaderboard-result`` (key ``result.json``); the instructors' ``collect.py`` reads it from
every namespace. Nothing is sent over the network from the laptop or the pod.

The Role allows create and update (replace) on ConfigMaps but not patch, so this uses
``kubectl create`` the first time and ``kubectl replace`` afterwards, never ``kubectl apply``.

``result.json`` holds the latest submission at the top level (``namespace``, ``submitted_at``,
``label``, ``config``, ``concurrency`` and the metrics) plus ``runs``, every submission so far.
"""

from __future__ import annotations

import json
from dataclasses import fields
from datetime import datetime, timezone
from typing import Any

from . import k8s
from .bench import STANDARD, BenchResult, benchmark_params, is_standard
from .manifest import ServingConfig, sglang_args

CONFIGMAP = "leaderboard-result"
KEY = "result.json"
MAX_BYTES = 64 * 1024
MAX_RUNS = 20

#: Metrics copied into each submission.
FIELDS = (
    "output_throughput",
    "median_ttft_ms",
    "p99_ttft_ms",
    "median_tpot_ms",
    "p99_tpot_ms",
    "p99_itl_ms",
    "request_throughput",
)

#: ServingConfig fields that identify the participant, not the tuning.
_NOT_TUNING = {"name", "namespace", "labels", "gateway"}


def config_diff(cfg: ServingConfig, baseline: ServingConfig | None = None) -> dict[str, Any]:
    """The ServingConfig fields that differ from the baseline (empty dict = the baseline)."""
    baseline = baseline or ServingConfig()
    out = {}
    for f in fields(ServingConfig):
        if f.name in _NOT_TUNING:
            continue
        v = getattr(cfg, f.name)
        if v != getattr(baseline, f.name):
            out[f.name] = list(v) if isinstance(v, tuple) else v
    return out


def make_entry(
    result: BenchResult,
    cfg: ServingConfig,
    namespace: str,
    label: str | None = None,
    now: datetime | None = None,
    kv_tokens: int | None = None,
) -> dict[str, Any]:
    diff = config_diff(cfg)
    entry = {
        "namespace": namespace,
        "submitted_at": (now or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "label": label or (", ".join(f"{k}={v}" for k, v in diff.items()) or "baseline"),
        "config": diff,
        "concurrency": result.concurrency,
        "input_len": result.input_len,
        "output_len": result.output_len,
        "benchmark": benchmark_params(result),
        "standard": is_standard(result),
        "cache_flushed": result.cache_flushed,
    }
    entry.update({k: result.metrics[k] for k in FIELDS if k in result.metrics})
    if kv_tokens:
        entry["kv_tokens"] = kv_tokens  # the server's KV pool, for the results wall
    return entry


_KV_PY = (
    "import json,urllib.request;"
    "print(json.load(urllib.request.urlopen('http://127.0.0.1:8000/get_server_info'))"
    ".get('max_total_num_tokens') or '')"
)


def kv_tokens(cfg: ServingConfig, namespace: str) -> int | None:
    """The KV pool size (``max_total_num_tokens``) of the Ready pod running exactly ``cfg``, or
    None if there is no such pod or it cannot be read. Best effort: never fails a submission."""
    try:
        args = sglang_args(cfg)
        for pod in k8s.workload_pods(cfg.name, namespace):
            main = next((c for c in pod["spec"]["containers"] if c.get("name") == "main"), {})
            ready = any(c.get("type") == "Ready" and c.get("status") == "True"
                        for c in pod.get("status", {}).get("conditions", []) or [])  # fmt: skip
            if ready and list(main.get("args", [])) == args and not pod["metadata"].get("deletionTimestamp"):
                name = pod["metadata"]["name"]
                out = k8s.kubectl(
                    "exec", "-n", namespace, name, "-c", "main", "--", "python3", "-c", _KV_PY, retry=True
                )
                return int(out.strip())
    except Exception:  # noqa: BLE001
        return None
    return None


def _configmap(namespace: str, doc: dict[str, Any]) -> str:
    return json.dumps(
        {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": CONFIGMAP,
                "namespace": namespace,
                "labels": {"workshop/leaderboard": "true"},
            },
            "data": {KEY: json.dumps(doc, indent=1)},
        }
    )


def read(namespace: str) -> dict[str, Any] | None:
    """The current ``result.json`` in ``namespace``, or None if nothing was submitted yet."""
    try:
        out = k8s.kubectl("get", "configmap", CONFIGMAP, "-n", namespace, "-o", "json")
    except RuntimeError as exc:
        if "NotFound" in str(exc) or "not found" in str(exc):
            return None
        raise
    return json.loads(json.loads(out)["data"][KEY])


def submit(
    result: BenchResult,
    cfg: ServingConfig,
    label: str | None = None,
    namespace: str | None = None,
    log=print,
) -> dict[str, Any]:
    """Store ``result`` for the leaderboard. Use a result from ``bench.standard_bench``; any
    other run is stored but marked ``standard: false`` and is not ranked."""
    ns = namespace or cfg.namespace or k8s.participant_namespace()
    entry = make_entry(result, cfg, ns, label, kv_tokens=kv_tokens(cfg, ns))
    current = read(ns)
    runs = ((current or {}).get("runs") or [])[-(MAX_RUNS - 1) :] + [entry]
    doc = {**entry, "runs": runs}
    body = _configmap(ns, doc)
    if len(body.encode()) > MAX_BYTES:
        raise ValueError(f"leaderboard result is {len(body)} bytes, over {MAX_BYTES}")
    verb = "create" if current is None else "replace"
    k8s.kubectl(verb, "-f", "-", stdin=body)
    tput = entry.get("output_throughput")
    log(
        f"submitted '{entry['label']}' to the leaderboard ({ns}, run {len(runs)}): "
        f"{tput:.0f} tok/s at concurrency {entry['concurrency']}"
        if tput is not None
        else f"submitted '{entry['label']}' ({ns}, run {len(runs)})"
    )
    return doc


def as_result(entry: dict[str, Any]) -> BenchResult:
    """A submitted run as a :class:`BenchResult` (the metrics in :data:`FIELDS`), for tables and plots."""
    params = entry.get("benchmark") or {}
    return BenchResult(
        label=entry.get("label", "?"),
        concurrency=int(params.get("max_concurrency", entry.get("concurrency", 0))),
        input_len=int(params.get("input_len", entry.get("input_len", 0))),
        output_len=int(params.get("output_len", entry.get("output_len", 0))),
        num_prompts=int(params.get("num_prompts", 0)),
        dataset=params.get("dataset", "random-ids"),
        seed=int(params.get("seed", 42)),
        range_ratio=params.get("range_ratio"),
        cache_flushed=bool(entry.get("cache_flushed", False)),
        metrics={k: float(entry[k]) for k in FIELDS if k in entry},
    )


def _is_standard_entry(entry: dict[str, Any]) -> bool:
    """Recomputed from the recorded parameters, not the stored flag: runs submitted before the
    standard benchmark changed (or with a hand-edited flag) do not count, nor runs measured without
    emptying the prefix cache first, nor runs an instructor marked ``excluded``."""
    return (
        entry.get("benchmark") == STANDARD
        and entry.get("cache_flushed") is True
        and not entry.get("excluded")
    )


def standard_results(namespace: str) -> dict[str, BenchResult]:
    """Your submitted standard-benchmark runs, the latest one per label, in submission order.

    Read from the cluster, so it survives a kernel restart or a new laptop."""
    doc = read(namespace) or {}
    out: dict[str, BenchResult] = {}
    for entry in doc.get("runs", []):
        if _is_standard_entry(entry):
            out.pop(entry.get("label"), None)
            out[entry.get("label", "?")] = as_result(entry)
    return out


def best_run(doc: dict[str, Any], max_p99_ttft_ms: float | None = None) -> dict | None:
    """The standard-benchmark run with the highest output throughput (optionally under a p99 TTFT cap).

    Runs measured with other parameters (other concurrency, fewer prompts, ...) never rank.
    """
    runs = [r for r in doc.get("runs", [doc]) if _is_standard_entry(r) and "output_throughput" in r]
    if max_p99_ttft_ms is not None:
        runs = [r for r in runs if r.get("p99_ttft_ms", float("inf")) <= max_p99_ttft_ms]
    return max(runs, key=lambda r: r["output_throughput"], default=None)


__all__ = [
    "CONFIGMAP",
    "KEY",
    "as_result",
    "best_run",
    "config_diff",
    "make_entry",
    "read",
    "standard_results",
    "submit",
]
