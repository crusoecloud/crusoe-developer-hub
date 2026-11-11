"""Render a KServe ``LLMInferenceService`` that runs SGLang.

The manifest is the same for both vendors except for the values that come from
the :class:`~workshop_utils.profiles.Profile`. Everything about *how* the model
is served (tensor parallelism, KV cache precision, scheduling policy, KV cache
tiering) lives in :class:`ServingConfig`, so a tuning experiment is a new
config, a re-render and a ``kubectl apply``.

How the pieces fit:

- ``spec.model.uri`` tells KServe where the weights are. For ``hf://`` the
  storage-initializer downloads them to ``/mnt/models`` before the server
  starts; for ``pvc://`` it mounts a volume you staged earlier.
- ``spec.template.containers[main]`` is the SGLang server. The ``command`` and
  ``args`` are set explicitly so the manifest works whichever base template
  your KServe version ships (the vLLM default or the newer SGLang template).
- ``spec.router`` asks KServe to create a Gateway API route for the service.
  Inside the cluster you can also reach the pod directly through the
  ``<name>-kserve-workload-svc`` Service on port 8000.

Flag names follow the SGLang server arguments reference:
https://docs.sglang.io/advanced_features/server_arguments.html
"""

from __future__ import annotations

from dataclasses import dataclass, field

import yaml

from .profiles import Profile

#: Port the server listens on. KServe routes to 8000, SGLang defaults to 30000, so we set it.
PORT = 8000

#: Where the storage-initializer puts the weights.
MODEL_DIR = "/mnt/models"


@dataclass(frozen=True)
class ServingConfig:
    """How the model is served. Vendor-independent."""

    name: str = "qwen3"
    namespace: str = "sglang-workshop"
    model_uri: str = "hf://Qwen/Qwen3-8B"
    served_model_name: str = "qwen3"
    replicas: int = 1

    # Parallelism and memory
    tp: int = 1
    mem_fraction_static: float | None = None
    kv_cache_dtype: str | None = None  # e.g. "fp8_e4m3"
    context_length: int | None = None

    # Scheduling
    chunked_prefill_size: int | None = None
    max_running_requests: int | None = None
    schedule_policy: str | None = None  # "fcfs" (default), "lpm", ...
    cuda_graph_max_bs: int | None = (
        None  # largest decode batch replayed from a CUDA graph (default 256 on H200)
    )

    # KV cache tiering (HiCache): GPU -> host memory [-> storage]
    hicache: bool = False
    hicache_ratio: float | None = None  # host pool size as a multiple of the GPU pool
    hicache_size_gb: int | None = None  # absolute host pool size; overrides the ratio
    page_size: int | None = None  # HiCache works best with pages, 64 is the documented example

    # Output parsing
    reasoning_parser: str | None = "qwen3"

    # Anything else, appended verbatim
    extra_args: tuple[str, ...] = ()

    # Pod sizing
    cpu_per_gpu: int = 8
    memory_gb_per_gpu: int = 64
    shm_size: str = "16Gi"
    hf_secret: str = "hf-secret"
    startup_minutes: int = 30

    # Routing
    gateway: bool = True
    labels: dict[str, str] = field(default_factory=dict)


def sglang_args(cfg: ServingConfig) -> list[str]:
    """The ``python3 -m sglang.launch_server`` arguments for a config."""
    args = [
        "--model-path",
        MODEL_DIR,
        "--served-model-name",
        cfg.served_model_name,
        "--host",
        "0.0.0.0",
        "--port",
        str(PORT),
        "--tp-size",
        str(cfg.tp),
        "--enable-metrics",
    ]
    if cfg.reasoning_parser:
        args += ["--reasoning-parser", cfg.reasoning_parser]
    if cfg.context_length is not None:
        args += ["--context-length", str(cfg.context_length)]
    if cfg.mem_fraction_static is not None:
        args += ["--mem-fraction-static", str(cfg.mem_fraction_static)]
    if cfg.kv_cache_dtype:
        args += ["--kv-cache-dtype", cfg.kv_cache_dtype]
    if cfg.chunked_prefill_size is not None:
        args += ["--chunked-prefill-size", str(cfg.chunked_prefill_size)]
    if cfg.max_running_requests is not None:
        args += ["--max-running-requests", str(cfg.max_running_requests)]
    if cfg.schedule_policy:
        args += ["--schedule-policy", cfg.schedule_policy]
    if cfg.cuda_graph_max_bs is not None:
        args += ["--cuda-graph-max-bs-decode", str(cfg.cuda_graph_max_bs)]
    if cfg.hicache:
        args += ["--enable-hierarchical-cache"]
        if cfg.hicache_size_gb is not None:
            args += ["--hicache-size", str(cfg.hicache_size_gb)]
        elif cfg.hicache_ratio is not None:
            args += ["--hicache-ratio", str(cfg.hicache_ratio)]
    if cfg.page_size is not None:
        args += ["--page-size", str(cfg.page_size)]
    args += list(cfg.extra_args)
    return args


#: ``sglang_args`` flags that are the same for every config.
_FIXED_FLAGS = {"--model-path": MODEL_DIR, "--host": "0.0.0.0", "--port": str(PORT)}
#: flag -> (ServingConfig field, type) for the flags that carry a value.
_VALUE_FLAGS = {
    "--served-model-name": ("served_model_name", str),
    "--tp-size": ("tp", int),
    "--reasoning-parser": ("reasoning_parser", str),
    "--context-length": ("context_length", int),
    "--mem-fraction-static": ("mem_fraction_static", float),
    "--kv-cache-dtype": ("kv_cache_dtype", str),
    "--chunked-prefill-size": ("chunked_prefill_size", int),
    "--max-running-requests": ("max_running_requests", int),
    "--schedule-policy": ("schedule_policy", str),
    "--cuda-graph-max-bs-decode": ("cuda_graph_max_bs", int),
    "--hicache-size": ("hicache_size_gb", int),
    "--hicache-ratio": ("hicache_ratio", float),
    "--page-size": ("page_size", int),
}


def config_from_container(container: dict, model_uri: str | None = None) -> dict:
    """The ServingConfig fields a running ``main`` container was started with: the inverse of
    :func:`sglang_args` plus the pod sizing from its resources. Unknown flags go to
    ``extra_args``, so ``sglang_args(ServingConfig(**fields))`` reproduces the arguments."""
    args = list(container.get("args", []))
    out: dict = {"reasoning_parser": None}  # absent flag = no parser
    extra: list[str] = []
    i = 0
    while i < len(args):
        flag = args[i]
        value = args[i + 1] if i + 1 < len(args) else None
        if flag in _FIXED_FLAGS and _FIXED_FLAGS[flag] == value:
            i += 2
        elif flag in _VALUE_FLAGS and value is not None:
            field_name, typ = _VALUE_FLAGS[flag]
            out[field_name] = typ(value)
            i += 2
        elif flag == "--enable-metrics":
            i += 1
        elif flag == "--enable-hierarchical-cache":
            out["hicache"] = True
            i += 1
        else:
            extra.append(flag)
            i += 1
    if extra:
        out["extra_args"] = tuple(extra)

    tp = out.get("tp", 1)
    limits = container.get("resources", {}).get("limits", {})
    if "cpu" in limits:
        out["cpu_per_gpu"] = int(str(limits["cpu"])) // tp
    if str(limits.get("memory", "")).endswith("Gi"):
        out["memory_gb_per_gpu"] = int(str(limits["memory"])[:-2]) // tp
    if model_uri:
        out["model_uri"] = model_uri
    return out


def render(prof: Profile, cfg: ServingConfig) -> dict:
    """Build the ``LLMInferenceService`` document as a plain dict."""
    cpu = str(cfg.cpu_per_gpu * cfg.tp)
    memory = f"{cfg.memory_gb_per_gpu * cfg.tp}Gi"
    env = [
        {
            "name": "HF_TOKEN",
            "valueFrom": {"secretKeyRef": {"name": cfg.hf_secret, "key": "HF_TOKEN", "optional": True}},
        },
        # The weights are already on disk; never let the server try the Hub at startup.
        {"name": "HF_HUB_OFFLINE", "value": "1"},
    ]
    env += [{"name": k, "value": v} for k, v in prof.env.items()]

    probe = {"httpGet": {"path": "/health", "port": PORT}}
    container = {
        "name": "main",
        "image": prof.image,
        "command": ["python3", "-m", "sglang.launch_server"],
        "args": sglang_args(cfg),
        "env": env,
        "ports": [{"name": "http", "containerPort": PORT, "protocol": "TCP"}],
        # The SGLang images run as root and write caches to the container
        # filesystem, so relax KServe's hardened defaults for this container.
        "securityContext": {"runAsNonRoot": False, "runAsUser": 0, "readOnlyRootFilesystem": False},
        "resources": {
            "limits": {prof.gpu_resource: str(cfg.tp), "cpu": cpu, "memory": memory},
            "requests": {prof.gpu_resource: str(cfg.tp), "cpu": cpu, "memory": memory},
        },
        # /health on SGLang runs a tiny generation, so give probes a few seconds.
        # Model load plus graph capture takes minutes; the startup probe covers that.
        "startupProbe": {
            **probe,
            "periodSeconds": 10,
            "timeoutSeconds": 10,
            "failureThreshold": max(1, cfg.startup_minutes * 6),
        },
        "readinessProbe": {**probe, "periodSeconds": 10, "timeoutSeconds": 8, "failureThreshold": 3},
        "livenessProbe": {**probe, "periodSeconds": 30, "timeoutSeconds": 10, "failureThreshold": 5},
    }

    doc = {
        "apiVersion": "serving.kserve.io/v1alpha1",
        "kind": "LLMInferenceService",
        "metadata": {
            "name": cfg.name,
            "namespace": cfg.namespace,
            "labels": {"workshop/vendor": prof.key, "workshop/tp": str(cfg.tp), **cfg.labels},
        },
        "spec": {
            "model": {"uri": cfg.model_uri, "name": cfg.served_model_name},
            "replicas": cfg.replicas,
            "template": {
                "containers": [container],
                # Tensor parallel ranks exchange activations through shared memory;
                # the base template's 1Gi /dev/shm is too small once tp > 1.
                "volumes": [{"name": "dshm", "emptyDir": {"medium": "Memory", "sizeLimit": cfg.shm_size}}],
                "nodeSelector": prof.node_selector,
            },
        },
    }
    if cfg.gateway:
        # Managed Gateway + HTTPRoute. The endpoint picker (scheduler) is left out:
        # with one replica there is nothing to pick, and its default scorers read
        # vLLM metric names.
        doc["spec"]["router"] = {"gateway": {}, "route": {}}
    return doc


class _PlainDumper(yaml.SafeDumper):
    """No anchors/aliases: repeated probe dicts are written out in full."""

    def ignore_aliases(self, data):  # noqa: D401
        return True


def to_yaml(doc: dict, header: str | None = None) -> str:
    """Serialize a document, optionally with a leading comment block."""
    body = yaml.dump(doc, Dumper=_PlainDumper, sort_keys=False, width=100)
    if header:
        lines = [f"# {line}".rstrip() for line in header.strip("\n").splitlines()]
        return "\n".join(lines) + "\n" + body
    return body
