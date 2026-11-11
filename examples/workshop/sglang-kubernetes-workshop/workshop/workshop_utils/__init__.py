"""Helpers for the workshop notebooks: deploy, chat with, benchmark and tune an SGLang server on Kubernetes.

The package is deliberately small. Each module maps to one thing a participant does:

- ``profiles``  the handful of values that differ between an NVIDIA and an AMD deployment
- ``manifest``  render a KServe ``LLMInferenceService`` for a vendor profile and serving config
- ``k8s``       thin ``kubectl`` wrappers: apply, wait, logs, port-forward
- ``chat``      talk to the OpenAI-compatible endpoint and read SGLang's native endpoints
- ``bench``     run ``sglang.bench_serving`` inside the serving pod and parse the result
- ``plots``     Crusoe-styled comparison charts
- ``service``   the participant's service: deploy, and redeploy after a change (stop-annotation route)
- ``leaderboard`` submit a benchmark result to the workshop leaderboard
- ``chat_ui``   a one-page browser chat: ``python -m workshop_utils.chat_ui <name>``
"""

from .bench import (
    STANDARD,
    BenchResult,
    bench_command,
    load_results,
    parse_result,
    run_bench,
    save_results,
    standard_bench,
)
from .chat import ChatResult, chat, list_models, metrics, server_info
from .k8s import (
    KSERVE_QUEUE_MSG,
    NoNamespaceError,
    PortForward,
    apply,
    current_namespace,
    delete_llmisvc,
    detect_accelerator,
    exists_llmisvc,
    kubectl,
    logs,
    node_gpus,
    participant_namespace,
    wait_ready,
    workload_deployment,
    workshop_accelerator,
)
from .leaderboard import standard_results, submit
from .manifest import ServingConfig, config_from_container, render, sglang_args, to_yaml
from .profiles import ACCELERATORS, PROFILES, SGLANG_VERSION, Profile, parse_accelerator, profile
from .service import (
    PARTICIPANT_SERVICE,
    Current,
    RedeployError,
    current_config,
    deploy,
    participant_config,
    participant_profile,
    redeploy,
    switch_to,
    wait_new_pod,
)

__all__ = [
    "KSERVE_QUEUE_MSG",
    "NoNamespaceError",
    "PARTICIPANT_SERVICE",
    "RedeployError",
    "deploy",
    "participant_config",
    "participant_namespace",
    "participant_profile",
    "redeploy",
    "standard_results",
    "submit",
    "switch_to",
    "Current",
    "config_from_container",
    "current_config",
    "wait_new_pod",
    "STANDARD",
    "standard_bench",
    "ACCELERATORS",
    "BenchResult",
    "ChatResult",
    "PROFILES",
    "PortForward",
    "Profile",
    "SGLANG_VERSION",
    "ServingConfig",
    "apply",
    "bench_command",
    "chat",
    "current_namespace",
    "delete_llmisvc",
    "kubectl",
    "list_models",
    "load_results",
    "logs",
    "metrics",
    "node_gpus",
    "detect_accelerator",
    "exists_llmisvc",
    "parse_accelerator",
    "parse_result",
    "profile",
    "render",
    "run_bench",
    "save_results",
    "server_info",
    "sglang_args",
    "to_yaml",
    "wait_ready",
    "workload_deployment",
    "workshop_accelerator",
]
