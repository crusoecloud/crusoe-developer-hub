"""Vendor profiles.

Everything that differs between serving on an NVIDIA node and an AMD node on
Crusoe Managed Kubernetes fits in one small record:

- the SGLang container image (CUDA build or ROCm build)
- the Kubernetes GPU resource name (``nvidia.com/gpu`` or ``amd.com/gpu``)
- the ``crusoe.ai/accelerator`` node label that pins the pod to the right pool
- a few environment variables the runtime expects

Image tags come from the official SGLang Docker Hub repository
(https://hub.docker.com/r/lmsysorg/sglang/tags). The CUDA image is the
``-runtime`` build (no dev toolchain, same engine). The ROCm images are built
per GPU family: ``mi35x`` for MI355X and ``mi30x`` for MI300X; they have no
runtime variant.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace

#: SGLang release the workshop pins. Bump this in one place to move both vendors.
SGLANG_VERSION = "v0.5.22"

#: ROCm toolchain tag used in the AMD image names.
ROCM_TAG = "rocm724"


@dataclass(frozen=True)
class Profile:
    """The vendor-specific half of a deployment."""

    key: str  # "nvidia" or "amd"
    label: str  # shown in tables and plots
    image: str
    gpu_resource: str  # Kubernetes extended resource name
    accelerator: str  # value of the crusoe.ai/accelerator node label
    gpu_memory_gb: int  # HBM per GPU, for the comparison table
    env: dict[str, str] = field(default_factory=dict)

    @property
    def node_selector(self) -> dict[str, str]:
        return {"crusoe.ai/accelerator": self.accelerator}


#: Crusoe accelerator labels validated with this workshop, with the vendor and HBM size.
#: Any other label works too: the vendor is its first word and the HBM size is read
#: from the label (``nvidia-h100-80gb-pcie``) or from the GPU family (``nvidia-l40s``).
#: List the labels on your cluster with ``kubectl get nodes -L crusoe.ai/accelerator``.
ACCELERATORS: dict[str, tuple[str, int]] = {
    "nvidia-h100-80gb-sxm-ib": ("nvidia", 80),
    "nvidia-h200-141gb-sxm-ib": ("nvidia", 141),
    "nvidia-a100-80gb-sxm-ib": ("nvidia", 80),
    "nvidia-a100-80gb": ("nvidia", 80),
    "nvidia-l40s": ("nvidia", 48),
    "amd-mi355x-288gb-roce": ("amd", 288),
    "amd-mi300x-192gb-ib": ("amd", 192),
}

#: HBM per GPU for labels that do not spell it out.
_FAMILY_MEMORY_GB = {
    "l40s": 48,
    "a100": 80,
    "h100": 80,
    "h200": 141,
    "b200": 180,
    "mi300x": 192,
    "mi325x": 256,
    "mi355x": 288,
}


def parse_accelerator(label: str) -> tuple[str, int]:
    """Vendor and HBM size (GB) of a ``crusoe.ai/accelerator`` label."""
    if label in ACCELERATORS:
        return ACCELERATORS[label]
    parts = label.lower().split("-")
    vendor = parts[0]
    if vendor not in PROFILES or len(parts) < 2:
        raise KeyError(f"cannot read a vendor from accelerator label {label!r}")
    memory = next((int(p[:-2]) for p in parts if re.fullmatch(r"\d+gb", p)), _FAMILY_MEMORY_GB.get(parts[1]))
    if memory is None:
        raise KeyError(
            f"unknown GPU memory for {label!r}; add it to ACCELERATORS in workshop_utils/profiles.py"
        )
    return vendor, memory


def _nvidia_image(version: str = SGLANG_VERSION) -> str:
    # The "-runtime" build is the same release without the dev toolchain; it is
    # the image validated for this workshop (driver 580, CUDA 13) and pulls faster.
    return f"lmsysorg/sglang:{version}-runtime"


def _amd_image(accelerator: str, version: str = SGLANG_VERSION) -> str:
    family = "mi30x" if ("mi300x" in accelerator or "mi325x" in accelerator) else "mi35x"
    return f"lmsysorg/sglang:{version}-{ROCM_TAG}-{family}"


PROFILES: dict[str, Profile] = {
    "nvidia": Profile(
        key="nvidia",
        label="NVIDIA H100 80GB",
        image=_nvidia_image(),
        gpu_resource="nvidia.com/gpu",
        accelerator="nvidia-h100-80gb-sxm-ib",
        gpu_memory_gb=80,
    ),
    "amd": Profile(
        key="amd",
        label="AMD MI355X 288GB",
        image=_amd_image("amd-mi355x-288gb-roce"),
        gpu_resource="amd.com/gpu",
        accelerator="amd-mi355x-288gb-roce",
        gpu_memory_gb=288,
        # AITER kernels are the fast path on Instinct GPUs; SGLang's AMD guide recommends it.
        env={"SGLANG_USE_AITER": "1"},
    ),
}


def profile(key: str, accelerator: str | None = None, image: str | None = None) -> Profile:
    """Return a vendor profile, optionally re-pointed at another Crusoe accelerator.

    ``profile("amd", accelerator="amd-mi300x-192gb-ib")`` swaps the node label,
    the HBM size and the ROCm image family in one go; any label from
    ``kubectl get nodes -L crusoe.ai/accelerator`` is accepted. ``image``
    overrides the container image, for example to pin a CUDA 12 build on
    older drivers.
    """
    if key not in PROFILES:
        raise KeyError(f"unknown vendor {key!r}; choose one of {sorted(PROFILES)}")
    prof = PROFILES[key]
    if accelerator is not None:
        vendor, memory = parse_accelerator(accelerator)
        if vendor != key:
            raise ValueError(f"accelerator {accelerator!r} is a {vendor} SKU, not {key}")
        label = accelerator.split("-")[1].upper() + f" {memory}GB"
        new_image = _amd_image(accelerator) if key == "amd" else prof.image
        prof = replace(
            prof,
            accelerator=accelerator,
            gpu_memory_gb=memory,
            image=new_image,
            label=f"{'AMD' if key == 'amd' else 'NVIDIA'} {label}",
        )
    if image is not None:
        prof = replace(prof, image=image)
    return prof
