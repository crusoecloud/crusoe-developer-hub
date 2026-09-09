"""
Model registry for Crusoe Managed AI and proprietary provider pricing.
All models available on Crusoe MI + comparison pricing for OpenAI and Anthropic.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelInfo:
    id: str
    display_name: str
    provider: str
    param_count: str
    input_price_per_1m: float   # USD per 1M input tokens
    output_price_per_1m: float  # USD per 1M output tokens
    description: str
    strengths: list[str]



# Crusoe Managed AI models
CRUSOE_MODELS: dict[str, ModelInfo] = {
    "meta-llama/Llama-3.3-70B-Instruct": ModelInfo(
        id="meta-llama/Llama-3.3-70B-Instruct",
        display_name="Llama 3.3 70B",
        provider="Meta",
        param_count="70B",
        input_price_per_1m=0.25,
        output_price_per_1m=0.75,
        description="Meta's flagship open-source instruction-tuned model",
        strengths=["General purpose", "Code generation", "Instruction following"],
    ),
    "deepseek-ai/DeepSeek-R1-0528": ModelInfo(
        id="deepseek-ai/DeepSeek-R1-0528",
        display_name="DeepSeek R1",
        provider="DeepSeek",
        param_count="671B MoE",
        input_price_per_1m=1.35,
        output_price_per_1m=5.40,
        description="DeepSeek's reasoning-focused model with chain-of-thought",
        strengths=["Multi-step reasoning", "Math", "Complex analysis"],
    ),
    "deepseek-ai/DeepSeek-V3-0324": ModelInfo(
        id="deepseek-ai/DeepSeek-V3-0324",
        display_name="DeepSeek V3",
        provider="DeepSeek",
        param_count="671B MoE",
        input_price_per_1m=0.50,
        output_price_per_1m=1.50,
        description="DeepSeek's general-purpose high-performance model",
        strengths=["Balanced quality/cost", "Summarization", "Code"],
    ),
    "deepseek-ai/Deepseek-V4-Flash": ModelInfo(
        id="deepseek-ai/Deepseek-V4-Flash",
        display_name="DeepSeek V4 Flash",
        provider="DeepSeek",
        param_count="TBC MoE",
        input_price_per_1m=0.15,
        output_price_per_1m=0.50,
        description="DeepSeek V4 Flash, the fast low-latency variant for planning and high-throughput agent loops",
        strengths=["Low latency", "Planning", "Throughput", "Cost efficient"],
    ),
    "deepseek-ai/DeepSeek-V4-Pro": ModelInfo(
        id="deepseek-ai/DeepSeek-V4-Pro",
        display_name="DeepSeek V4 Pro",
        provider="DeepSeek",
        param_count="671B MoE",
        input_price_per_1m=0.60,
        output_price_per_1m=2.00,
        description="DeepSeek V4 Pro, the high-capability variant for deep reasoning, complex code, and tool-calling agents",
        strengths=["Deep reasoning", "Code generation", "Tool calls", "Long context"],
    ),
    "Qwen/Qwen3-235B-A22B-Instruct-2507": ModelInfo(
        id="Qwen/Qwen3-235B-A22B-Instruct-2507",
        display_name="Qwen3 235B",
        provider="Alibaba",
        param_count="235B MoE",
        input_price_per_1m=0.22,
        output_price_per_1m=0.80,
        description="Alibaba's large-scale multilingual instruction model",
        strengths=["Multilingual", "Structured extraction", "Reasoning"],
    ),
    "moonshotai/Kimi-K2-Thinking": ModelInfo(
        id="moonshotai/Kimi-K2-Thinking",
        display_name="Kimi K2",
        provider="Moonshot",
        param_count="1T MoE",
        input_price_per_1m=0.60,
        output_price_per_1m=2.50,
        description="Moonshot's thinking-optimized reasoning model",
        strengths=["Deep reasoning", "Long context", "Analysis"],
    ),
    "google/gemma-3-12b-it": ModelInfo(
        id="google/gemma-3-12b-it",
        display_name="Gemma 3 12B",
        provider="Google",
        param_count="12B",
        input_price_per_1m=0.08,
        output_price_per_1m=0.30,
        description="Google's efficient small model — fast and low cost",
        strengths=["Low latency", "Cost efficient", "Fast triage"],
    ),
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B": ModelInfo(
        id="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B",
        display_name="Nemotron 3 Super 120B",
        provider="NVIDIA",
        param_count="120B MoE",
        input_price_per_1m=0.30,
        output_price_per_1m=2.40,
        description="NVIDIA's high-capability sparse MoE model",
        strengths=["Reasoning", "Code generation", "Instruction following"],
    ),
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B": ModelInfo(
        id="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
        display_name="Nemotron 3 Nano 30B",
        provider="NVIDIA",
        param_count="30B MoE",
        input_price_per_1m=0.05,
        output_price_per_1m=0.20,
        description="NVIDIA's efficient small MoE model, ultra low cost",
        strengths=["Low latency", "Cost efficient", "Edge deployment"],
    ),
    "nvidia/Nemotron-3-Nano-Omni-Reasoning-30B-A3B": ModelInfo(
        id="nvidia/Nemotron-3-Nano-Omni-Reasoning-30B-A3B",
        display_name="Nemotron 3 Nano Omni Reasoning 30B",
        provider="NVIDIA",
        param_count="30B MoE A3B",
        input_price_per_1m=0.08,
        output_price_per_1m=0.40,
        description="Crusoe-served Nemotron Nano variant with omni-modal inputs and chain-of-thought reasoning",
        strengths=["Reasoning", "Multimodal", "Low latency", "Cost efficient"],
    ),
    "openai/gpt-oss-120b": ModelInfo(
        id="openai/gpt-oss-120b",
        display_name="GPT-OSS 120B",
        provider="OpenAI",
        param_count="120B",
        input_price_per_1m=0.15,
        output_price_per_1m=0.60,
        description="OpenAI's open-source model release",
        strengths=["General purpose", "Instruction following", "Code"],
    ),
}

# Proprietary models for cost comparison
PROPRIETARY_MODELS: dict[str, ModelInfo] = {
    "gpt-4o": ModelInfo(
        id="gpt-4o",
        display_name="GPT-4o",
        provider="OpenAI",
        param_count="N/A",
        input_price_per_1m=2.50,
        output_price_per_1m=10.00,
        description="OpenAI's flagship multimodal model",
        strengths=["General purpose", "Multimodal", "Code"],
    ),
    "gpt-4o-mini": ModelInfo(
        id="gpt-4o-mini",
        display_name="GPT-4o Mini",
        provider="OpenAI",
        param_count="N/A",
        input_price_per_1m=0.15,
        output_price_per_1m=0.60,
        description="OpenAI's cost-efficient model",
        strengths=["Fast", "Cost efficient", "Good enough for many tasks"],
    ),
    "claude-sonnet-4": ModelInfo(
        id="claude-sonnet-4",
        display_name="Claude Sonnet 4",
        provider="Anthropic",
        param_count="N/A",
        input_price_per_1m=3.00,
        output_price_per_1m=15.00,
        description="Anthropic's balanced performance model",
        strengths=["Reasoning", "Code", "Long context"],
    ),
    "claude-haiku-3.5": ModelInfo(
        id="claude-haiku-3.5",
        display_name="Claude Haiku 3.5",
        provider="Anthropic",
        param_count="N/A",
        input_price_per_1m=0.80,
        output_price_per_1m=4.00,
        description="Anthropic's fast, cost-efficient model",
        strengths=["Speed", "Cost efficient", "Classification"],
    ),
}


def get_crusoe_model_ids() -> list[str]:
    return list(CRUSOE_MODELS.keys())


def get_crusoe_display_names() -> dict[str, str]:
    return {m.display_name: mid for mid, m in CRUSOE_MODELS.items()}


def get_model_info(model_id: str) -> Optional[ModelInfo]:
    return CRUSOE_MODELS.get(model_id) or PROPRIETARY_MODELS.get(model_id)
