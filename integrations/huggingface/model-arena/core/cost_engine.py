"""
Cost calculation engine for comparing Crusoe Managed AI vs proprietary providers.
"""

from dataclasses import dataclass
from typing import List, Optional
from core.models import CRUSOE_MODELS, PROPRIETARY_MODELS, ModelInfo


@dataclass
class CostBreakdown:
    model_name: str
    provider: str
    provider_type: str  # "crusoe" or "proprietary"
    input_cost: float
    output_cost: float
    total_cost: float
    input_tokens: int
    output_tokens: int


def calculate_query_cost(
    model_id: str,
    input_tokens: int,
    output_tokens: int,
) -> Optional[CostBreakdown]:
    """Calculate cost for a single query."""
    model = CRUSOE_MODELS.get(model_id) or PROPRIETARY_MODELS.get(model_id)
    if not model:
        return None

    input_cost = (input_tokens / 1_000_000) * model.input_price_per_1m
    output_cost = (output_tokens / 1_000_000) * model.output_price_per_1m
    provider_type = "crusoe" if model_id in CRUSOE_MODELS else "proprietary"

    return CostBreakdown(
        model_name=model.display_name,
        provider=model.provider,
        provider_type=provider_type,
        input_cost=input_cost,
        output_cost=output_cost,
        total_cost=input_cost + output_cost,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def calculate_volume_cost(
    model_id: str,
    input_tokens_per_query: int,
    output_tokens_per_query: int,
    queries_per_day: int,
) -> dict:
    """Calculate costs at daily/monthly/yearly scale."""
    single = calculate_query_cost(model_id, input_tokens_per_query, output_tokens_per_query)
    if not single:
        return {}

    daily = single.total_cost * queries_per_day
    monthly = daily * 30
    yearly = daily * 365

    return {
        "model_name": single.model_name,
        "provider": single.provider,
        "provider_type": single.provider_type,
        "cost_per_query": single.total_cost,
        "daily": daily,
        "monthly": monthly,
        "yearly": yearly,
    }


def compare_all_providers(
    input_tokens: int,
    output_tokens: int,
    queries_per_day: int,
    crusoe_model_ids: Optional[List[str]] = None,
) -> list[dict]:
    """
    Compare costs across all Crusoe models and proprietary alternatives.
    Returns a sorted list from cheapest to most expensive (monthly).
    """
    results = []

    # Crusoe models
    models_to_check = crusoe_model_ids or list(CRUSOE_MODELS.keys())
    for model_id in models_to_check:
        cost = calculate_volume_cost(model_id, input_tokens, output_tokens, queries_per_day)
        if cost:
            results.append(cost)

    # Proprietary models
    for model_id in PROPRIETARY_MODELS:
        cost = calculate_volume_cost(model_id, input_tokens, output_tokens, queries_per_day)
        if cost:
            results.append(cost)

    results.sort(key=lambda x: x["monthly"])
    return results


def format_cost(amount: float) -> str:
    """Format a dollar amount for display."""
    if amount < 0.01:
        return f"${amount:.4f}"
    elif amount < 1:
        return f"${amount:.3f}"
    elif amount < 1000:
        return f"${amount:.2f}"
    else:
        return f"${amount:,.0f}"
