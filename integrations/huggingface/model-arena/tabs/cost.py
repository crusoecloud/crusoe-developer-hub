"""
Tab 2: Cost — Real economics of open-source vs proprietary LLMs.
Interactive cost calculator with volume scaling.
"""

import streamlit as st
import plotly.graph_objects as go

from core.models import CRUSOE_MODELS, PROPRIETARY_MODELS, get_model_info
from core.cost_engine import compare_all_providers, format_cost, calculate_query_cost


def render():
    st.markdown("### The real cost of open-source vs. proprietary LLMs")
    st.markdown(
        "Configure your workload and see how Crusoe Managed AI compares "
        "to proprietary APIs at scale. No hidden fees, no surprises."
    )

    # Workload configuration
    col1, col2 = st.columns(2)

    with col1:
        input_tokens = st.slider(
            "Avg input tokens per query",
            min_value=50,
            max_value=4000,
            value=500,
            step=50,
            help="Average number of tokens in your prompt (including system prompt)",
        )
        output_tokens = st.slider(
            "Avg output tokens per query",
            min_value=50,
            max_value=4000,
            value=300,
            step=50,
            help="Average number of tokens in the model response",
        )

    with col2:
        queries_per_day = st.slider(
            "Queries per day",
            min_value=100,
            max_value=1_000_000,
            value=10_000,
            step=1000,
            format="%d",
            help="Total API calls per day across your application",
        )
        st.markdown("")
        st.markdown("")
        view_period = st.radio(
            "View costs by",
            ["Per Query", "Daily", "Monthly", "Yearly"],
            index=2,
            horizontal=True,
        )

    # Select which Crusoe models to include
    crusoe_names = {m.display_name: mid for mid, m in CRUSOE_MODELS.items()}
    selected_crusoe = st.multiselect(
        "Crusoe Managed AI models to compare",
        options=list(crusoe_names.keys()),
        default=list(crusoe_names.keys())[:4],
    )
    selected_crusoe_ids = [crusoe_names[n] for n in selected_crusoe]

    if not selected_crusoe_ids:
        st.info("Select at least one Crusoe model to compare.")
        return

    # Calculate costs
    results = compare_all_providers(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        queries_per_day=queries_per_day,
        crusoe_model_ids=selected_crusoe_ids,
    )

    if not results:
        st.warning("No cost data available.")
        return

    # Determine which cost field to show
    period_key = {
        "Per Query": "cost_per_query",
        "Daily": "daily",
        "Monthly": "monthly",
        "Yearly": "yearly",
    }[view_period]

    st.markdown("---")

    # Bar chart
    fig = go.Figure()

    colors = []
    for r in results:
        if r["provider_type"] == "crusoe":
            colors.append("#00C896")  # Crusoe green
        else:
            colors.append("#6C7A89")  # Grey for proprietary

    fig.add_trace(go.Bar(
        x=[r["model_name"] for r in results],
        y=[r[period_key] for r in results],
        marker_color=colors,
        text=[format_cost(r[period_key]) for r in results],
        textposition="outside",
        hovertemplate=(
            "<b>%{x}</b><br>"
            f"{view_period}: %{{text}}<br>"
            "<extra></extra>"
        ),
    ))

    fig.update_layout(
        title=f"{view_period} Cost Comparison",
        xaxis_title="",
        yaxis_title=f"Cost ({view_period})",
        yaxis_tickprefix="$",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial"),
        height=450,
        margin=dict(t=60, b=40),
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Legend
    col_leg1, col_leg2 = st.columns(2)
    with col_leg1:
        st.markdown("🟢 **Crusoe Managed AI** (open-source models)")
    with col_leg2:
        st.markdown("⬜ **Proprietary APIs** (OpenAI, Anthropic)")

    st.markdown("---")

    # Detailed table
    st.markdown("### Detailed Breakdown")

    for r in results:
        provider_badge = "🟢" if r["provider_type"] == "crusoe" else "⬜"
        with st.expander(f"{provider_badge} {r['model_name']} — {format_cost(r[period_key])} {view_period.lower()}"):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Per Query", format_cost(r["cost_per_query"]))
            c2.metric("Daily", format_cost(r["daily"]))
            c3.metric("Monthly", format_cost(r["monthly"]))
            c4.metric("Yearly", format_cost(r["yearly"]))

            info = get_model_info(
                next((mid for mid, m in CRUSOE_MODELS.items() if m.display_name == r["model_name"]), None)
                or next((mid for mid, m in PROPRIETARY_MODELS.items() if m.display_name == r["model_name"]), "")
            )
            if info:
                st.caption(f"{info.provider} · {info.param_count} · {info.description}")

    # Savings summary
    st.markdown("---")
    st.markdown("### Potential Savings")

    crusoe_results = [r for r in results if r["provider_type"] == "crusoe"]
    prop_results = [r for r in results if r["provider_type"] == "proprietary"]

    if crusoe_results and prop_results:
        cheapest_crusoe = min(crusoe_results, key=lambda x: x["monthly"])
        avg_proprietary = sum(r["monthly"] for r in prop_results) / len(prop_results)

        if avg_proprietary > 0:
            savings_pct = ((avg_proprietary - cheapest_crusoe["monthly"]) / avg_proprietary) * 100
            monthly_savings = avg_proprietary - cheapest_crusoe["monthly"]

            col_s1, col_s2, col_s3 = st.columns(3)
            col_s1.metric(
                "Cheapest Crusoe Model",
                cheapest_crusoe["model_name"],
                f"{format_cost(cheapest_crusoe['monthly'])}/mo",
            )
            col_s2.metric(
                "Avg Proprietary Cost",
                format_cost(avg_proprietary) + "/mo",
            )
            col_s3.metric(
                "Monthly Savings",
                format_cost(monthly_savings),
                f"{savings_pct:.0f}% less",
            )

    st.caption(
        "Pricing is estimated and may vary. Crusoe Managed AI prices are based on "
        "current published rates. Proprietary API prices are based on publicly listed "
        "pricing as of March 2026. Always verify current rates before making decisions."
    )
