"""
Model Arena — Compare, Cost, and Crash-Test Open-Source LLMs on Crusoe Managed AI

A HuggingFace Spaces app that lets developers:
1. Compare open-source models head-to-head (Arena)
2. See real cost economics vs proprietary APIs (Cost)
3. Simulate endpoint outages and watch failover in action (Resilience)

All inference powered by Crusoe Managed AI.
"""

import streamlit as st
import os

# Page config
st.set_page_config(
    page_title="Model Arena | Crusoe Managed AI",
    page_icon="⚔️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Header
st.markdown(
    """
    <div style="text-align: center; padding: 1rem 0 0.5rem 0;">
        <h1 style="margin-bottom: 0.2rem;">⚔️ Model Arena</h1>
        <p style="color: #666; font-size: 1.1rem; margin-top: 0;">
            Compare, Cost, and Crash-Test Open-Source LLMs on
            <strong>Crusoe Managed AI</strong>
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# API key check
api_key = os.environ.get("CRUSOE_API_KEY", "")
if not api_key:
    st.warning(
        "**CRUSOE_API_KEY not set.** "
        "Add your Crusoe Managed AI API key as an environment variable "
        "or HuggingFace Spaces secret to enable live inference."
    )
    st.markdown(
        "```bash\n"
        "export CRUSOE_API_KEY=your_api_key_here\n"
        "```"
    )
    st.stop()

# Tabs
tab_arena, tab_cost, tab_resilience = st.tabs([
    "⚔️ Arena",
    "💰 Cost",
    "🛡️ Resilience",
])

with tab_arena:
    from tabs.arena import render as render_arena
    render_arena()

with tab_cost:
    from tabs.cost import render as render_cost
    render_cost()

with tab_resilience:
    from tabs.resilience import render as render_resilience
    render_resilience()

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; padding: 0.5rem 0; color: #888; font-size: 0.85rem;">
        Built by <strong>Crusoe AI</strong> Developer Relations ·
        Powered by <strong>Crusoe Managed AI</strong> ·
        All inference runs on purpose-built neocloud GPU infrastructure<br>
        <a href="https://crusoe.ai" target="_blank" style="color: #00C896;">crusoe.ai</a>
    </div>
    """,
    unsafe_allow_html=True,
)
