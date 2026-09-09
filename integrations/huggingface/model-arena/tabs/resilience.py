"""
Tab 3: Resilience — Failover simulator.
Demonstrates what happens when an API endpoint goes down
and the agent reroutes to a fallback model on Crusoe Managed AI.
"""

import streamlit as st
import time
from concurrent.futures import ThreadPoolExecutor

from core.models import CRUSOE_MODELS, get_model_info
from core.inference import run_inference, InferenceResult


PIPELINE_STEPS = {
    "Summarize": "Summarize the following text in 2-3 concise sentences.",
    "Classify": "Classify the sentiment of the following text as positive, negative, or neutral. Respond with just the classification and a one-sentence explanation.",
    "Extract": "Extract all key entities (people, organizations, locations, dates) from the following text. Return them as a structured list.",
}

SAMPLE_TEXT = """
Crusoe AI announced a major expansion of their Managed Inference platform today,
adding support for seven open-source LLMs including Meta's Llama 3.3, DeepSeek-R1,
and Google's Gemma 3. The San Francisco-based company, led by CEO Chase Lochmiller,
said the update positions their neocloud infrastructure as the go-to platform for
developers building multi-model AI agents. Industry analysts from Gartner noted that
the move reflects a broader shift away from single-vendor LLM dependence toward
resilient, multi-provider architectures. The update is available immediately through
Crusoe's API at no additional cost to existing customers.
"""


def render():
    st.markdown("### What happens when your model endpoint goes down?")
    st.markdown(
        "Configure a 3-step agent pipeline, assign models to each step, "
        "then simulate an outage. Watch the pipeline reroute to a fallback "
        "model on Crusoe Managed AI in real time."
    )

    model_options = {m.display_name: mid for mid, m in CRUSOE_MODELS.items()}
    display_names = list(model_options.keys())

    # Pipeline configuration
    st.markdown("---")
    st.markdown("### Configure Your Pipeline")

    # Input text
    input_text = st.text_area(
        "Input text for the pipeline",
        value=SAMPLE_TEXT.strip(),
        height=120,
    )

    # Step configuration
    step_cols = st.columns(3)
    step_models = {}
    step_fallbacks = {}

    for i, (step_name, _) in enumerate(PIPELINE_STEPS.items()):
        with step_cols[i]:
            st.markdown(f"**Step {i+1}: {step_name}**")
            step_models[step_name] = st.selectbox(
                f"Primary model",
                options=display_names,
                index=min(i * 2, len(display_names) - 1),
                key=f"primary_{step_name}",
            )
            step_fallbacks[step_name] = st.selectbox(
                f"Fallback model",
                options=display_names,
                index=min(i * 2 + 1, len(display_names) - 1),
                key=f"fallback_{step_name}",
            )

    st.markdown("---")

    # Simulation controls
    st.markdown("### Simulate Outage")

    kill_options = ["None"] + list(PIPELINE_STEPS.keys())
    kill_step = st.selectbox(
        "Which step's primary model should go down?",
        options=kill_options,
        index=0,
        help="Select a pipeline step to simulate its primary model becoming unavailable.",
    )

    col_run, col_info = st.columns([1, 2])
    with col_run:
        run_pipeline = st.button("Run Pipeline", type="primary", use_container_width=True)
    with col_info:
        if kill_step != "None":
            killed_model = step_models[kill_step]
            fallback_model = step_fallbacks[kill_step]
            st.warning(
                f"Outage simulated on **{killed_model}** at Step: {kill_step}. "
                f"Pipeline will reroute to **{fallback_model}**."
            )

    if run_pipeline and input_text.strip():
        st.markdown("---")
        st.markdown("### Pipeline Execution")

        pipeline_input = input_text.strip()
        total_start = time.perf_counter()
        step_results = {}

        for step_idx, (step_name, system_prompt) in enumerate(PIPELINE_STEPS.items()):
            st.markdown(f"#### Step {step_idx + 1}: {step_name}")

            primary_name = step_models[step_name]
            fallback_name = step_fallbacks[step_name]
            primary_id = model_options[primary_name]
            fallback_id = model_options[fallback_name]

            is_killed = (kill_step == step_name)
            used_fallback = False
            result = None

            col_status, col_detail = st.columns([1, 3])

            if is_killed:
                # Simulate the outage
                with col_status:
                    st.error("PRIMARY DOWN")
                with col_detail:
                    st.markdown(
                        f"~~{primary_name}~~ — endpoint unavailable (simulated outage)  \n"
                        f"Rerouting to **{fallback_name}**..."
                    )

                # Run on fallback
                result = run_inference(
                    model_id=fallback_id,
                    prompt=pipeline_input,
                    system_prompt=system_prompt,
                    max_tokens=512,
                    temperature=0.3,
                )
                used_fallback = True
            else:
                # Run on primary
                with col_status:
                    with st.spinner(f"Running {primary_name}..."):
                        result = run_inference(
                            model_id=primary_id,
                            prompt=pipeline_input,
                            system_prompt=system_prompt,
                            max_tokens=512,
                            temperature=0.3,
                        )

                if result.error:
                    # Real error — try fallback
                    with col_status:
                        st.warning("PRIMARY FAILED")
                    with col_detail:
                        st.markdown(
                            f"**{primary_name}** returned error: `{result.error}`  \n"
                            f"Rerouting to **{fallback_name}**..."
                        )
                    result = run_inference(
                        model_id=fallback_id,
                        prompt=pipeline_input,
                        system_prompt=system_prompt,
                        max_tokens=512,
                        temperature=0.3,
                    )
                    used_fallback = True
                else:
                    with col_status:
                        st.success("OK")

            # Display result
            if result and not result.error:
                model_used = fallback_name if used_fallback else primary_name
                badge = "🔄 FALLBACK" if used_fallback else "✅ PRIMARY"

                st.markdown(f"**{badge}** — {model_used}")

                met1, met2, met3 = st.columns(3)
                met1.metric("Latency", f"{result.total_latency:.2f}s")
                met2.metric("Output Tokens", result.output_tokens)
                met3.metric("Tokens/sec", f"{result.tokens_per_second:.0f}")

                with st.expander("View response", expanded=(step_idx == 0)):
                    st.markdown(result.response_text)

                step_results[step_name] = {
                    "result": result,
                    "used_fallback": used_fallback,
                    "model_used": model_used,
                }

                # Feed output to next step
                pipeline_input = result.response_text

            elif result and result.error:
                st.error(f"Both primary and fallback failed: {result.error}")
                st.stop()

        # Pipeline summary
        total_time = time.perf_counter() - total_start

        st.markdown("---")
        st.markdown("### Pipeline Summary")

        sum_cols = st.columns(4)
        sum_cols[0].metric("Total Pipeline Time", f"{total_time:.2f}s")
        sum_cols[1].metric("Steps Completed", f"{len(step_results)}/3")
        fallback_count = sum(1 for s in step_results.values() if s["used_fallback"])
        sum_cols[2].metric("Fallbacks Triggered", fallback_count)
        sum_cols[3].metric(
            "Pipeline Status",
            "PASSED" if len(step_results) == 3 else "FAILED",
        )

        if kill_step != "None" and len(step_results) == 3:
            st.success(
                "The pipeline completed successfully despite the simulated outage. "
                "Multi-model architecture with automatic failover kept the agent running. "
                "This is what resilient agentic engineering looks like on Crusoe Managed AI."
            )
        elif kill_step == "None" and len(step_results) == 3:
            st.info(
                "Pipeline completed with all primary models. "
                "Try simulating an outage to see how failover works!"
            )

        # Comparison table if fallback was used
        if fallback_count > 0:
            st.markdown("---")
            st.markdown("### Failover Impact Analysis")

            for step_name, data in step_results.items():
                if data["used_fallback"]:
                    primary_name = step_models[step_name]
                    fallback_name = step_fallbacks[step_name]
                    st.markdown(f"**{step_name}:** {primary_name} → {fallback_name}")

                    # Run primary for comparison (if it wasn't a simulated kill)
                    st.caption(
                        f"The fallback model ({fallback_name}) completed in "
                        f"{data['result'].total_latency:.2f}s with "
                        f"{data['result'].output_tokens} tokens. "
                        f"In a production system, this reroute happens automatically "
                        f"with no user-facing downtime."
                    )
