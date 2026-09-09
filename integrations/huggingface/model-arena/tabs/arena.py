"""
Tab 1: Arena — Head-to-head model comparison.
Users select 2-3 models, enter a prompt, and watch them respond side by side.
"""

import streamlit as st
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.models import CRUSOE_MODELS, get_model_info
from core.inference import run_inference, InferenceResult
from core.leaderboard import record_vote, get_leaderboard, get_total_votes


def render():
    st.markdown("### Compare open-source LLMs head-to-head on Crusoe Managed AI")
    st.markdown(
        "Select models, enter a prompt, and see how they perform side by side "
        "— latency, token count, and response quality, all powered by Crusoe Managed AI."
    )

    # Model selection
    model_options = {m.display_name: mid for mid, m in CRUSOE_MODELS.items()}
    display_names = list(model_options.keys())

    selected_names = st.multiselect(
        "Select 2-3 models to compare",
        options=display_names,
        default=display_names[:2],
        max_selections=3,
    )

    if len(selected_names) < 2:
        st.info("Please select at least 2 models to compare.")
        return

    selected_model_ids = [model_options[name] for name in selected_names]

    # Prompt input
    prompt = st.text_area(
        "Enter your prompt",
        placeholder="e.g., Explain the difference between TCP and UDP in a way a junior developer would understand.",
        height=100,
    )

    col_run, col_temp = st.columns([1, 1])
    with col_temp:
        temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1)

    with col_run:
        run_button = st.button("Run Arena", type="primary", use_container_width=True)

    if run_button and prompt.strip():
        st.markdown("---")

        # Create columns for side-by-side display
        cols = st.columns(len(selected_model_ids))
        placeholders = {}
        metric_placeholders = {}

        for i, model_id in enumerate(selected_model_ids):
            info = get_model_info(model_id)
            with cols[i]:
                st.markdown(f"**{info.display_name}**")
                st.caption(f"{info.provider} · {info.param_count}")
                metric_placeholders[model_id] = st.empty()
                placeholders[model_id] = st.empty()
                placeholders[model_id].markdown("_Waiting for response..._")

        # Run inference in parallel using threads
        results: dict[str, InferenceResult] = {}

        def run_model(model_id):
            return run_inference(
                model_id=model_id,
                prompt=prompt,
                max_tokens=1024,
                temperature=temperature,
            )

        with ThreadPoolExecutor(max_workers=len(selected_model_ids)) as executor:
            future_to_model = {
                executor.submit(run_model, mid): mid for mid in selected_model_ids
            }
            for future in as_completed(future_to_model):
                model_id = future_to_model[future]
                result = future.result()
                results[model_id] = result

                # Update the UI
                info = get_model_info(model_id)
                idx = selected_model_ids.index(model_id)

                with cols[idx]:
                    if result.error:
                        placeholders[model_id].error(f"Error: {result.error}")
                        metric_placeholders[model_id].empty()
                    else:
                        metric_placeholders[model_id].markdown(
                            f"**{result.total_latency:.2f}s** · "
                            f"{result.output_tokens} tokens · "
                            f"{result.tokens_per_second:.0f} tok/s"
                        )
                        placeholders[model_id].markdown(result.response_text)

        # Store results in session state for voting
        st.session_state["arena_results"] = results
        st.session_state["arena_prompt"] = prompt
        st.session_state["arena_models"] = selected_model_ids

    # Voting section
    if "arena_results" in st.session_state and st.session_state["arena_results"]:
        results = st.session_state["arena_results"]
        valid_results = {k: v for k, v in results.items() if not v.error}

        if len(valid_results) >= 2:
            st.markdown("---")
            st.markdown("### Vote for the best response")

            vote_cols = st.columns(len(valid_results))
            for i, (model_id, result) in enumerate(valid_results.items()):
                info = get_model_info(model_id)
                with vote_cols[i]:
                    if st.button(
                        f"Vote {info.display_name}",
                        key=f"vote_{model_id}",
                        use_container_width=True,
                    ):
                        loser_ids = [m for m in valid_results if m != model_id]
                        record_vote(
                            winner_id=model_id,
                            loser_ids=loser_ids,
                            prompt=st.session_state.get("arena_prompt", ""),
                        )
                        st.success(f"Vote recorded for {info.display_name}!")

    # Leaderboard
    st.markdown("---")
    st.markdown("### Community Leaderboard")

    total = get_total_votes()
    board = get_leaderboard()

    if board:
        st.caption(f"{total} total votes from the community")
        for rank, entry in enumerate(board, 1):
            info = get_model_info(entry["model_id"])
            name = info.display_name if info else entry["model_id"]
            win_pct = entry["win_rate"] * 100

            col_rank, col_name, col_stats = st.columns([0.5, 2, 3])
            with col_rank:
                if rank == 1:
                    st.markdown(f"**#{rank}**")
                else:
                    st.markdown(f"#{rank}")
            with col_name:
                st.markdown(f"**{name}**")
            with col_stats:
                st.progress(entry["win_rate"])
                st.caption(
                    f"{win_pct:.0f}% win rate · "
                    f"{entry['wins']}W / {entry['losses']}L · "
                    f"{entry['total_matchups']} matchups"
                )
    else:
        st.info("No votes yet. Run a comparison and cast the first vote!")
