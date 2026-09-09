COMPARISON_TEMPLATE = '''import streamlit as st
from openai import OpenAI
import os

TITLE = "{title}"
DESCRIPTION = "{description}"
SYSTEM_PROMPT = """{system_prompt}"""
MODEL_A = "{model_a}"
MODEL_A_LABEL = "{model_a_label}"
MODEL_B = "{model_b}"
MODEL_B_LABEL = "{model_b_label}"

client = OpenAI(
    api_key=os.environ.get("CRUSOE_API_KEY"),
    base_url=os.environ.get("CRUSOE_BASE_URL", "https://api.crusoe.ai/v1"),
)

st.set_page_config(page_title=TITLE, page_icon="⚖️", layout="wide")
st.title(TITLE)
st.caption(DESCRIPTION)
st.divider()

with st.sidebar:
    st.caption("Powered by [Crusoe](https://crusoe.ai)")
    st.markdown(f"**Model A:** `{{MODEL_A}}`")
    st.markdown(f"**Model B:** `{{MODEL_B}}`")

prompt = st.text_area(
    "Enter your prompt:",
    height=120,
    placeholder="Ask anything to compare both models...",
)


def _stream_response(model: str) -> str:
    """Stream a response from a model, handling both content and reasoning_content."""
    full_response = ""
    thinking_placeholder = st.empty()
    answer_placeholder = st.empty()
    thinking_text = ""
    answer_text = ""

    stream = client.chat.completions.create(
        model=model,
        messages=[
            {{"role": "system", "content": SYSTEM_PROMPT}},
            {{"role": "user", "content": prompt}},
        ],
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta

        # Handle reasoning/thinking content (DeepSeek R1, Kimi-K2-Thinking, etc.)
        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning:
            thinking_text += reasoning
            thinking_placeholder.markdown(
                f"<details><summary>💭 Thinking...</summary>\n\n{thinking_text}\n\n</details>",
                unsafe_allow_html=True,
            )

        # Handle final answer content
        if delta.content:
            answer_text += delta.content
            answer_placeholder.markdown(answer_text)
            full_response += delta.content

    return full_response


if st.button("⚡ Compare Models", type="primary", disabled=not prompt):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"🤖 {MODEL_A_LABEL}")
        try:
            _stream_response(MODEL_A)
        except Exception as e:
            st.error(f"Error: {{e}}")

    with col2:
        st.subheader(f"🤖 {MODEL_B_LABEL}")
        try:
            _stream_response(MODEL_B)
        except Exception as e:
            st.error(f"Error: {{e}}")
'''
