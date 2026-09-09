DASHBOARD_TEMPLATE = '''import streamlit as st
from openai import OpenAI
import os

TITLE = "{title}"
DESCRIPTION = "{description}"
SYSTEM_PROMPT = """{system_prompt}"""
MODEL = "{model}"
INPUT_LABEL = "{input_label}"
INPUT_PLACEHOLDER = "{input_placeholder}"

client = OpenAI(
    api_key=os.environ.get("CRUSOE_API_KEY"),
    base_url=os.environ.get("CRUSOE_BASE_URL", "https://api.crusoe.ai/v1"),
)

st.set_page_config(page_title=TITLE, page_icon="📊", layout="wide")
st.title(TITLE)
st.caption(DESCRIPTION)
st.divider()

col_input, col_output = st.columns([1, 2])

with col_input:
    st.subheader("Input")
    user_input = st.text_area(INPUT_LABEL, placeholder=INPUT_PLACEHOLDER, height=200)
    analyze = st.button("🔍 Analyze", type="primary", disabled=not user_input)

with col_output:
    st.subheader("Analysis")
    if analyze and user_input:
        stream = client.chat.completions.create(
            model=MODEL,
            messages=[
                {{"role": "system", "content": SYSTEM_PROMPT}},
                {{"role": "user", "content": user_input}},
            ],
            stream=True,
        )

        def get_stream():
            for chunk in stream:
                delta = chunk.choices[0].delta
                # Skip reasoning_content from thinking models — yield only final answer
                if delta.content:
                    yield delta.content

        st.write_stream(get_stream())
    else:
        st.info("Enter your input and click Analyze to get started.")

with st.sidebar:
    st.caption("Powered by [Crusoe](https://crusoe.ai)")
    st.markdown(f"**Model:** `{MODEL}`")
'''
