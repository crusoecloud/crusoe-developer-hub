CHATBOT_TEMPLATE = '''import streamlit as st
from openai import OpenAI
import os

TITLE = "{title}"
DESCRIPTION = "{description}"
SYSTEM_PROMPT = """{system_prompt}"""
MODEL = "{model}"
CHAT_PLACEHOLDER = "{chat_placeholder}"

client = OpenAI(
    api_key=os.environ.get("CRUSOE_API_KEY"),
    base_url=os.environ.get("CRUSOE_BASE_URL", "https://api.crusoe.ai/v1"),
)

st.set_page_config(page_title=TITLE, page_icon="💬", layout="centered")
st.title(TITLE)
st.caption(DESCRIPTION)
st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input(CHAT_PLACEHOLDER):
    st.session_state.messages.append({{"role": "user", "content": prompt}})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=MODEL,
            messages=[
                {{"role": "system", "content": SYSTEM_PROMPT}},
                *st.session_state.messages,
            ],
            stream=True,
        )

        def get_stream():
            for chunk in stream:
                delta = chunk.choices[0].delta
                # Skip reasoning_content from thinking models — yield only final answer
                if delta.content:
                    yield delta.content

        response = st.write_stream(get_stream())
        st.session_state.messages.append({{"role": "assistant", "content": response}})

with st.sidebar:
    st.caption("Powered by [Crusoe](https://crusoe.ai)")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()
'''
