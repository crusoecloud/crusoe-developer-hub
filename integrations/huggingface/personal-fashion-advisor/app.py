import streamlit as st
from openai import OpenAI
import os

TITLE = "Style Savvy: Your Personal Fashion Advisor"
DESCRIPTION = "Chat with an AI stylist for personalized fashion advice and trend insights."
SYSTEM_PROMPT = """You are a knowledgeable and enthusiastic fashion expert. Provide tailored style recommendations, explain current trends, suggest outfits for occasions, and offer inclusive advice for all body types. Ask clarifying questions when needed, and maintain a friendly, encouraging tone. If asked about non-fashion topics, politely redirect to style discussions."""
CHAT_PLACEHOLDER = "Describe your style or ask for fashion advice..."

MODEL_OPTIONS = {
    "deepseek-ai/DeepSeek-R1-0528": "DeepSeek R1",
    "moonshotai/Kimi-K2-Thinking": "Kimi K2"
}

client = OpenAI(
    api_key=os.environ.get("CRUSOE_API_KEY"),
    base_url=os.environ.get("CRUSOE_BASE_URL", "https://managed-inference-api-proxy.crusoecloud.com/v1/"),
)

st.set_page_config(page_title=TITLE, page_icon="👗", layout="centered")
st.title(f"👗 {TITLE}")
st.markdown(f"*{DESCRIPTION}*")
st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "model" not in st.session_state:
    st.session_state.model = "moonshotai/Kimi-K2-Thinking"

with st.sidebar:
    st.markdown("### ✨ Features")
    st.markdown("""
    - 👔 **Personalized Outfit Recommendations**
    - 🍂 **Seasonal Trend Analysis**
    - 🤝 **Body Type & Style Inclusivity**
    - 🌱 **Sustainable Fashion Tips**
    """)
    
    st.divider()
    
    st.markdown("### ⚙️ Model Settings")
    selected_model = st.selectbox(
        "Choose AI Model",
        options=list(MODEL_OPTIONS.keys()),
        format_func=lambda x: MODEL_OPTIONS[x],
        index=0,
        key="model_selector"
    )
    st.session_state.model = selected_model
    
    st.divider()
    
    st.caption("Powered by [Crusoe](https://crusoe.ai)")
    if st.button("🗑️ Clear Chat", type="secondary", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "👗"):
        st.markdown(message["content"])

if prompt := st.chat_input(CHAT_PLACEHOLDER):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="👗"):
        stream = client.chat.completions.create(
            model=st.session_state.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                *st.session_state.messages,
            ],
            stream=True,
        )

        def get_stream():
            in_think = False
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta.content:
                    text = delta.content
                    # Filter out <think>...</think> blocks from reasoning models
                    if "<think>" in text:
                        in_think = True
                        text = text.split("<think>")[0]
                    if "</think>" in text:
                        in_think = False
                        text = text.split("</think>", 1)[1]
                    if not in_think and text:
                        yield text

        response = st.write_stream(get_stream())
        st.session_state.messages.append({"role": "assistant", "content": response})