FORM_WIZARD_TEMPLATE = '''import streamlit as st
from openai import OpenAI
import os

TITLE = "{title}"
DESCRIPTION = "{description}"
SYSTEM_PROMPT = """{system_prompt}"""
MODEL = "{model}"
STEPS = {steps}

client = OpenAI(
    api_key=os.environ.get("CRUSOE_API_KEY"),
    base_url=os.environ.get("CRUSOE_BASE_URL", "https://api.crusoe.ai/v1"),
)

st.set_page_config(page_title=TITLE, page_icon="🧙", layout="centered")
st.title(TITLE)
st.caption(DESCRIPTION)
st.divider()

if "step" not in st.session_state:
    st.session_state.step = 0
if "answers" not in st.session_state:
    st.session_state.answers = {{}}
if "result" not in st.session_state:
    st.session_state.result = None

if st.session_state.step < len(STEPS):
    current_step = STEPS[st.session_state.step]
    progress = st.session_state.step / len(STEPS)
    st.progress(progress, text=f"Step {{st.session_state.step + 1}} of {{len(STEPS)}}")
    st.subheader(current_step["question"])

    answer = st.text_input(
        "Your answer:",
        key=f"answer_{{st.session_state.step}}",
        placeholder="Type your answer here...",
    )

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("Next →", type="primary", disabled=not answer):
            st.session_state.answers[current_step["key"]] = answer
            st.session_state.step += 1
            st.rerun()

else:
    st.success("Great! Generating your personalized recommendations...")
    summary = "\\n".join([f"- {{k.replace('_', ' ').title()}}: {{v}}" for k, v in st.session_state.answers.items()])

    if st.session_state.result is None:
        stream = client.chat.completions.create(
            model=MODEL,
            messages=[
                {{"role": "system", "content": SYSTEM_PROMPT}},
                {{"role": "user", "content": f"Based on the following information, provide detailed recommendations:\\n{{summary}}"}},
            ],
            stream=True,
        )

        def get_stream():
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield content

        st.subheader("Your Recommendations")
        result = st.write_stream(get_stream())
        st.session_state.result = result
    else:
        st.subheader("Your Recommendations")
        st.markdown(st.session_state.result)

    st.divider()
    if st.button("Start Over"):
        st.session_state.step = 0
        st.session_state.answers = {{}}
        st.session_state.result = None
        st.rerun()

with st.sidebar:
    st.caption("Powered by [Crusoe](https://crusoe.ai)")
    if st.session_state.answers:
        st.subheader("Your answers so far")
        for k, v in st.session_state.answers.items():
            st.markdown(f"**{{k.replace('_', ' ').title()}}:** {{v}}")
'''
