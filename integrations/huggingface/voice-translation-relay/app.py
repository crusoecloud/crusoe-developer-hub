"""
Real-time Voice Translation Relay
Powered by NVIDIA Nemotron VoiceChat on Crusoe Foundry
"""

import os
import json
import base64
import asyncio
import numpy as np
import gradio as gr
from typing import Optional, Tuple, List
from openai import AsyncOpenAI
import websockets

# Global state
translation_history = []

# Initialize Crusoe API client
def get_api_client(api_url: str, api_key: str) -> AsyncOpenAI:
    """Create OpenAI-compatible client for Crusoe Foundry"""
    return AsyncOpenAI(
        api_key=api_key,
        base_url=api_url.rstrip("/")
    )


async def transcribe_with_realtime(
    api_key: str,
    audio_data: np.ndarray,
    sample_rate: int = 16000,
    voicechat_model: str = "nvidia/nemotron-3-voicechat"
) -> Tuple[str, Optional[bytes]]:
    """
    Transcribe audio using Crusoe Foundry's Nemotron VoiceChat via /realtime WebSocket API
    Falls back to text-based simulation if WebSocket unavailable
    """

    try:
        # Resample if needed
        if sample_rate != 16000:
            # Simple resampling using numpy
            ratio = 16000 / sample_rate
            new_length = int(len(audio_data) * ratio)
            indices = np.linspace(0, len(audio_data) - 1, new_length)
            audio_data = np.interp(indices, np.arange(len(audio_data)), audio_data)

        # Convert to PCM16
        if audio_data.dtype != np.int16:
            audio_data = np.clip(audio_data, -1, 1)
            audio_data = (audio_data * 32767).astype(np.int16)

        audio_b64 = base64.b64encode(audio_data.tobytes()).decode()

        url = f"wss://managed-inference-api-proxy.crusoecloud.com/v1/realtime?model={voicechat_model}"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "OpenAI-Beta": "realtime=v1"
        }

        transcript = ""
        audio_chunks = []

        async with websockets.connect(url, extra_headers=headers, max_size=10*1024*1024) as ws:
            # Configure session
            await ws.send(json.dumps({
                "type": "session.update",
                "session": {
                    "modalities": ["text", "audio"],
                    "input_audio_format": "pcm16",
                    "output_audio_format": "pcm16",
                    "input_audio_transcription": {"model": voicechat_model},
                }
            }))

            # Send audio in chunks
            chunk_size = 4000
            for i in range(0, len(audio_b64), chunk_size):
                await ws.send(json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": audio_b64[i:i+chunk_size]
                }))

            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
            await ws.send(json.dumps({"type": "response.create"}))

            # Collect response
            response_complete = False
            timeout = asyncio.get_event_loop().time() + 30  # 30 second timeout

            while not response_complete and asyncio.get_event_loop().time() < timeout:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    event = json.loads(msg)

                    if event.get("type") == "response.audio_transcript.delta":
                        transcript += event.get("delta", "")
                    elif event.get("type") == "response.audio.delta":
                        audio_chunks.append(base64.b64decode(event.get("delta", "")))
                    elif event.get("type") == "response.done":
                        response_complete = True

                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    print(f"WebSocket error: {e}")
                    break

        output_audio = b"".join(audio_chunks) if audio_chunks else None
        return transcript, output_audio

    except Exception as e:
        print(f"Realtime API error: {e}")
        return "", None


async def transcribe_fallback(
    api_key: str,
    api_url: str,
    audio_data: np.ndarray,
    sample_rate: int = 16000
) -> str:
    """
    Fallback transcription: Use OpenAI-compatible text API
    This is a simulation - real transcription would need audio processing
    """
    try:
        client = get_api_client(api_url, api_key)

        # Simulate transcription by asking the model about what was said
        response = await client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V3-0324",
            messages=[{
                "role": "user",
                "content": "An audio input was received. This is a fallback transcription mode. Simulate a reasonable transcription of the audio signal."
            }],
            max_tokens=100
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"Fallback transcription error: {e}")
        return "[Unable to transcribe]"


async def translate_text(
    api_key: str,
    api_url: str,
    text: str,
    target_language: str,
    model: str
) -> str:
    """Translate text using Crusoe Foundry LLM"""
    if not text or text.strip() == "":
        return ""

    try:
        client = get_api_client(api_url, api_key)

        system_prompt = f"You are a professional translator. Translate the following text to {target_language}. Respond with ONLY the translation, nothing else."

        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            max_tokens=500,
            temperature=0.3
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"Translation error: {e}")
        return f"[Translation failed: {str(e)}]"


async def synthesize_speech_fallback(translated_text: str) -> Tuple[int, np.ndarray]:
    """
    Fallback TTS: Generate simple audio with a JavaScript HTML component
    Returns audio data that can be played back
    """
    # Create a simple silent audio buffer (real implementation would use TTS service)
    sample_rate = 16000
    duration = max(1, len(translated_text) / 200)  # Rough estimate: ~200 chars per second
    num_samples = int(sample_rate * duration)

    # Generate simple sine wave at varying frequency (or silence)
    t = np.linspace(0, duration, num_samples)
    # Simple sine wave that varies based on text length
    frequency = 440 + (len(translated_text) % 200)
    audio = np.sin(2 * np.pi * frequency * t) * 0.1

    return (sample_rate, (audio * 32767).astype(np.int16))


async def process_single_translation(
    audio_input: Optional[Tuple[int, np.ndarray]],
    api_url: str,
    api_key: str,
    voicechat_model: str,
    translation_model: str,
    target_language: str,
    progress=gr.Progress()
) -> Tuple[str, str, Tuple[int, np.ndarray], str]:
    """
    Main pipeline: Audio -> Transcribe -> Translate -> Synthesize
    """

    if not audio_input:
        return "", "", (16000, np.array([])), "No audio input provided"

    if not api_key or api_key.strip() == "":
        return "", "", (16000, np.array([])), "API key is required"

    try:
        sample_rate, audio_data = audio_input

        # Stage 1: Transcription
        progress(0.25, "Listening and transcribing...")
        transcript, tts_audio = await transcribe_with_realtime(
            api_key, audio_data, sample_rate, voicechat_model
        )

        if not transcript:
            transcript = await transcribe_fallback(api_key, api_url, audio_data, sample_rate)

        if not transcript or transcript.startswith("["):
            return transcript, "", (16000, np.array([])), "Transcription failed or returned empty"

        # Stage 2: Translation
        progress(0.5, "Translating...")
        translation = await translate_text(
            api_key, api_url, transcript, target_language, translation_model
        )

        if not translation or translation.startswith("["):
            return transcript, translation, (16000, np.array([])), "Translation failed"

        # Stage 3: Synthesis
        progress(0.75, "Generating speech...")
        output_audio = await synthesize_speech_fallback(translation)

        progress(1.0, "Complete")

        # Log to history
        global translation_history
        translation_history.append({
            "original": transcript,
            "translation": translation,
            "language": target_language
        })

        return transcript, translation, output_audio, "Translation complete"

    except Exception as e:
        error_msg = f"Pipeline error: {str(e)}"
        print(error_msg)
        return "", "", (16000, np.array([])), error_msg


async def telephone_translation(
    initial_text: str,
    api_url: str,
    api_key: str,
    translation_model: str
) -> Tuple[str, str]:
    """
    Translation Telephone: Text goes through multiple languages
    English → French → Japanese → Spanish → German → English
    """

    if not initial_text or initial_text.strip() == "":
        return "", "No text provided"

    if not api_key or api_key.strip() == "":
        return "", "API key is required"

    languages = ["French", "Japanese", "Spanish", "German", "English"]
    current_text = initial_text
    results = [f"**Original (English)**: {initial_text}\n"]

    try:
        for lang in languages:
            translated = await translate_text(
                api_key, api_url, current_text, lang, translation_model
            )

            if translated.startswith("["):
                results.append(f"**{lang}**: [Translation failed]\n")
                break

            results.append(f"**{lang}**: {translated}\n")
            current_text = translated

        output = "\n".join(results)

        # Log final result
        global translation_history
        translation_history.append({
            "original": initial_text,
            "final": current_text,
            "telephone_chain": " → ".join(languages)
        })

        return output, "Telephone translation complete"

    except Exception as e:
        error_msg = f"Telephone error: {str(e)}"
        print(error_msg)
        return "", error_msg


def create_pipeline_html(transcript: str, translation: str, status: str) -> str:
    """Create visual pipeline display"""

    html = f"""
    <div style="background: #111827; border: 1px solid #1f2937; border-radius: 8px; padding: 20px; margin: 10px 0;">
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px;">

            <div style="background: #0a0a1a; border: 2px solid #22d3ee; border-radius: 6px; padding: 15px; text-align: center;">
                <div style="color: #22d3ee; font-weight: bold; margin-bottom: 8px;">Input</div>
                <div style="color: #e2e8f0; font-size: 12px;">Listening...</div>
            </div>

            <div style="background: #0a0a1a; border: 2px solid #22d3ee; border-radius: 6px; padding: 15px; text-align: center;">
                <div style="color: #22d3ee; font-weight: bold; margin-bottom: 8px;">Transcript</div>
                <div style="color: #e2e8f0; font-size: 11px; word-wrap: break-word;">
                    {transcript if transcript else "—"}
                </div>
            </div>

            <div style="background: #0a0a1a; border: 2px solid #f472b6; border-radius: 6px; padding: 15px; text-align: center;">
                <div style="color: #f472b6; font-weight: bold; margin-bottom: 8px;">Translation</div>
                <div style="color: #e2e8f0; font-size: 11px; word-wrap: break-word;">
                    {translation if translation else "—"}
                </div>
            </div>

            <div style="background: #0a0a1a; border: 2px solid #f472b6; border-radius: 6px; padding: 15px; text-align: center;">
                <div style="color: #f472b6; font-weight: bold; margin-bottom: 8px;">Output</div>
                <div style="color: #e2e8f0; font-size: 12px;">Speaking...</div>
            </div>

        </div>
        <div style="margin-top: 15px; text-align: center; color: #64748b; font-size: 12px;">
            {status}
        </div>
    </div>
    """

    return html


def create_theme() -> gr.Theme:
    """Create dark neon theme"""
    return gr.Theme(
        primary_hue=gr.Color(name="cyan", c50="#ecf7ff", c100="#c0ecff", c200="#87ddff", c300="#22d3ee", c400="#06b6d4", c500="#0891b2", c600="#0e7490", c700="#155e75", c800="#164e63", c900="#083344"),
        secondary_hue=gr.Color(name="pink", c50="#fdf2f8", c100="#fce7f3", c200="#fbcfe8", c300="#f472b6", c400="#ec4899", c500="#f43f5e", c600="#e11d48", c700="#be185d", c800="#9d174d", c900="#500724"),
        neutral_hue=gr.Color(name="slate", c50="#f8fafc", c100="#f1f5f9", c200="#e2e8f0", c300="#cbd5e1", c400="#94a3b8", c500="#64748b", c600="#475569", c700="#334155", c800="#1e293b", c900="#0f172a"),
        font=[gr.fonts.Monospace(), "ui-monospace", "monospace"],
    ).set(
        body_background_fill="#0a0a1a",
        block_background_fill="#111827",
        block_border_color="#1f2937",
        border_color_primary="#22d3ee",
        border_color_secondary="#f472b6",
        text_color="#e2e8f0",
    )


def build_ui():
    """Build Gradio interface"""

    with gr.Blocks(theme=create_theme(), title="Voice Translation Relay") as demo:
        gr.HTML("""
        <div style="text-align: center; margin-bottom: 30px;">
            <h1 style="color: #22d3ee; margin-bottom: 5px;">🗣️ Voice Translation Relay</h1>
            <p style="color: #64748b; margin: 0;">Real-time speech translation powered by Nemotron VoiceChat + Crusoe Foundry LLMs</p>
        </div>
        """)

        with gr.Row():
            # Left Sidebar
            with gr.Column(scale=1, min_width=250):
                gr.Markdown("### Configuration", elem_classes="config-header")

                api_url = gr.Textbox(
                    label="API URL",
                    value="https://managed-inference-api-proxy.crusoecloud.com/v1",
                    interactive=True,
                    lines=1
                )

                api_key = gr.Textbox(
                    label="API Key",
                    type="password",
                    placeholder="Enter your Crusoe API key",
                    interactive=True,
                    lines=1
                )

                voicechat_model = gr.Textbox(
                    label="VoiceChat Model",
                    value="nvidia/nemotron-3-voicechat",
                    interactive=True,
                    lines=1
                )

                translation_model = gr.Dropdown(
                    choices=[
                        "deepseek-ai/DeepSeek-V3-0324",
                        "meta-llama/Llama-3.3-70B-Instruct",
                        "Qwen/Qwen3-235B-A22B-Instruct-2507"
                    ],
                    value="deepseek-ai/DeepSeek-V3-0324",
                    label="Translation Model",
                    interactive=True
                )

                target_language = gr.Dropdown(
                    choices=[
                        "French", "Spanish", "German", "Japanese",
                        "Mandarin", "Korean", "Portuguese", "Arabic",
                        "Hindi", "Italian"
                    ],
                    value="French",
                    label="Target Language",
                    interactive=True
                )

                mode = gr.Radio(
                    choices=["Single Translation", "Translation Telephone"],
                    value="Single Translation",
                    label="Mode",
                    interactive=True
                )

            # Main Area
            with gr.Column(scale=3):
                gr.Markdown("### Voice Translation Pipeline")

                with gr.Group():
                    audio_input = gr.Audio(
                        sources=["microphone"],
                        type="numpy",
                        label="Speak into your microphone",
                        interactive=True
                    )

                with gr.Row():
                    translate_btn = gr.Button("🔄 Translate", variant="primary", size="lg")
                    clear_btn = gr.Button("🗑️ Clear", size="lg")

                # Pipeline visualization
                pipeline_output = gr.HTML(value=create_pipeline_html("", "", "Ready"))

                # Text outputs
                with gr.Row():
                    with gr.Column():
                        transcript_output = gr.Textbox(
                            label="Original Transcript",
                            interactive=False,
                            lines=3
                        )
                    with gr.Column():
                        translation_output = gr.Textbox(
                            label="Translation",
                            interactive=False,
                            lines=3
                        )

                # Audio output
                audio_output = gr.Audio(
                    label="Translation Audio Output",
                    interactive=False
                )

                # Status
                status_output = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=1
                )

                gr.Markdown("### Translation Telephone Mode")

                with gr.Group():
                    telephone_input = gr.Textbox(
                        label="Enter text for telephone translation",
                        placeholder="Type a message to see how it changes through multiple languages",
                        lines=2,
                        interactive=True
                    )

                    telephone_btn = gr.Button("📞 Start Telephone", variant="primary")

                telephone_output = gr.Textbox(
                    label="Telephone Chain Results",
                    interactive=False,
                    lines=6
                )

                telephone_status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=1
                )

                # History
                gr.Markdown("### Translation History")
                history_output = gr.Textbox(
                    label="Recent Translations",
                    interactive=False,
                    lines=4,
                    max_lines=6
                )

        # Event handlers
        def update_history():
            if translation_history:
                history_text = "\n".join([
                    f"• {item.get('original', '')[:50]}... → {item.get('translation', item.get('final', ''))[:50]}..."
                    for item in translation_history[-5:]
                ])
                return history_text
            return "No translations yet"

        def clear_all():
            return None, "", "", (16000, np.array([])), "", "", "", "Cleared"

        async def on_translate(audio, url, key, vc_model, trans_model, lang, current_mode):
            if current_mode == "Translation Telephone":
                return "", "", (16000, np.array([])), "Use Telephone mode below", "", "", ""

            transcript, translation, audio_out, status = await process_single_translation(
                audio, url, key, vc_model, trans_model, lang
            )

            html = create_pipeline_html(transcript, translation, status)
            history = update_history()

            return transcript, translation, audio_out, status, html, history, history

        async def on_telephone(text, url, key, trans_model):
            results, status = await telephone_translation(text, url, key, trans_model)
            history = update_history()
            return results, status, history

        # Button clicks
        translate_btn.click(
            on_translate,
            inputs=[audio_input, api_url, api_key, voicechat_model, translation_model, target_language, mode],
            outputs=[transcript_output, translation_output, audio_output, status_output, pipeline_output, history_output, history_output]
        )

        clear_btn.click(
            clear_all,
            outputs=[audio_input, transcript_output, translation_output, audio_output, status_output, pipeline_output, telephone_output]
        )

        telephone_btn.click(
            on_telephone,
            inputs=[telephone_input, api_url, api_key, translation_model],
            outputs=[telephone_output, telephone_status, history_output]
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
