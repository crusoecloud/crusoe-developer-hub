---
title: Voice Translation Relay
emoji: 🗣️
colorFrom: cyan
colorTo: pink
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: true
license: mit
short_description: Real-time speech translation with Nemotron VoiceChat on Crusoe Foundry
---

# Voice Translation Relay

Real-time speech translation powered by NVIDIA Nemotron VoiceChat running on Crusoe Foundry's managed inference platform.

## Features

- **Voice-to-Voice Translation Pipeline**: Speak into your microphone, get translated audio back
  - Speech-to-text transcription via Nemotron VoiceChat
  - Translation via Crusoe Foundry LLMs (DeepSeek, Llama, Qwen)
  - Text-to-speech synthesis with fallback support
  
- **Translation Telephone**: Watch how meaning drifts across languages
  - English → French → Japanese → Spanish → German → English
  - Text-based chain translation using Crusoe LLMs
  
- **Real-time Pipeline Visualization**: See each stage of the translation process
  - Input waveform → Transcribed text → Translated text → Output audio

## How It Works

### Architecture

1. **Audio Input**: Microphone capture via Gradio
2. **Transcription**: OpenAI-compatible `/realtime` WebSocket API to Nemotron VoiceChat
3. **Translation**: REST API calls to Crusoe Foundry text LLMs
4. **Synthesis**: Text-to-speech output (with graceful fallback)

### Crusoe Foundry Integration

- **API Endpoint**: `https://api.inference.crusoecloud.com/v1`
- **VoiceChat Model**: `nvidia/nemotron-3-voicechat` (WebSocket `/realtime` API)
- **Translation Models**:
  - DeepSeek-V3-0324
  - Llama-3.3-70B-Instruct
  - Qwen3-235B-A22B-Instruct-2507

### Audio Format

- PCM16 (16-bit signed integer)
- 16 kHz sample rate
- Mono channel

## Configuration

Before using the app:

1. Get a Crusoe API key from [https://console.crusoecloud.com](https://console.crusoecloud.com)
2. Enter your API key in the Configuration panel
3. Select your preferred translation model and target language
4. Choose between "Single Translation" or "Translation Telephone" mode

## Modes

### Single Translation Mode
1. Speak into your microphone
2. Click "Translate"
3. Watch the pipeline process your speech
4. Listen to the translated audio

### Translation Telephone Mode
1. Enter text in the telephone input field
2. Click "📞 Start Telephone"
3. See the message transform through multiple languages
4. Observe how meaning changes (or stays the same!) across languages

## Supported Languages

- French
- Spanish
- German
- Japanese
- Mandarin
- Korean
- Portuguese
- Arabic
- Hindi
- Italian

## Technical Details

### WebSocket Realtime API

The app connects to Nemotron VoiceChat using OpenAI's Realtime Protocol:

```
wss://api.inference.crusoecloud.com/v1/realtime?model=nvidia/nemotron-3-voicechat
```

Key events:
- `session.update`: Configure input/output formats
- `input_audio_buffer.append`: Send audio chunks
- `response.create`: Request transcription/synthesis
- `response.audio_transcript.delta`: Receive transcript text
- `response.audio.delta`: Receive synthesized audio

### Fallback Behavior

If the WebSocket connection is unavailable:
- Transcription falls back to text LLM simulation
- Synthesis uses browser Web Speech API

## Running Locally

```bash
pip install -r requirements.txt
python app.py
```

The app will launch at `http://localhost:7860`

## Deployment

This app is configured for HuggingFace Spaces deployment. Simply upload:
- `app.py`
- `requirements.txt`
- `README.md` (with metadata header)

## Limitations & Notes

- Audio processing is optimized for clear speech
- Translation quality depends on the selected LLM
- Latency varies based on API response times
- Browser must support Audio Recording API
- Currently uses simplified TTS fallback; integrate a TTS service for production

## License

MIT

## Credits

Built with:
- [Gradio](https://gradio.app/) - Web UI framework
- [OpenAI Python Client](https://github.com/openai/openai-python) - API integration
- [NVIDIA Nemotron VoiceChat](https://www.nvidia.com/en-us/ai/) - Speech models
- [Crusoe Foundry](https://crusoecloud.com) - Managed inference platform
