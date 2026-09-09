import os
from dotenv import load_dotenv

load_dotenv()

CRUSOE_API_KEY = os.getenv("CRUSOE_API_KEY", "no-key-set")
CRUSOE_API_BASE = os.getenv(
    "CRUSOE_API_BASE",
    "https://managed-inference-api-proxy.crusoecloud.com/v1/",
)
VOICECHAT_API_BASE = os.getenv(
    "VOICECHAT_API_BASE",
    "https://demo-voicechat-placeholder.crusoecloud.com/v1/",
)

# --- Models ---
HOST_A_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
HOST_B_MODEL = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B"
VOICECHAT_MODEL = "nvidia/NVIDIA-Nemotron-3-VoiceChat"

# --- Host defaults ---
DEFAULT_HOST_A = {
    "name": "Atlas",
    "color": "#8B5CF6",
    "personality": (
        "Enthusiastic deep-diver. Goes on fascinating tangents, connects "
        "unexpected dots, says 'actually' and 'here's the wild part' a lot. "
        "Verbose but captivating."
    ),
}

DEFAULT_HOST_B = {
    "name": "Nova",
    "color": "#10B981",
    "personality": (
        "Sharp, witty reactor. Keeps things grounded with humor, challenges "
        "bold claims, asks the questions the audience is thinking. Concise "
        "and punchy."
    ),
}

# --- TTS voices (edge-tts) ---
TTS_VOICE_A = "en-US-ChristopherNeural"  # Male, friendly conversational
TTS_VOICE_B = "en-US-AriaNeural"         # Female, versatile conversational

INSPIRE_CATEGORIES = [
    "funny",
    "techy",
    "deep",
    "business",
    "weird",
    "science",
    "history",
    "pop-culture",
]
