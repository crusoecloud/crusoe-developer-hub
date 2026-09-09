import os
from dotenv import load_dotenv

load_dotenv()

CRUSOE_API_KEY = os.getenv("CRUSOE_API_KEY", "no-key-set")
CRUSOE_API_BASE = os.getenv(
    "CRUSOE_API_BASE",
    "https://managed-inference-api-proxy.crusoecloud.com/v1/",
)

MODEL_NAME = os.getenv("DEBATE_MODEL", "google/gemma-4-31b-it")

MAX_ROUNDS = 5
MIN_ROUNDS = 1
MAX_CONTEXT_MESSAGES = 40

# Reading-pace throttle for streamed text. 0 = no throttle (raw model speed).
CHARS_PER_SECOND = float(os.getenv("CHARS_PER_SECOND", "25"))
