import os
from dotenv import load_dotenv

load_dotenv()

CRUSOE_API_KEY = os.environ.get("CRUSOE_API_KEY", "")
CRUSOE_BASE_URL = os.environ.get("CRUSOE_BASE_URL", "https://api.inference.crusoecloud.com/v1/")

HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_USERNAME = os.environ.get("HF_USERNAME", "")

# Model assignments per pipeline stage
INTENT_MODEL = os.environ.get("INTENT_MODEL", "deepseek-ai/DeepSeek-R1-0528")
CODEGEN_MODEL = os.environ.get("CODEGEN_MODEL", "moonshotai/Kimi-K2-Thinking")
HEALER_MODEL = os.environ.get("HEALER_MODEL", "Qwen/Qwen3-235B-A22B-Instruct-2507")
