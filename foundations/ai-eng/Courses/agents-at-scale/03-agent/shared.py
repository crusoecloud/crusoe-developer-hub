import asyncio
from config import settings

SLOTS = asyncio.Semaphore(settings.max_concurrency)
