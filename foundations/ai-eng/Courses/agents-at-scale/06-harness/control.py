import sys
from collections import Counter
from pathlib import Path
import structlog
from openai import APIStatusError, APITimeoutError, RateLimitError
from tenacity import RetryError, retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "03-agent"))
from config import client, settings

log = structlog.get_logger()


class Stuck(Exception):
    pass


class ProviderUnavailable(Exception):
    pass


class LoopGuard:
    def __init__(self):
        self.seen = Counter()

    def check(self, call) -> str | None:
        key = (call.function.name, call.function.arguments)
        self.seen[key] += 1
        if self.seen[key] == 2:
            return ("You already called this tool with identical arguments. "
                    "The result has not changed. Change the query or finish.")
        if self.seen[key] >= 3:
            raise Stuck(key)
        return None


@retry(wait=wait_exponential_jitter(initial=1, max=30), stop=stop_after_attempt(5),
       retry=retry_if_exception_type((RateLimitError, APIStatusError, APITimeoutError)))
async def complete(model, messages, tools):
    return await client.chat.completions.create(model=model, messages=messages, tools=tools, tool_choice="auto",
                                                timeout=settings.request_timeout_s)


async def complete_with_fallback(messages, tools, models: list[str] | None = None):
    for model in models or [settings.model, *settings.fallback_models]:
        try:
            return await complete(model, messages, tools)
        except RetryError:
            log.warning("falling back", model=model)
    raise ProviderUnavailable


def make_dispatch_once(dispatch):
    done: dict[str, str] = {}

    async def dispatch_once(call) -> str:
        if call.id in done:
            return done[call.id]
        result = await dispatch(call)
        done[call.id] = result
        return result

    return dispatch_once
