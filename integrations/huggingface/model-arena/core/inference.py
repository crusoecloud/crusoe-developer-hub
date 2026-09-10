"""
Crusoe Managed AI inference wrapper using OpenAI-compatible API.
Handles streaming, timing, and token counting.
"""

import os
import time
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI


@dataclass
class InferenceResult:
    model_id: str
    response_text: str
    time_to_first_token: float  # seconds
    total_latency: float        # seconds
    input_tokens: int
    output_tokens: int
    total_tokens: int
    tokens_per_second: float
    error: Optional[str] = None


def get_client() -> OpenAI:
    api_key = os.environ.get("CRUSOE_API_KEY", "")
    if not api_key:
        raise ValueError("CRUSOE_API_KEY environment variable is not set")
    return OpenAI(
        base_url="https://api.inference.crusoecloud.com/v1/",
        api_key=api_key,
    )


def run_inference(
    model_id: str,
    prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    max_tokens: int = 1024,
    temperature: float = 0.7,
) -> InferenceResult:
    """Run non-streaming inference and return timed result."""
    client = get_client()
    start = time.perf_counter()

    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        total_latency = time.perf_counter() - start

        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        total_tokens = usage.total_tokens if usage else 0
        text = response.choices[0].message.content or ""
        tps = output_tokens / total_latency if total_latency > 0 else 0

        return InferenceResult(
            model_id=model_id,
            response_text=text,
            time_to_first_token=total_latency,  # non-streaming, same as total
            total_latency=total_latency,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            tokens_per_second=tps,
        )
    except Exception as e:
        return InferenceResult(
            model_id=model_id,
            response_text="",
            time_to_first_token=0,
            total_latency=time.perf_counter() - start,
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            tokens_per_second=0,
            error=str(e),
        )


def run_inference_streaming(
    model_id: str,
    prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    max_tokens: int = 1024,
    temperature: float = 0.7,
):
    """
    Generator that yields (chunk_text, partial_result) tuples.
    The final yield has the complete InferenceResult.
    """
    client = get_client()
    start = time.perf_counter()
    first_token_time = None
    full_text = ""
    output_tokens = 0

    try:
        stream = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            stream_options={"include_usage": True},
        )

        input_tokens = 0

        for chunk in stream:
            # Track usage from the final chunk
            if chunk.usage:
                input_tokens = chunk.usage.prompt_tokens
                output_tokens = chunk.usage.completion_tokens

            if chunk.choices and chunk.choices[0].delta.content:
                token_text = chunk.choices[0].delta.content
                if first_token_time is None:
                    first_token_time = time.perf_counter() - start
                full_text += token_text
                output_tokens += 1
                yield token_text, None

        total_latency = time.perf_counter() - start
        tps = output_tokens / total_latency if total_latency > 0 else 0

        result = InferenceResult(
            model_id=model_id,
            response_text=full_text,
            time_to_first_token=first_token_time or total_latency,
            total_latency=total_latency,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            tokens_per_second=tps,
        )
        yield "", result

    except Exception as e:
        total_latency = time.perf_counter() - start
        result = InferenceResult(
            model_id=model_id,
            response_text=full_text,
            time_to_first_token=first_token_time or 0,
            total_latency=total_latency,
            input_tokens=0,
            output_tokens=output_tokens,
            total_tokens=output_tokens,
            tokens_per_second=0,
            error=str(e),
        )
        yield "", result
