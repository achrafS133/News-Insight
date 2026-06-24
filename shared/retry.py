"""Shared retry helpers for HTTP and LLM calls."""

from __future__ import annotations

from typing import TypeVar

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

T = TypeVar("T")


def api_retry(max_attempts: int = 3):
    """Exponential backoff with jitter for transient API failures."""
    return retry(
        reraise=True,
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential_jitter(initial=1, max=30),
        retry=retry_if_exception_type((TimeoutError, ConnectionError, OSError)),
    )
