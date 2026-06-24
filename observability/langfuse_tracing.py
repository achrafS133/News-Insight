"""Langfuse v4 tracing helpers (optional when keys are not configured)."""

from __future__ import annotations

import functools
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from typing import Any, ParamSpec, TypeVar

try:
    from langfuse import Langfuse
except Exception:  # pragma: nocover - optional dependency
    Langfuse = None

from config.settings import get_settings

P = ParamSpec("P")
R = TypeVar("R")

_langfuse_client: Langfuse | None = None
_tracing_enabled: bool | None = None


def _looks_configured(value: str) -> bool:
    if not value or value.endswith("-") or value.endswith("..."):
        return False
    return len(value.strip()) >= 12


def _is_tracing_enabled() -> bool:
    global _tracing_enabled
    if _tracing_enabled is None:
        settings = get_settings()
        _tracing_enabled = _looks_configured(settings.langfuse_public_key) and _looks_configured(
            settings.langfuse_secret_key.get_secret_value()
        )
    return _tracing_enabled


def get_langfuse_client() -> Langfuse | None:
    global _langfuse_client
    if not _is_tracing_enabled():
        return None
    if _langfuse_client is None:
        settings = get_settings()
        _langfuse_client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key.get_secret_value(),
            host=settings.langfuse_host,
        )
    return _langfuse_client


def flush_langfuse() -> None:
    client = get_langfuse_client()
    if client is not None:
        client.flush()


class LangfuseTraceContext:
    """Span wrapper for Airflow callbacks and batch jobs."""

    def __init__(
        self,
        name: str,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        input_data: Any | None = None,
    ) -> None:
        self.name = name
        self.user_id = user_id
        self.session_id = session_id
        self.metadata = {**(metadata or {}), "tags": tags or []}
        self.input_data = input_data
        self._cm: Any = None
        self._observation: Any = None
        self._output: Any = None
        self._start = 0.0

    def __enter__(self) -> LangfuseTraceContext:
        client = get_langfuse_client()
        self._start = time.perf_counter()
        if client is None:
            return self

        trace_id = None
        if self.session_id:
            trace_id = client.create_trace_id(seed=self.session_id)

        self._cm = client.start_as_current_observation(
            name=self.name,
            as_type="span",
            input=self.input_data,
            metadata=self.metadata,
            trace_context={"trace_id": trace_id} if trace_id else None,
        )
        self._observation = self._cm.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._cm is not None and self._observation is not None:
            latency_ms = (time.perf_counter() - self._start) * 1000
            self._observation.update(
                output=self._output,
                metadata={**self.metadata, "latency_ms": latency_ms},
                level="ERROR" if exc else "DEFAULT",
                status_message=str(exc) if exc else None,
            )
            self._cm.__exit__(exc_type, exc, tb)
        flush_langfuse()

    def update_output(self, output: Any) -> None:
        self._output = output


def trace_span(
    name: str, *, tags: list[str] | None = None
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            with LangfuseTraceContext(
                name,
                tags=tags or [],
                metadata={"callable": func.__name__},
            ):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def trace_generation(
    name: str,
    *,
    model: str | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            client = get_langfuse_client()
            if client is None:
                return func(*args, **kwargs)

            with client.start_as_current_observation(
                name=name,
                as_type="generation",
                model=model,
                metadata={"callable": func.__name__},
            ) as generation:
                result = func(*args, **kwargs)
                generation.update(output=result)
                flush_langfuse()
                return result

        return wrapper

    return decorator


@contextmanager
def generation_span(
    name: str,
    *,
    model: str,
    input_messages: list[dict[str, str]],
    metadata: dict[str, Any] | None = None,
) -> Iterator[Any]:
    client = get_langfuse_client()
    if client is None:
        yield None
        return

    with client.start_as_current_observation(
        name=name,
        as_type="generation",
        model=model,
        input=input_messages,
        metadata=metadata,
    ) as generation:
        yield generation
    flush_langfuse()
