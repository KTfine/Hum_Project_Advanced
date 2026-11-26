from __future__ import annotations

import asyncio
import json


def ensure_asyncio_timeout() -> None:
    """Provide the Python 3.11 timeout context on the Python 3.10 eval env."""
    if not hasattr(asyncio, "timeout"):
        try:
            from async_timeout import timeout
        except ImportError:
            from contextlib import asynccontextmanager

            @asynccontextmanager
            async def timeout(delay):
                task = asyncio.current_task()
                if task is None:
                    raise RuntimeError("asyncio.timeout requires a running task")
                handle = asyncio.get_running_loop().call_later(delay, task.cancel)
                try:
                    yield
                except asyncio.CancelledError as exc:
                    raise TimeoutError from exc
                finally:
                    handle.cancel()
        asyncio.timeout = timeout  # type: ignore[attr-defined]


def parse_deepseek_json(text: str) -> dict:
    stripped = text.strip()
    if stripped.startswith("```json") and stripped.endswith("```"):
        stripped = stripped[7:-3].strip()
    elif stripped.startswith("```") and stripped.endswith("```"):
        stripped = stripped[3:-3].strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError("response must be a single JSON object") from exc
    if not isinstance(value, dict):
        raise ValueError("response must be a single JSON object")
    return value
