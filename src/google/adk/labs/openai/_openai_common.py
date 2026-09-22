# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helpers for the labs OpenAI wrappers.

Home for parsing, mapping, and client-construction logic that the Chat
Completions wrapper (``_openai_llm.py``) and the Responses wrapper
(``_openai_responses_llm.py``) share. The Responses model keeps its own
status-based ``_map_finish_reason`` (it maps a Responses API *status*, not a
finish-reason string), so that mapper is not shared here.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from collections.abc import Callable
import inspect
from typing import cast

from google.genai import types

__all__ = [
    "build_api_key",
    "map_finish_reason",
]


def map_finish_reason(
    finish_reason: str | None,
) -> types.FinishReason | None:
  """Maps an OpenAI chat-completion finish reason to an ADK FinishReason.

  A finish reason that is present but not one we recognize maps to
  ``FINISH_REASON_UNSPECIFIED``, matching the convention in
  ``models/anthropic_llm.py`` and ``models/apigee_llm.py`` (``OTHER`` is
  reserved for recognized abnormal terminations).
  """
  if finish_reason in ("stop", "tool_calls", "function_call"):
    return types.FinishReason.STOP
  if finish_reason == "length":
    return types.FinishReason.MAX_TOKENS
  if finish_reason == "content_filter":
    return types.FinishReason.SAFETY
  if not finish_reason:
    return None
  return types.FinishReason.FINISH_REASON_UNSPECIFIED


def build_api_key(
    api_key: str | Callable[[], str] | Callable[[], Awaitable[str]] | None,
) -> str | Callable[[], Awaitable[str]] | None:
  """Adapts ``api_key`` to what ``AsyncOpenAI`` accepts.

  ``AsyncOpenAI`` takes either a string key or an async provider
  (``Callable[[], Awaitable[str]]``) that it awaits on every request, so a
  credential that expires (e.g. a Vertex AI OAuth token) is refreshed without
  rebuilding the client. A string or ``None`` is returned unchanged. A callable
  -- whether it returns the key synchronously or as an awaitable -- is wrapped
  in an async provider, so both styles work and the SDK owns the per-request
  refresh.
  """
  if api_key is None or isinstance(api_key, str):
    return api_key

  provider = api_key

  async def _async_provider() -> str:
    # An async provider is awaited directly on the loop. A sync provider may
    # perform blocking I/O (e.g. synchronous token acquisition), so run it off
    # the event loop with ``asyncio.to_thread`` to avoid stalling the loop;
    # dispatching it in a worker thread also avoids a ``RuntimeError`` if the
    # callable internally touches the loop (e.g. ``asyncio.ensure_future``).
    if inspect.iscoroutinefunction(provider):
      return await cast(Callable[[], Awaitable[str]], provider)()
    value: str | Awaitable[str] = await asyncio.to_thread(
        cast(Callable[[], str], provider)
    )
    if inspect.isawaitable(value):
      return await value
    return value

  return _async_provider
