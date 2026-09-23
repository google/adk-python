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
from collections.abc import Mapping
import inspect
import logging
import re
from typing import Any
from typing import cast

from google.genai import types
from pydantic import ValidationError

logger = logging.getLogger("google_adk." + __name__)

__all__ = [
    "build_api_key",
    "is_reasoning_model",
    "map_finish_reason",
    "serialize_system_instruction",
    "strip_unsupported_sampling_params",
    "tool_choice",
]

# Matches the OpenAI *reasoning* model families (the o-series and the gpt-5.x /
# gpt-6.x families, optionally namespaced e.g. ``openai/o3-mini``). These reject
# a non-default ``temperature`` / ``top_p`` on both the Chat Completions and
# Responses APIs, and on Chat Completions additionally require
# ``max_completion_tokens`` in place of ``max_tokens``.
# Deliberately does NOT match ``gpt-4o`` / ``gpt-4.1`` or non-OpenAI models
# such as ``xai/grok-4.6``, which accept the classic parameters. It also
# excludes the ``-chat`` variants (``gpt-5-chat``, ``gpt-5-chat-latest``,
# ``gpt-5.1-chat``), which are the non-reasoning chat models: they accept
# ``temperature`` / ``top_p`` and take no reasoning-effort parameter. The
# ``-chat`` exclusion is a lookahead on the whole tail (``(?!.*-chat)``) so a
# minor-version dot (``gpt-5.1-``) cannot slip a chat model through ahead of it.
_REASONING_MODEL_RE = re.compile(
    r"(?:^|/)(?:o\d+|gpt-[56](?!.*-chat))(?:\..*|-.*|$)", re.IGNORECASE
)


def is_reasoning_model(model: str | None) -> bool:
  """Returns True if ``model`` is an OpenAI reasoning model.

  Reasoning models (o-series, gpt-5.x, gpt-6.x) reject non-default
  ``temperature`` / ``top_p`` on both the Chat Completions and Responses APIs,
  and on Chat Completions additionally require ``max_completion_tokens`` in
  place of ``max_tokens``.
  """
  if not model:
    return False
  return bool(_REASONING_MODEL_RE.search(model))


# The only ``temperature`` / ``top_p`` value a reasoning model accepts.
_DEFAULT_SAMPLING_VALUE = 1


def strip_unsupported_sampling_params(
    kwargs: dict[str, Any], model: str | None
) -> None:
  """Strips ``temperature`` / ``top_p`` that a reasoning model would reject.

  Reasoning models accept only the default value (1) for ``temperature`` and
  ``top_p`` on both the Chat Completions and Responses APIs. A non-default
  value would make the backend 400, so drop it (with a warning) in place;
  the default is left untouched. No-op for non-reasoning models.
  """
  if not is_reasoning_model(model):
    return
  for name in ("temperature", "top_p"):
    value = kwargs.get(name)
    if value is not None and value != _DEFAULT_SAMPLING_VALUE:
      kwargs.pop(name, None)
      logger.warning(
          "Ignoring %s=%r: reasoning model %r accepts only the default"
          " value (%s); set it to that or remove it from the request"
          " config.",
          name,
          value,
          model,
          _DEFAULT_SAMPLING_VALUE,
      )


def serialize_system_instruction(
    system_instruction: types.ContentUnion | types.ContentUnionDict | None,
) -> str | None:
  """Serializes an ADK system instruction to plain text.

  ``config.system_instruction`` is usually a ``str``, but the field type allows
  a ``Part``, ``Content``, a mapping, or a list of these. Flatten any of them to
  the text OpenAI expects for a system message / instructions field.

  Returns:
    The flattened instruction text, or ``None`` when there is nothing usable to
    send. Each ``None`` case omits the system message: an empty/falsy
    instruction; a ``Part``/``Content``/list carrying no text (e.g. only inline
    data or a ``File``); a mapping that neither ``Content`` nor ``Part`` can
    parse; or an unsupported type. The mapping-parse-failure and
    unsupported-type cases are logged at ``warning`` so a dropped instruction
    can be diagnosed.
  """
  if not system_instruction:
    return None
  if isinstance(system_instruction, str):
    return system_instruction
  if isinstance(system_instruction, types.Part):
    return system_instruction.text or None
  if isinstance(system_instruction, types.Content):
    text = "".join(part.text or "" for part in system_instruction.parts or [])
    return text or None
  if isinstance(system_instruction, Mapping):
    # A mapping may be Part-shaped ({"text": ...}) or Content-shaped
    # ({"role": ..., "parts": [...]}). Dispatch on the recognized model rather
    # than assuming Part, which raises a ValidationError on a Content-shaped
    # dict. ``model_validate`` (not ``**`` unpacking) keeps mypy happy. A
    # mapping pydantic cannot accept is dropped instead of crashing: an
    # unrecognized shape raises ValidationError, and a non-string key raises
    # TypeError (pydantic expands mapping keys the way ``**`` does), so both
    # are caught.
    model = types.Content if "parts" in system_instruction else types.Part
    try:
      return serialize_system_instruction(
          model.model_validate(dict(system_instruction))
      )
    except (ValidationError, TypeError):
      logger.warning(
          "Could not parse system instruction mapping as %s.", model.__name__
      )
      return None
  if isinstance(system_instruction, list):
    texts: list[str] = []
    for item in system_instruction:
      serialized = serialize_system_instruction(item)
      if serialized:
        texts.append(serialized)
    return "\n".join(texts) or None
  logger.warning(
      "Ignoring system instruction of unsupported type %s; no text to send.",
      type(system_instruction).__name__,
  )
  return None


def tool_choice(
    config: types.GenerateContentConfig | None,
) -> str | None:
  """Maps an ADK function-calling mode to an OpenAI ``tool_choice`` value.

  Mapping:

  * ``ANY`` -> ``"required"`` (the model must call a tool)
  * ``NONE`` -> ``"none"`` (the model must not call a tool)
  * ``AUTO`` -> ``"auto"`` (the model decides)
  * ``VALIDATED``, ``MODE_UNSPECIFIED``, or no config -> ``None`` (no explicit
    choice; each caller decides the fallback -- the Chat Completions wrapper
    sends ``"auto"`` when tools are present, the Responses wrapper omits
    ``tool_choice`` and leaves the provider default)

  ``allowed_function_names`` is not applied: OpenAI's ``tool_choice`` can force a
  single named function or "any tool", but cannot express a subset allow-list of
  several functions, so ``ANY`` maps to ``"required"`` and the model may pick any
  declared tool. Restricting the callable set to a named subset is not
  supported.
  """
  if (
      not config
      or not config.tool_config
      or not config.tool_config.function_calling_config
  ):
    return None
  mode = config.tool_config.function_calling_config.mode
  if mode == types.FunctionCallingConfigMode.ANY:
    return "required"
  if mode == types.FunctionCallingConfigMode.NONE:
    return "none"
  if mode == types.FunctionCallingConfigMode.AUTO:
    return "auto"
  return None


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
