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

"""Helpers for the OpenAI model wrappers.

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
import os
import re
from typing import Any
from typing import cast
from typing import Literal

from google.genai import types
from pydantic import Field
from pydantic import model_validator
from pydantic import ValidationError

logger = logging.getLogger("google_adk." + __name__)

__all__ = [
    "OpenAIGenerateContentConfig",
    "OpenAIReasoningEffort",
    "build_api_key",
    "build_reasoning_effort",
    "is_reasoning_model",
    "map_finish_reason",
    "serialize_system_instruction",
    "strip_unsupported_sampling_params",
    "supported_efforts",
    "targets_default_openai_host",
    "tool_choice",
]

# The OpenAI *reasoning effort* tiers, ordered from least to most effort. The
# set a given model actually accepts is model-dependent (see
# ``supported_efforts``); sending an unsupported tier is an API 400.
OpenAIReasoningEffort = Literal[
    "minimal", "low", "medium", "high", "xhigh", "max"
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


# --- Reasoning effort --------------------------------------------------------
#
# Which effort tiers a model accepts depends on BOTH the model family AND the
# API surface: the same gpt-6.x/gpt-5.6.x model accepts ``max`` on the Responses
# API but rejects it (400) on Chat Completions. A tier outside a model's set is
# an API 400, so ``supported_efforts`` resolves (model, surface) to the accepted
# set.
#
#   * gpt-6.x / gpt-5.5.x / gpt-5.6.x (advanced flagships):
#       - Chat Completions: low/medium/high/xhigh (no minimal, no max).
#       - Responses:        low/medium/high/xhigh/max (no minimal).
#   * gpt-5 / gpt-5-mini / gpt-5-nano: minimal/low/medium/high.
#   * o-series (o1, o3, o3-mini, o4-mini): low/medium/high (no minimal).
#   * o1-mini / o1-preview: do not accept a reasoning-effort parameter at all
#     (the parameter was introduced with the o1 GA release).
#
# The advanced-family sets are verified live (2026-09) against gpt-6-astra and
# gpt-5.6-{sol,terra,luna} on both surfaces (see the reasoning integration
# tests). The gpt-5 and o-series sets follow the OpenAI docs and are not
# live-verified here. Adjust here if a family starts/stops accepting a tier.
ApiSurface = Literal["chat", "responses"]

_EFFORTS_ADVANCED_CHAT: frozenset[str] = frozenset(
    {"low", "medium", "high", "xhigh"}
)
_EFFORTS_ADVANCED_RESPONSES: frozenset[str] = frozenset(
    {"low", "medium", "high", "xhigh", "max"}
)
_EFFORTS_GPT5: frozenset[str] = frozenset({"minimal", "low", "medium", "high"})
_EFFORTS_OSERIES: frozenset[str] = frozenset({"low", "medium", "high"})
_NO_EFFORTS: frozenset[str] = frozenset()

# gpt-6.x, gpt-5.5.x and gpt-5.6.x (a minor version follows ``gpt-6`` / a dotted
# minor follows ``gpt-5``): the extended tier range including xhigh (+ max on
# the Responses surface).
_ADVANCED_RE = re.compile(
    r"(?:^|/)(?:gpt-6|gpt-5\.[56])(?:[.\-]|$)", re.IGNORECASE
)
# Plain gpt-5 / gpt-5-mini / gpt-5-nano (and dated snapshots like
# ``gpt-5-2025-08-07``): ``gpt-5`` NOT followed by a dotted minor version.
_GPT5_RE = re.compile(r"(?:^|/)gpt-5(?![.\d])", re.IGNORECASE)
# o1-mini and o1-preview predate the reasoning-effort parameter and reject it.
_O1_NO_EFFORT_RE = re.compile(
    r"(?:^|/)o1-(?:mini|preview)(?:[.\-]|$)", re.IGNORECASE
)


def supported_efforts(model: str | None, api: ApiSurface) -> frozenset[str]:
  """Returns the reasoning-effort tiers ``model`` accepts on ``api``.

  Args:
    model: The model id.
    api: The API surface -- ``"chat"`` (Chat Completions) or ``"responses"``.
      Advanced flagships accept ``max`` on Responses but not on Chat.

  Returns:
    The accepted effort tiers. An empty set means the model takes no
    reasoning-effort parameter at all (either it is not a reasoning model, or --
    like ``o1-mini`` / ``o1-preview`` -- it is one that rejects the
    parameter).
  """
  # Validate the API surface up front so an unknown value fails deterministically
  # for every model family, not only the advanced ones below.
  if api not in ("chat", "responses"):
    raise ValueError(f"Unknown API surface {api!r}.")
  if not model or not is_reasoning_model(model):
    return _NO_EFFORTS
  if _O1_NO_EFFORT_RE.search(model):
    return _NO_EFFORTS
  if _ADVANCED_RE.search(model):
    if api == "responses":
      return _EFFORTS_ADVANCED_RESPONSES
    return _EFFORTS_ADVANCED_CHAT
  if _GPT5_RE.search(model):
    return _EFFORTS_GPT5
  # Conservative default: the low/medium/high range shared by the o-series
  # (o1/o3/o4...). This is also reached by any recognized reasoning model that
  # is not special-cased above -- notably ``gpt-5.x`` outside 5.5/5.6 (e.g.
  # ``gpt-5.1``), which matches neither _ADVANCED_RE nor _GPT5_RE. An
  # unrecognized future model (e.g. gpt-7) returns _NO_EFFORTS above, so a
  # configured effort raises until it is special-cased here or the caller
  # passes validate=False.
  return _EFFORTS_OSERIES


class OpenAIGenerateContentConfig(types.GenerateContentConfig):
  """GenerateContentConfig with OpenAI-specific reasoning controls.

  This is the recommended way to configure reasoning effort for OpenAI models.
  Set ``effort`` directly to pick a tier; the exact tiers a model accepts are
  model-dependent (see ``supported_efforts``).

  The standard ``thinking_config`` (``thinking_level`` / ``thinking_budget``) is
  intentionally unsupported: the genai ``ThinkingLevel`` enum
  (minimal/low/medium/high) cannot express OpenAI's full, model-dependent effort
  range (which also includes ``xhigh`` / ``max``), so mirroring
  ``AnthropicGenerateContentConfig`` we require the OpenAI-native ``effort``
  field instead and reject a ``thinking_config`` that sets either field.

  Attributes:
    effort: The reasoning effort tier (e.g. ``"minimal"``, ``"high"``,
      ``"max"``). This is the only supported way to configure reasoning effort
      on this config; setting ``thinking_config.thinking_level`` or
      ``thinking_budget`` is rejected (see ``_validate_no_thinking_config``).
  """

  effort: OpenAIReasoningEffort | None = Field(
      default=None,
      description=(
          "Configures the OpenAI reasoning effort tier. This is the"
          " recommended way to control reasoning depth on OpenAI reasoning"
          " models; the accepted tiers are model-dependent."
      ),
  )

  @model_validator(mode="after")
  def _validate_no_thinking_config(self) -> "OpenAIGenerateContentConfig":
    """Rejects a standard thinking_config on the OpenAI-specific config.

    ``build_reasoning_effort`` treats both ``thinking_level`` and
    ``thinking_budget`` as unsupported, so reject either here rather than
    silently ignoring a ``thinking_budget`` set alongside ``effort``.
    """
    if self.thinking_config and (
        self.thinking_config.thinking_level is not None
        or self.thinking_config.thinking_budget is not None
    ):
      raise ValueError(
          "thinking_config (thinking_level / thinking_budget) is not supported"
          " in OpenAIGenerateContentConfig. Use the `effort` field directly to"
          " configure reasoning effort."
      )
    return self


def targets_default_openai_host(
    *,
    client: object | None,
    base_url: str | None,
    azure_endpoint: str | None = None,
) -> bool:
  """Returns True when the request targets api.openai.com, not a compatible backend.

  Reasoning-effort validation against the OpenAI per-model tables is only safe
  on the real OpenAI backend. An injected ``client``, a custom ``base_url`` (or
  Azure ``azure_endpoint``), or ``OPENAI_BASE_URL`` in the environment (which
  the default ``AsyncOpenAI`` client reads) all mean an OpenAI-compatible
  backend (e.g. Grok on Vertex AI, or an Azure deployment) may be in use, whose
  model id need not be an OpenAI id -- so the tier is passed through and the
  backend rejects an unsupported one. ``AZURE_OPENAI_ENDPOINT`` is deliberately
  not consulted: no wrapper here reads it, so a request made while it is set
  still reaches the default host and must still be validated.
  Both the Chat Completions and Responses wrappers share this predicate to avoid
  drift.
  """
  return (
      client is None
      and base_url is None
      and azure_endpoint is None
      and not os.environ.get("OPENAI_BASE_URL")
  )


def build_reasoning_effort(
    config: types.GenerateContentConfig | None,
    model: str | None,
    api: ApiSurface,
    *,
    validate: bool = True,
) -> str | None:
  """Resolves the reasoning-effort tier to send for ``model`` on ``api``.

  The effort tier comes from ``OpenAIGenerateContentConfig.effort``. The
  standard ``thinking_config`` (``thinking_level`` / ``thinking_budget``) is not
  supported for OpenAI models -- the genai ``ThinkingLevel`` enum cannot express
  OpenAI's model-dependent tier range. On an ``OpenAIGenerateContentConfig`` a
  ``thinking_config`` is rejected at construction (see
  ``_validate_no_thinking_config``); on a plain ``GenerateContentConfig`` it is
  ignored here with a logged warning, mirroring
  ``AnthropicGenerateContentConfig``. Callers must use
  ``OpenAIGenerateContentConfig`` and set ``effort`` directly.

  Args:
    config: The request config, ideally an ``OpenAIGenerateContentConfig``.
    model: The target model id, used to validate the tier is accepted.
    api: The API surface the request targets (``"chat"`` or ``"responses"``);
      the accepted tiers differ between surfaces.
    validate: Whether to validate the tier against ``model``. Callers pass
      ``False`` whenever the request may target an OpenAI-compatible backend
      rather than api.openai.com -- that is, when an explicit ``client`` was
      injected, a custom ``base_url`` (or Azure ``azure_endpoint``) was set, or
      ``OPENAI_BASE_URL`` is present. On such a backend (e.g. Grok on Vertex AI
      or an Azure deployment) the ``model`` id may be a partner id or a
      deployment name that does not reveal the accepted tiers, so the configured
      tier is passed through and the backend rejects an unsupported one.

  Returns:
    The effort string to send, or ``None`` when no effort is configured.

  Raises:
    ValueError: If ``validate`` and ``effort`` is set but the target
      model/surface does not accept it (an empty supported set, or a tier
      outside the range).
  """
  if not config:
    return None

  if isinstance(config, OpenAIGenerateContentConfig) and config.effort:
    effort = config.effort
  else:
    if config.thinking_config and (
        config.thinking_config.thinking_level is not None
        or config.thinking_config.thinking_budget is not None
    ):
      # A logger warning is used rather than ``warnings.warn``: the default
      # once-per-process warnings filter keys on the warn() call site, so a
      # single ``stacklevel`` cannot fit both the shallower Chat and the deeper
      # Responses call chains -- on one surface the message would be attributed
      # to internal ADK code and suppressed after the first request.
      logger.warning(
          "Standard thinking_config is not supported for OpenAI models and"
          " will be ignored. Use OpenAIGenerateContentConfig and set the"
          " `effort` field directly to configure reasoning effort."
      )
    return None

  if not validate:
    return effort

  supported = supported_efforts(model, api)
  if not supported:
    raise ValueError(
        f"Model {model!r} does not accept a reasoning effort parameter; remove"
        f" `effort` (got {effort!r}) from the config for this model."
    )
  if effort not in supported:
    raise ValueError(
        f"Reasoning effort {effort!r} is not supported by model {model!r} on"
        f" the {api} API. Supported tiers: {sorted(supported)}."
    )
  return effort


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
