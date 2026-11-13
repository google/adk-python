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

"""Sample agent configured from Parameter Manager at runtime."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import textwrap
import time

from google.adk import Agent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.integrations.parameter_manager import ParameterManagerClient
import google.auth

logger = logging.getLogger("google_adk." + __name__)

# Parameter Manager resources are regional; "global" is the default location.
LOCATION = os.getenv("PARAMETER_MANAGER_LOCATION") or "global"

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT") or google.auth.default()[1]
if not PROJECT_ID:
  raise ValueError(
      "Set GOOGLE_CLOUD_PROJECT, or use credentials that carry a project."
  )

# Both parameters reach the model (system prompt and tool output), so neither
# may hold secrets or Secret Manager `__REF__(...)` references.
INSTRUCTION_PARAMETER_ID = (
    os.getenv("ADK_INSTRUCTION_PARAMETER_ID") or "adk-sample-instruction"
)
CONFIG_PARAMETER_ID = (
    os.getenv("ADK_CONFIG_PARAMETER_ID") or "adk-sample-config"
)

_FALLBACK_INSTRUCTION = textwrap.dedent("""\
    You are a support assistant. Answer briefly and politely.
""")

# ADK resolves the instruction on every LLM request, including after each tool
# call, so the rendered parameter is cached for a short TTL instead of costing
# one Parameter Manager RPC per model call.
_INSTRUCTION_TTL_SECONDS = 30
_cached_instruction: str | None = None
_cached_at = 0.0

_parameter_client = ParameterManagerClient(
    location=None if LOCATION == "global" else LOCATION
)


def _resource_name(parameter_id: str) -> str:
  return (
      f"projects/{PROJECT_ID}/locations/{LOCATION}"
      f"/parameters/{parameter_id}/versions/latest"
  )


async def instruction_from_parameter(ctx: ReadonlyContext) -> str:
  """Loads the agent instruction from Parameter Manager, cached for a short TTL."""
  global _cached_instruction, _cached_at
  del ctx
  now = time.monotonic()
  if (
      _cached_instruction is not None
      and now - _cached_at < _INSTRUCTION_TTL_SECONDS
  ):
    return _cached_instruction

  resource_name = _resource_name(INSTRUCTION_PARAMETER_ID)
  try:
    _cached_instruction = await asyncio.to_thread(
        _parameter_client.get_parameter, resource_name
    )
    _cached_at = now
  except Exception:  # pylint: disable=broad-exception-caught
    logger.warning(
        "Could not read instruction from %s; using the cached or built-in"
        " fallback instruction.",
        resource_name,
        exc_info=True,
    )
  return _cached_instruction or _FALLBACK_INSTRUCTION


async def get_runtime_config(key: str | None = None) -> dict[str, object]:
  """Reads the agent's runtime configuration from Parameter Manager.

  The configuration is a JSON object, for example feature flags, supported
  regions or escalation contacts.

  Args:
    key: Optional top-level key to return. Returns the whole object when
      omitted.

  Returns:
    A dict with the resource name and the requested configuration, or an
    error description if the parameter cannot be read or parsed.
  """
  resource_name = _resource_name(CONFIG_PARAMETER_ID)

  def _error(message: str) -> dict[str, object]:
    return {"status": "error", "resource_name": resource_name, "error": message}

  try:
    raw = await asyncio.to_thread(
        _parameter_client.get_parameter, resource_name
    )
    config = json.loads(raw)
  except json.JSONDecodeError as e:
    return _error(f"Parameter is not valid JSON: {e}")
  except Exception as e:  # pylint: disable=broad-exception-caught
    return _error(f"{type(e).__name__}: {e}")

  if not isinstance(config, dict):
    return _error(
        f"Parameter must be a JSON object, got {type(config).__name__}."
    )
  if key is not None and key not in config:
    return _error(f"Key {key!r} not found. Available: {sorted(config)}")
  return {
      "status": "ok",
      "resource_name": resource_name,
      "config": config[key] if key is not None else config,
  }


root_agent = Agent(
    name="parameter_manager_agent",
    description=(
        "Support assistant whose instruction and runtime configuration are"
        " managed centrally in Google Cloud Parameter Manager."
    ),
    instruction=instruction_from_parameter,
    tools=[get_runtime_config],
)
