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

"""Unit tests for reasoning-effort helpers in _openai_common."""

import asyncio
import logging

from google.adk.integrations.openai._openai_common import build_api_key
from google.adk.integrations.openai._openai_common import build_reasoning_effort
from google.adk.integrations.openai._openai_common import is_reasoning_model
from google.adk.integrations.openai._openai_common import OpenAIGenerateContentConfig
from google.adk.integrations.openai._openai_common import supported_efforts
from google.adk.integrations.openai._openai_common import targets_default_openai_host
from google.genai import types
import pytest

# ---------------------------------------------------------------------------
# supported_efforts: per-model tier gating
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model,api,expected",
    [
        # Advanced flagships on Chat Completions: low..xhigh (no minimal/max).
        ("gpt-6-astra", "chat", {"low", "medium", "high", "xhigh"}),
        ("gpt-5.6-sol", "chat", {"low", "medium", "high", "xhigh"}),
        ("gpt-5.5", "chat", {"low", "medium", "high", "xhigh"}),
        ("openai/gpt-6", "chat", {"low", "medium", "high", "xhigh"}),
        # Advanced flagships on Responses: add max (still no minimal).
        (
            "gpt-6-astra",
            "responses",
            {"low", "medium", "high", "xhigh", "max"},
        ),
        (
            "gpt-5.6-sol",
            "responses",
            {"low", "medium", "high", "xhigh", "max"},
        ),
        # gpt-5 family: minimal..high (surface-independent here).
        ("gpt-5", "chat", {"minimal", "low", "medium", "high"}),
        ("gpt-5", "responses", {"minimal", "low", "medium", "high"}),
        ("gpt-5-mini", "chat", {"minimal", "low", "medium", "high"}),
        ("gpt-5-nano", "responses", {"minimal", "low", "medium", "high"}),
        ("gpt-5-2025-08-07", "chat", {"minimal", "low", "medium", "high"}),
        # o-series: low..high, no minimal.
        ("o1", "chat", {"low", "medium", "high"}),
        ("o3", "responses", {"low", "medium", "high"}),
        ("o3-mini", "chat", {"low", "medium", "high"}),
        ("o4-mini", "responses", {"low", "medium", "high"}),
        # gpt-5.x outside 5.5/5.6 (e.g. gpt-5.1) matches neither the advanced
        # nor the plain-gpt-5 pattern, so it takes the conservative default.
        ("gpt-5.1", "chat", {"low", "medium", "high"}),
        ("gpt-5.1", "responses", {"low", "medium", "high"}),
        # No reasoning-effort parameter at all.
        ("o1-mini", "chat", set()),
        ("o1-preview", "chat", set()),
        ("o1-preview-2024-09-12", "responses", set()),
        ("gpt-4o", "chat", set()),
        ("gpt-4.1", "responses", set()),
        ("xai/grok-4.6", "chat", set()),
        (None, "chat", set()),
    ],
)
def test_supported_efforts(model, api, expected):
  assert supported_efforts(model, api) == frozenset(expected)


# ---------------------------------------------------------------------------
# OpenAIGenerateContentConfig
# ---------------------------------------------------------------------------


def test_config_effort_field_defaults_none():
  assert OpenAIGenerateContentConfig().effort is None


def test_config_accepts_effort():
  assert OpenAIGenerateContentConfig(effort="high").effort == "high"


def test_config_rejects_thinking_level():
  with pytest.raises(
      ValueError, match="is not supported in OpenAIGenerateContentConfig"
  ):
    OpenAIGenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_level=types.ThinkingLevel.HIGH
        )
    )


def test_config_rejects_thinking_budget():
  # thinking_budget is treated as unsupported by build_reasoning_effort, so the
  # config validator rejects it too rather than silently ignoring it.
  with pytest.raises(
      ValueError, match="is not supported in OpenAIGenerateContentConfig"
  ):
    OpenAIGenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=1024)
    )


def test_config_allows_thinking_config_without_level_or_budget():
  # A ThinkingConfig with neither thinking_level nor thinking_budget (e.g.
  # include_thoughts) is fine.
  cfg = OpenAIGenerateContentConfig(
      thinking_config=types.ThinkingConfig(include_thoughts=True)
  )
  assert cfg.thinking_config.include_thoughts is True


# ---------------------------------------------------------------------------
# build_reasoning_effort
# ---------------------------------------------------------------------------


def test_build_effort_none_config():
  assert build_reasoning_effort(None, "gpt-5", "chat") is None


def test_build_effort_from_openai_config():
  cfg = OpenAIGenerateContentConfig(effort="minimal")
  assert build_reasoning_effort(cfg, "gpt-5", "chat") == "minimal"


def test_build_effort_advanced_tier_responses():
  cfg = OpenAIGenerateContentConfig(effort="max")
  assert build_reasoning_effort(cfg, "gpt-6-astra", "responses") == "max"


def test_build_effort_max_rejected_on_chat_allowed_on_responses():
  """Advanced models accept ``max`` on Responses but not Chat Completions."""
  cfg = OpenAIGenerateContentConfig(effort="max")
  assert build_reasoning_effort(cfg, "gpt-6-astra", "responses") == "max"
  with pytest.raises(ValueError, match="on the chat API"):
    build_reasoning_effort(cfg, "gpt-6-astra", "chat")


def test_build_effort_none_when_not_configured():
  cfg = OpenAIGenerateContentConfig()
  assert build_reasoning_effort(cfg, "gpt-5", "chat") is None


def test_build_effort_plain_config_returns_none():
  # A plain GenerateContentConfig with no thinking config: nothing to send.
  assert (
      build_reasoning_effort(types.GenerateContentConfig(), "gpt-5", "chat")
      is None
  )


def test_build_effort_unsupported_tier_raises():
  cfg = OpenAIGenerateContentConfig(effort="minimal")
  with pytest.raises(ValueError, match="not supported by model 'o3'"):
    build_reasoning_effort(cfg, "o3", "chat")


def test_build_effort_xhigh_unsupported_on_gpt5():
  cfg = OpenAIGenerateContentConfig(effort="xhigh")
  with pytest.raises(ValueError, match="not supported by model 'gpt-5'"):
    build_reasoning_effort(cfg, "gpt-5", "chat")


def test_build_effort_non_reasoning_model_raises():
  cfg = OpenAIGenerateContentConfig(effort="high")
  with pytest.raises(ValueError, match="does not accept a reasoning effort"):
    build_reasoning_effort(cfg, "gpt-4o", "chat")


@pytest.mark.parametrize("model", ["o1-mini", "o1-preview"])
def test_build_effort_o1_without_effort_raises(model):
  cfg = OpenAIGenerateContentConfig(effort="low")
  with pytest.raises(ValueError, match="does not accept a reasoning effort"):
    build_reasoning_effort(cfg, model, "chat")


def test_build_effort_validate_false_passes_through():
  # validate=False is for OpenAI-compatible backends whose model id does not
  # reveal the accepted tiers (e.g. Grok on Vertex AI): the tier is passed
  # through and the backend rejects an unsupported one. The same call raises
  # under validate=True because the id is not a known reasoning model.
  cfg = OpenAIGenerateContentConfig(effort="high")
  assert build_reasoning_effort(cfg, "grok-4", "chat", validate=False) == "high"
  with pytest.raises(ValueError, match="does not accept a reasoning effort"):
    build_reasoning_effort(cfg, "grok-4", "chat")


def test_build_effort_warns_and_ignores_thinking_level(caplog):
  cfg = types.GenerateContentConfig(
      thinking_config=types.ThinkingConfig(
          thinking_level=types.ThinkingLevel.HIGH
      )
  )
  with caplog.at_level(logging.WARNING):
    assert build_reasoning_effort(cfg, "gpt-5", "chat") is None
  assert "not supported for OpenAI models" in caplog.text


def test_build_effort_warns_and_ignores_thinking_budget(caplog):
  cfg = types.GenerateContentConfig(
      thinking_config=types.ThinkingConfig(thinking_budget=1024)
  )
  with caplog.at_level(logging.WARNING):
    assert build_reasoning_effort(cfg, "gpt-5", "responses") is None
  assert "not supported for OpenAI models" in caplog.text


# ---------------------------------------------------------------------------
# is_reasoning_model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model,expected",
    [
        # o-series and gpt-5/6 families are reasoning models.
        ("o1", True),
        ("o3-mini", True),
        ("o4-mini", True),
        ("gpt-5", True),
        ("gpt-5-mini", True),
        ("gpt-5.6-sol", True),
        ("gpt-6-astra", True),
        ("openai/gpt-5", True),
        # Namespaced deployments are matched too.
        ("openai/o3-mini", True),
        ("azure/o1", True),
        # Regression: the -chat variants are NON-reasoning chat models and must
        # keep temperature / top_p. The old regex matched these by mistake.
        ("gpt-5-chat", False),
        ("gpt-5-chat-latest", False),
        ("gpt-5.1-chat-latest", False),
        # Classic chat models and non-OpenAI models are not reasoning models.
        ("gpt-4o", False),
        ("gpt-4.1", False),
        ("xai/grok-4.6", False),
        (None, False),
        ("", False),
    ],
)
def test_is_reasoning_model(model, expected):
  assert is_reasoning_model(model) is expected


# ---------------------------------------------------------------------------
# build_api_key
# ---------------------------------------------------------------------------


def test_build_api_key_string_passthrough():
  assert build_api_key("secret") == "secret"


def test_build_api_key_none_passthrough():
  assert build_api_key(None) is None


def test_build_api_key_wraps_sync_callable_as_async_provider():
  calls = {"n": 0}

  def provider() -> str:
    calls["n"] += 1
    return f"token-{calls['n']}"

  wrapped = build_api_key(provider)
  # A callable is adapted into an async provider; nothing is resolved eagerly.
  assert callable(wrapped)
  assert calls["n"] == 0
  # The SDK awaits the provider on every request, re-resolving each time.
  assert asyncio.run(wrapped()) == "token-1"
  assert asyncio.run(wrapped()) == "token-2"


def test_build_api_key_wraps_async_callable_as_async_provider():
  calls = {"n": 0}

  async def provider() -> str:
    calls["n"] += 1
    return f"token-{calls['n']}"

  wrapped = build_api_key(provider)
  assert callable(wrapped)
  assert calls["n"] == 0
  assert asyncio.run(wrapped()) == "token-1"
  assert asyncio.run(wrapped()) == "token-2"


# ---------------------------------------------------------------------------
# targets_default_openai_host: the predicate gating tier validation
# ---------------------------------------------------------------------------


@pytest.fixture
def _clean_openai_env(monkeypatch):
  """Removes the backend-selecting env vars so branches are tested in isolation."""
  monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
  monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)


def test_targets_default_host_true_when_nothing_overrides(_clean_openai_env):
  assert targets_default_openai_host(client=None, base_url=None) is True


def test_targets_default_host_false_with_injected_client(_clean_openai_env):
  assert targets_default_openai_host(client=object(), base_url=None) is False


def test_targets_default_host_false_with_base_url(_clean_openai_env):
  assert (
      targets_default_openai_host(
          client=None, base_url="https://host.example/v1"
      )
      is False
  )


def test_targets_default_host_false_with_azure_endpoint(_clean_openai_env):
  assert (
      targets_default_openai_host(
          client=None,
          base_url=None,
          azure_endpoint="https://x.openai.azure.com",
      )
      is False
  )


def test_targets_default_host_false_with_openai_base_url_env(
    _clean_openai_env, monkeypatch
):
  monkeypatch.setenv("OPENAI_BASE_URL", "https://host.example/v1")
  assert targets_default_openai_host(client=None, base_url=None) is False


def test_targets_default_host_ignores_azure_endpoint_env(
    _clean_openai_env, monkeypatch
):
  # No wrapper reads AZURE_OPENAI_ENDPOINT, so a request made while it is set
  # still reaches the default host and must still be validated.
  monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://x.openai.azure.com")
  assert targets_default_openai_host(client=None, base_url=None) is True


def test_targets_default_host_ignores_empty_env(_clean_openai_env, monkeypatch):
  # An empty string is falsy, so it does not count as an override.
  monkeypatch.setenv("OPENAI_BASE_URL", "")
  assert targets_default_openai_host(client=None, base_url=None) is True


@pytest.mark.parametrize("api", ["chatt", "", "completions"])
def test_supported_efforts_rejects_unknown_api_surface(api):
  # The surface is validated up front, for reasoning and non-reasoning models.
  for model in ("gpt-5.6-sol", "gpt-4o", None):
    with pytest.raises(ValueError, match="Unknown API surface"):
      supported_efforts(model, api)
