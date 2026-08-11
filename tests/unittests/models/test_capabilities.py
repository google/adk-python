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

"""Tests for LlmCapabilities and the BaseLlm.capabilities property."""

from __future__ import annotations

import contextlib
from typing import AsyncGenerator
from typing import Iterator
import warnings

from google.adk.models import LlmCapabilities
from google.adk.models.anthropic_llm import Claude
from google.adk.models.apigee_llm import ApigeeLlm
from google.adk.models.base_llm import BaseLlm
from google.adk.models.gemma_llm import Gemma
from google.adk.models.gemma_llm import Gemma3Ollama
from google.adk.models.google_llm import Gemini
from google.adk.models.lite_llm import LiteLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
import pydantic
import pytest


def _disable_enterprise_mode(monkeypatch: pytest.MonkeyPatch) -> None:
  """Clears both env vars that enable enterprise mode."""
  monkeypatch.delenv('GOOGLE_GENAI_USE_ENTERPRISE', raising=False)
  # Consulted as a deprecated fallback when the preferred var is absent.
  monkeypatch.delenv('GOOGLE_GENAI_USE_VERTEXAI', raising=False)


@contextlib.contextmanager
def _assert_no_warning() -> Iterator[None]:
  """Fails if any warning is raised inside the block."""
  with warnings.catch_warnings(record=True) as raised:
    warnings.simplefilter('always')
    yield
  assert not [str(w.message) for w in raised]


class _BareLlm(BaseLlm):
  """A model that adds nothing on top of BaseLlm."""

  model: str = 'bare-model'

  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    yield LlmResponse()


# -- The value object ---------------------------------------------------------


def test_capabilities_are_immutable():
  """Assigning to a resolved capability raises instead of silently no-op."""
  capabilities = LlmCapabilities()

  with pytest.raises(pydantic.ValidationError):
    capabilities.output_schema_and_tools = True


def test_unknown_capability_is_rejected():
  """Constructing with an unknown capability name raises."""
  with pytest.raises(pydantic.ValidationError):
    LlmCapabilities(no_such_capability=True)


def test_model_copy_silently_ignores_an_unknown_capability():
  """Why the documented override builds a new snapshot instead of copying.

  ``model_copy(update=...)`` skips validation, so a misspelled capability name
  attaches as an unrelated attribute while every real capability keeps its old
  value -- no error, and a clean-looking ``model_dump()``. Building a new
  snapshot from the parent's, the way ``BaseLlm.capabilities`` documents,
  validates and therefore raises.
  """
  stale = LlmCapabilities().model_copy(
      update={'output_schema_with_tools': True}
  )

  assert not stale.output_schema_and_tools
  assert stale.model_dump() == {'output_schema_and_tools': False}

  with pytest.raises(pydantic.ValidationError):
    LlmCapabilities(
        **LlmCapabilities().model_dump() | {'output_schema_with_tools': True}
    )


def test_capabilities_is_not_a_serialized_field():
  """capabilities is a property, so it must stay out of the model dump."""
  assert 'capabilities' not in _BareLlm().model_dump()


# -- The deprecated name-based fallback on BaseLlm ----------------------------


def test_fallback_grants_a_gemini_named_model_and_warns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """A model that predates self-reporting keeps resolving as it did before."""
  monkeypatch.setenv('GOOGLE_GENAI_USE_ENTERPRISE', '1')
  model = _BareLlm(model='gemini-2.5-pro')

  with pytest.warns(FutureWarning, match='_BareLlm relies on name-based'):
    assert model.capabilities.output_schema_and_tools


@pytest.mark.parametrize(
    'model, enterprise_mode',
    [
        ('bare-model', '1'),  # Not a Gemini id at all.
        ('gemini-2.5-pro', '0'),  # Not on Vertex AI.
        ('gemini-2.5-pro', None),  # Not on Vertex AI.
    ],
)
def test_fallback_stays_quiet_when_it_denies(
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    enterprise_mode: str | None,
) -> None:
  """The warning only fires for models whose behavior the removal changes."""
  if enterprise_mode is None:
    _disable_enterprise_mode(monkeypatch)
  else:
    monkeypatch.setenv('GOOGLE_GENAI_USE_ENTERPRISE', enterprise_mode)

  with _assert_no_warning():
    assert not _BareLlm(model=model).capabilities.output_schema_and_tools


def test_declaring_capabilities_outright_bypasses_the_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """The documented migration for a BaseLlm subclass silences the warning."""
  monkeypatch.setenv('GOOGLE_GENAI_USE_ENTERPRISE', '1')

  class _SelfReportingLlm(_BareLlm):
    model: str = 'gemini-2.5-pro'

    @property
    def capabilities(self) -> LlmCapabilities:
      return LlmCapabilities(output_schema_and_tools=True)

  with _assert_no_warning():
    assert _SelfReportingLlm().capabilities.output_schema_and_tools


def test_subclass_can_override_a_capability():
  """A subclass can force-enable a capability its parent denies."""

  class _OverridingLlm(_BareLlm):

    @property
    def capabilities(self) -> LlmCapabilities:
      return LlmCapabilities(
          **super().capabilities.model_dump()
          | {'output_schema_and_tools': True}
      )

  assert _OverridingLlm().capabilities.output_schema_and_tools


# -- Models that self-report ---------------------------------------------------


@pytest.mark.parametrize(
    'model, enterprise_mode, expected',
    [
        ('gemini-2.5-pro', '1', True),
        ('gemini-2.5-flash', '1', True),
        ('gemini-2.5-pro', '0', False),
        ('gemini-2.5-pro', None, False),
        ('gemini-early-exp', '1', True),
    ],
)
def test_gemini_output_schema_and_tools(
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    enterprise_mode: str | None,
    expected: bool,
) -> None:
  """Gemini pairs schema with tools only on Vertex AI.

  Declaring the capability itself, it never reaches the fallback on ``BaseLlm``
  and so is never nagged to migrate.
  """
  if enterprise_mode is None:
    _disable_enterprise_mode(monkeypatch)
  else:
    monkeypatch.setenv('GOOGLE_GENAI_USE_ENTERPRISE', enterprise_mode)

  with _assert_no_warning():
    assert Gemini(model=model).capabilities.output_schema_and_tools == expected


def test_gemini_capabilities_follow_environment_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Capabilities are recomputed, not frozen at construction time."""
  _disable_enterprise_mode(monkeypatch)
  gemini = Gemini(model='gemini-2.5-pro')
  assert not gemini.capabilities.output_schema_and_tools

  monkeypatch.setenv('GOOGLE_GENAI_USE_ENTERPRISE', '1')

  assert gemini.capabilities.output_schema_and_tools


def test_gemini_capabilities_follow_model_reassignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """BaseLlm is mutable, so a reassigned model must be re-resolved."""
  monkeypatch.setenv('GOOGLE_GENAI_USE_ENTERPRISE', '1')
  gemini = Gemini(model='not-a-gemini-model')
  assert not gemini.capabilities.output_schema_and_tools

  gemini.model = 'gemini-2.5-pro'

  assert gemini.capabilities.output_schema_and_tools


def test_apigee_inherits_gemini_capabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """ApigeeLlm extends Gemini, so the Gemini rule applies to its model id.

  Its id also passes the fallback on ``BaseLlm``, which would report the same
  value, so the absence of a warning is what distinguishes inheriting Gemini's
  declaration from silently relying on that fallback.
  """
  monkeypatch.setenv('GOOGLE_GENAI_USE_ENTERPRISE', '1')

  with _assert_no_warning():
    assert ApigeeLlm(
        model='apigee/vertex_ai/gemini-2.5-pro'
    ).capabilities.output_schema_and_tools


def test_gemma_does_not_support_output_schema_and_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Gemma extends Gemini but its model id never passes the Gemini check."""
  monkeypatch.setenv('GOOGLE_GENAI_USE_ENTERPRISE', '1')

  assert not Gemma().capabilities.output_schema_and_tools


def test_claude_does_not_support_output_schema_and_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Claude does not self-report and its id fails the name-based fallback."""
  monkeypatch.setenv('GOOGLE_GENAI_USE_ENTERPRISE', '1')

  with _assert_no_warning():
    assert not Claude(
        model='claude-3-7-sonnet@20250219'
    ).capabilities.output_schema_and_tools


def test_litellm_supports_output_schema_and_tools():
  """LiteLLM reconciles schema and tools for every provider it fronts."""
  with _assert_no_warning():
    assert LiteLlm(model='openai/gpt-4o').capabilities.output_schema_and_tools


def test_gemma3_ollama_inherits_litellm_capabilities():
  """Gemma3Ollama extends LiteLlm and inherits its capability."""
  assert Gemma3Ollama().capabilities.output_schema_and_tools
