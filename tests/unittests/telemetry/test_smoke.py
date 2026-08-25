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

"""The private internals the functional test harness reaches into.

Not ours, not public, and an upgrade may rename or drop them. One test
each, so that a harness that quietly stops doing what it says fails here
rather than as a puzzling golden diff.
"""

from __future__ import annotations

from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.genai.models import AsyncModels
from mcp import StdioServerParameters
from opentelemetry.instrumentation._semconv import _OpenTelemetrySemanticConventionStability
from opentelemetry.instrumentation.google_genai import GoogleGenAiSdkInstrumentor
import pytest

from .functional._scenarios import EXPERIMENTAL_OPT_IN
from .functional._scenarios import gemini_test_model
from .functional._scenarios import OTEL_OPT_IN


def test_semconv_stability_cache_can_be_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """``otel_instrumentor`` resets this to re-read the opt-in per case.

  Without it the instrumentation libraries resolve
  ``OTEL_SEMCONV_STABILITY_OPT_IN`` once per process, and every case after
  the first is recorded under whichever semconv the first one asked for.
  """
  monkeypatch.setattr(
      _OpenTelemetrySemanticConventionStability, "_initialized", False
  )
  monkeypatch.setattr(
      _OpenTelemetrySemanticConventionStability,
      "_OTEL_SEMCONV_STABILITY_SIGNAL_MAPPING",
      {},
  )
  monkeypatch.setenv(OTEL_OPT_IN, EXPERIMENTAL_OPT_IN)

  _OpenTelemetrySemanticConventionStability._initialize()

  assert (
      _OpenTelemetrySemanticConventionStability._OTEL_SEMCONV_STABILITY_SIGNAL_MAPPING
  )
  assert _OpenTelemetrySemanticConventionStability._initialized


def test_mcp_toolset_session_manager_is_patchable() -> None:
  """``build_mcp_test_runner`` swaps these two out for the fake session.

  Without them the scenario would dial a real MCP server over stdio. A
  rename here is why an MCP test would suddenly want a subprocess.
  """
  toolset = McpToolset(
      connection_params=StdioConnectionParams(
          server_params=StdioServerParameters(command="unused-by-test"),
      )
  )

  manager = toolset._mcp_session_manager

  assert callable(manager.create_session)
  assert callable(manager.close)


def test_the_instrumentor_wraps_the_sdk_call_the_harness_mocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """The mock has to go in first, or the instrumentor wraps the real SDK.

  ``gemini_test_model`` patches ``AsyncModels.generate_content`` and
  ``otel_instrumentor`` wraps whatever it finds there; this is what makes
  the OTel recording instrument the mock rather than the network.
  """
  gemini_test_model(monkeypatch)
  mocked = AsyncModels.generate_content

  instrumentor = GoogleGenAiSdkInstrumentor()
  instrumentor.instrument()
  try:
    assert AsyncModels.generate_content is not mocked
    assert AsyncModels.generate_content.__wrapped__ is mocked
  finally:
    instrumentor.uninstrument()

  assert AsyncModels.generate_content is mocked
