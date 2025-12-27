# Copyright 2025 Google LLC
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

from unittest import mock
import pytest

from google.adk.models.google_llm import Gemini
from google.adk.models import llm_response as llm_response_mod
from google.adk.models import gemini_context_cache_manager as cache_mod


@pytest.mark.asyncio
async def test_disable_google_llm_telemetry(monkeypatch, context_manager_with_span):
    monkeypatch.setenv("OTEL_SDK_DISABLED", "false")
    monkeypatch.setenv("ADK_TELEMETRY_DISABLED", "false")
    start_span = mock.Mock(return_value=context_manager_with_span)
    monkeypatch.setattr(
        "google.adk.telemetry.tracing.tracer.start_as_current_span",
        start_span,
    )

    gemini = Gemini(disable_telemetry=True)

    # Avoid real Client construction
    fake_client = mock.MagicMock()
    fake_client.vertexai = False
    fake_client.aio.models.generate_content = mock.AsyncMock(
        return_value=mock.MagicMock()
    )
    gemini.__dict__["api_client"] = fake_client

    # Prevent cache validation code running (the bit that touches expire_time)
    monkeypatch.setattr(
        cache_mod.GeminiContextCacheManager,
        "handle_context_caching",
        mock.AsyncMock(return_value=None),
    )

    req = mock.MagicMock()
    req.cache_config = object()  # force the cache path
    req.model = "gemini-2.5-flash"
    req.contents = []
    req.config = mock.MagicMock()
    req.config.tools = None
    req.config.system_instruction = ""
    req.config.model_dump = mock.Mock(return_value={})
    req.config.http_options = None

    monkeypatch.setattr(
        llm_response_mod.LlmResponse, "create", mock.Mock(return_value=mock.MagicMock())
    )

    async for _ in gemini.generate_content_async(req, stream=False):
        break

    assert start_span.call_count == 0


@pytest.mark.asyncio
async def test_enable_google_llm_telemetry(monkeypatch, context_manager_with_span):
    monkeypatch.setenv("OTEL_SDK_DISABLED", "false")
    monkeypatch.setenv("ADK_TELEMETRY_DISABLED", "false")
    start_span = mock.Mock(return_value=context_manager_with_span)
    monkeypatch.setattr(
        "google.adk.telemetry.tracing.tracer.start_as_current_span",
        start_span,
    )

    gemini = Gemini(disable_telemetry=False)

    # Avoid real Client construction
    fake_client = mock.MagicMock()
    fake_client.vertexai = False
    fake_client.aio.models.generate_content = mock.AsyncMock(
        return_value=mock.MagicMock()
    )
    gemini.__dict__["api_client"] = fake_client

    # Prevent cache validation code running (the bit that touches expire_time)
    monkeypatch.setattr(
        cache_mod.GeminiContextCacheManager,
        "handle_context_caching",
        mock.AsyncMock(return_value=None),
    )

    req = mock.MagicMock()
    req.cache_config = object()  # force the cache path
    req.model = "gemini-2.5-flash"
    req.contents = []
    req.config = mock.MagicMock()
    req.config.tools = None
    req.config.system_instruction = ""
    req.config.model_dump = mock.Mock(return_value={})
    req.config.http_options = None

    monkeypatch.setattr(
        llm_response_mod.LlmResponse, "create", mock.Mock(return_value=mock.MagicMock())
    )

    async for _ in gemini.generate_content_async(req, stream=False):
        break

    assert start_span.call_count > 0
