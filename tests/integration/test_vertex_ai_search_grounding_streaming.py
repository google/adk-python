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

"""Integration tests for grounding metadata preservation in SSE streaming.

Verifies that grounding_metadata from VertexAiSearchTool reaches the final
non-partial event in both progressive and non-progressive SSE streaming modes.

Prerequisites:
  - GOOGLE_CLOUD_PROJECT env var set to a GCP project with Vertex AI enabled
  - Discovery Engine API enabled (discoveryengine.googleapis.com)
  - Authenticated via `gcloud auth application-default login`

Usage:
  GOOGLE_CLOUD_PROJECT=my-project pytest
  tests/integration/test_vertex_ai_search_grounding_streaming.py -v -s
"""

from __future__ import annotations

import json
import os
import time
import uuid

from google.adk.features._feature_registry import FeatureName
from google.adk.features._feature_registry import temporary_feature_override
from google.genai import types
import pytest

_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
_COLLECTION = "default_collection"
_DATA_STORE_ID = f"adk-grounding-test-{uuid.uuid4().hex[:8]}"
_DATA_STORE_DISPLAY_NAME = "ADK Grounding Integration Test"
_MODEL = "gemini-2.0-flash"

_TEST_DOCUMENTS = (
    {
        "id": "doc-adk-overview",
        "title": "ADK Overview",
        "content": (
            "The Agent Development Kit (ADK) is an open-source framework by"
            " Google for building AI agents. ADK supports multi-agent"
            " architectures, tool use, and integrates with Gemini models."
            " ADK was first released in April 2025."
        ),
    },
    {
        "id": "doc-adk-tools",
        "title": "ADK Built-in Tools",
        "content": (
            "ADK provides built-in tools including VertexAiSearchTool for"
            " grounded search, GoogleSearchTool for web search, and"
            " CodeExecutionTool for running code. The VertexAiSearchTool"
            " returns grounding metadata with citations pointing to source"
            " documents."
        ),
    },
)


def _parent_path() -> str:
  return f"projects/{_PROJECT}/locations/{_LOCATION}/collections/{_COLLECTION}"


def _data_store_path() -> str:
  return f"{_parent_path()}/dataStores/{_DATA_STORE_ID}"


@pytest.fixture(scope="module")
def project_id():
  if not _PROJECT:
    pytest.skip("GOOGLE_CLOUD_PROJECT env var not set")
  return _PROJECT


@pytest.fixture(scope="module")
def data_store_resource(project_id) -> str:
  """Create a Vertex AI Search data store with test documents."""
  from google.api_core.exceptions import AlreadyExists
  from google.cloud import discoveryengine_v1beta as discoveryengine

  ds_client = discoveryengine.DataStoreServiceClient()
  doc_client = discoveryengine.DocumentServiceClient()

  # Create data store
  try:
    request = discoveryengine.CreateDataStoreRequest(
        parent=_parent_path(),
        data_store=discoveryengine.DataStore(
            display_name=_DATA_STORE_DISPLAY_NAME,
            industry_vertical=discoveryengine.IndustryVertical.GENERIC,
            solution_types=[discoveryengine.SolutionType.SOLUTION_TYPE_SEARCH],
            content_config=discoveryengine.DataStore.ContentConfig.NO_CONTENT,
        ),
        data_store_id=_DATA_STORE_ID,
    )
    operation = ds_client.create_data_store(request=request)
    print(f"\nCreating data store '{_DATA_STORE_ID}'...")
    operation.result(timeout=120)
    print("Data store created.")
  except AlreadyExists:
    print(f"\nData store '{_DATA_STORE_ID}' already exists, reusing.")

  # Ingest test documents
  branch = f"{_data_store_path()}/branches/default_branch"
  for doc_data in _TEST_DOCUMENTS:
    json_data = json.dumps({
        "title": doc_data["title"],
        "description": doc_data["content"],
    })
    doc = discoveryengine.Document(
        id=doc_data["id"],
        json_data=json_data,
    )
    try:
      doc_client.create_document(
          parent=branch,
          document=doc,
          document_id=doc_data["id"],
      )
      print(f"  Created document: {doc_data['id']}")
    except AlreadyExists:
      doc_client.update_document(
          document=discoveryengine.Document(
              name=f"{branch}/documents/{doc_data['id']}",
              json_data=json_data,
          ),
      )
      print(f"  Updated document: {doc_data['id']}")

  print("Waiting 5s for indexing...")
  time.sleep(5)

  yield _data_store_path()

  # Cleanup — best-effort, ignore errors from Discovery Engine LRO
  try:
    operation = ds_client.delete_data_store(name=_data_store_path())
    operation.result(timeout=120)
    print(f"\nDeleted data store '{_DATA_STORE_ID}'.")
  except Exception as e:
    print(f"\nFailed to delete data store '{_DATA_STORE_ID}': {e}")


class TestIntegrationVertexAiSearchGrounding:
  """Integration tests hitting real Vertex AI with VertexAiSearchTool."""

  @pytest.mark.parametrize("llm_backend", ["VERTEX"], indirect=True)
  @pytest.mark.parametrize(
      "progressive_sse, label",
      [
          (True, "Progressive SSE"),
          (False, "Non-Progressive SSE"),
      ],
  )
  @pytest.mark.asyncio
  async def test_grounding_metadata_with_sse_streaming(
      self, project_id, data_store_resource, progressive_sse, label
  ):
    """Verifies grounding_metadata in SSE streaming modes."""
    from google.adk.agents.llm_agent import LlmAgent
    from google.adk.tools.vertex_ai_search_tool import VertexAiSearchTool

    agent = LlmAgent(
        name="test_agent",
        model=_MODEL,
        tools=[VertexAiSearchTool(data_store_id=data_store_resource)],
        instruction="Answer questions using the search tool.",
    )

    with temporary_feature_override(
        FeatureName.PROGRESSIVE_SSE_STREAMING, progressive_sse
    ):
      all_events, saved_events = await self._run_agent_streaming(
          agent, project_id
      )

    self._report_events(label, all_events, saved_events)

    saved_with_grounding = [e for e in saved_events if e["has_grounding"]]
    assert (
        saved_with_grounding
    ), f"No saved (non-partial) events have grounding_metadata with {label}."

  @pytest.mark.parametrize("llm_backend", ["VERTEX"], indirect=True)
  @pytest.mark.asyncio
  async def test_grounding_metadata_without_streaming(
      self, project_id, data_store_resource
  ):
    """Without streaming, grounding_metadata should always be present."""
    from google.adk.agents.llm_agent import LlmAgent
    from google.adk.agents.run_config import RunConfig
    from google.adk.agents.run_config import StreamingMode
    from google.adk.runners import Runner
    from google.adk.sessions.in_memory_session_service import InMemorySessionService
    from google.adk.tools.vertex_ai_search_tool import VertexAiSearchTool
    from google.adk.utils.context_utils import Aclosing

    agent = LlmAgent(
        name="test_agent",
        model=_MODEL,
        tools=[VertexAiSearchTool(data_store_id=data_store_resource)],
        instruction="Answer questions using the search tool.",
    )

    session_service = InMemorySessionService()
    runner = Runner(
        app_name="test_app",
        agent=agent,
        session_service=session_service,
    )
    session = await session_service.create_session(
        app_name="test_app", user_id="test_user"
    )

    run_config = RunConfig(streaming_mode=StreamingMode.NONE)
    events = []
    async with Aclosing(
        runner.run_async(
            user_id="test_user",
            session_id=session.id,
            new_message=types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text="What built-in tools does ADK provide?"
                    )
                ],
            ),
            run_config=run_config,
        )
    ) as agen:
      async for event in agen:
        events.append({
            "author": event.author,
            "partial": event.partial,
            "has_grounding": event.grounding_metadata is not None,
            "has_content": bool(event.content and event.content.parts),
        })

    print("\n=== No Streaming ===")
    for i, e in enumerate(events):
      print(
          f"  Event {i}: author={e['author']}, partial={e['partial']},"
          f" grounding={e['has_grounding']}, content={e['has_content']}"
      )

    model_events = [e for e in events if e["author"] == "test_agent"]
    with_grounding = [e for e in model_events if e["has_grounding"]]
    assert (
        with_grounding
    ), "No events have grounding_metadata even without streaming."

  async def _run_agent_streaming(self, agent, project_id):
    from google.adk.agents.run_config import RunConfig
    from google.adk.agents.run_config import StreamingMode
    from google.adk.runners import Runner
    from google.adk.sessions.in_memory_session_service import InMemorySessionService
    from google.adk.utils.context_utils import Aclosing

    session_service = InMemorySessionService()
    runner = Runner(
        app_name="test_app",
        agent=agent,
        session_service=session_service,
    )
    session = await session_service.create_session(
        app_name="test_app", user_id="test_user"
    )

    run_config = RunConfig(streaming_mode=StreamingMode.SSE)
    all_events = []
    async with Aclosing(
        runner.run_async(
            user_id="test_user",
            session_id=session.id,
            new_message=types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text="What is ADK and when was it first released?"
                    )
                ],
            ),
            run_config=run_config,
        )
    ) as agen:
      async for event in agen:
        all_events.append({
            "author": event.author,
            "partial": event.partial,
            "has_grounding": event.grounding_metadata is not None,
            "has_content": bool(event.content and event.content.parts),
        })

    saved_events = [e for e in all_events if e["partial"] is not True]
    return all_events, saved_events

  def _report_events(self, label, all_events, saved_events):
    print(f"\n=== {label} — All Events ===")
    for i, e in enumerate(all_events):
      print(
          f"  Event {i}: author={e['author']}, partial={e['partial']},"
          f" grounding={e['has_grounding']},"
          f" content={e['has_content']}"
      )
    print(f"\n=== {label} — Saved (non-partial) Events ===")
    for i, e in enumerate(saved_events):
      print(
          f"  Event {i}: author={e['author']}, partial={e['partial']},"
          f" grounding={e['has_grounding']},"
          f" content={e['has_content']}"
      )
    partial_with_grounding = [
        e for e in all_events if e["partial"] is True and e["has_grounding"]
    ]
    if partial_with_grounding:
      print(
          f"\n  NOTE: {len(partial_with_grounding)} partial event(s)"
          " had grounding_metadata but were NOT saved to session."
      )
