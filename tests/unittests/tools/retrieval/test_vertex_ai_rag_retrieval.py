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

import dataclasses
from types import SimpleNamespace
from typing import Optional

from google.adk.agents.llm_agent import Agent
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.retrieval.vertex_ai_rag_retrieval import VertexAiRagRetrieval
from google.genai import types
import pytest

from ... import testing_utils

_CORPUS = "projects/p/locations/l/ragCorpora/c"


@dataclasses.dataclass
class _LegacyRagResource:
  """Stands in for vertexai.rag.RagResource.

  ADK no longer depends on google-cloud-aiplatform, so the compatibility path
  is exercised against the shape rather than the class.
  """

  rag_corpus: Optional[str] = None
  rag_file_ids: Optional[list[str]] = None


def noop_tool(x: str) -> str:
  return x


def _mock_client(mocker, contexts=()):
  """Patches agentplatform.Client and returns (factory, client)."""
  client = mocker.Mock()
  client.aio.rag.retrieve_contexts = mocker.AsyncMock()
  client.aio.aclose = mocker.AsyncMock()
  client.aio.rag.retrieve_contexts.return_value = SimpleNamespace(
      contexts=SimpleNamespace(contexts=list(contexts))
  )
  factory = mocker.patch("agentplatform.Client", return_value=client)
  return factory, client


def _retrieve_kwargs(client):
  return client.aio.rag.retrieve_contexts.call_args.kwargs


def test_rag_resources_accept_genai_type():
  retrieval = VertexAiRagRetrieval(
      name="rag_retrieval",
      description="rag_retrieval",
      rag_resources=[
          types.VertexRagStoreRagResource(
              rag_corpus=_CORPUS, rag_file_ids=["file-1"]
          )
      ],
  )

  assert retrieval.vertex_rag_store.rag_resources == [
      types.VertexRagStoreRagResource(
          rag_corpus=_CORPUS, rag_file_ids=["file-1"]
      )
  ]


def test_rag_resources_accept_legacy_vertexai_resource():
  # The pre-migration public API took vertexai.rag.RagResource, so anything
  # carrying its two fields has to keep working.
  retrieval = VertexAiRagRetrieval(
      name="rag_retrieval",
      description="rag_retrieval",
      rag_resources=[
          _LegacyRagResource(rag_corpus=_CORPUS, rag_file_ids=["file-1"])
      ],
  )

  assert retrieval.vertex_rag_store.rag_resources == [
      types.VertexRagStoreRagResource(
          rag_corpus=_CORPUS, rag_file_ids=["file-1"]
      )
  ]


def test_rag_resources_accept_real_vertexai_resource():
  rag = pytest.importorskip(
      "vertexai.preview.rag",
      reason="google-cloud-aiplatform is an optional dependency.",
  )

  retrieval = VertexAiRagRetrieval(
      name="rag_retrieval",
      description="rag_retrieval",
      rag_resources=[
          rag.RagResource(rag_corpus=_CORPUS, rag_file_ids=["file-1"])
      ],
  )

  assert retrieval.vertex_rag_store.rag_resources == [
      types.VertexRagStoreRagResource(
          rag_corpus=_CORPUS, rag_file_ids=["file-1"]
      )
  ]


def test_rag_resources_accept_dicts():
  retrieval = VertexAiRagRetrieval(
      name="rag_retrieval",
      description="rag_retrieval",
      rag_resources=[{"rag_corpus": _CORPUS, "rag_file_ids": ["file-1"]}],
  )

  assert retrieval.vertex_rag_store.rag_resources == [
      types.VertexRagStoreRagResource(
          rag_corpus=_CORPUS, rag_file_ids=["file-1"]
      )
  ]


def test_rag_resources_accept_mixed_forms():
  retrieval = VertexAiRagRetrieval(
      name="rag_retrieval",
      description="rag_retrieval",
      rag_resources=[
          types.VertexRagStoreRagResource(rag_corpus="a"),
          _LegacyRagResource(rag_corpus="b"),
          {"rag_corpus": "c"},
      ],
  )

  assert [
      resource.rag_corpus
      for resource in retrieval.vertex_rag_store.rag_resources
  ] == ["a", "b", "c"]


def test_rag_resources_reject_unshaped_objects():
  with pytest.raises(TypeError, match="rag_corpus"):
    VertexAiRagRetrieval(
        name="rag_retrieval",
        description="rag_retrieval",
        rag_resources=[object()],
    )


@pytest.mark.asyncio
async def test_run_async_derives_project_and_location_from_the_store_name(
    mocker, monkeypatch: pytest.MonkeyPatch
):
  monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
  monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)
  retrieval = VertexAiRagRetrieval(
      name="rag_retrieval",
      description="rag_retrieval",
      rag_resources=[
          types.VertexRagStoreRagResource(
              rag_corpus=_CORPUS, rag_file_ids=["file-1"]
          )
      ],
  )
  factory, client = _mock_client(mocker)

  await retrieval.run_async(args={"query": "q"}, tool_context=mocker.Mock())

  factory.assert_called_once_with(project="p", location="l")
  client.aio.rag.retrieve_contexts.assert_awaited_once()
  kwargs = _retrieve_kwargs(client)
  assert kwargs["vertex_rag_store"].rag_resources == [
      types.VertexRagStoreRagResource(
          rag_corpus=_CORPUS, rag_file_ids=["file-1"]
      )
  ]
  assert kwargs["query"].text == "q"


@pytest.mark.asyncio
async def test_run_async_keeps_similarity_top_k_off_the_store(mocker):
  # RetrieveContextsRequest.VertexRagStore has no similarity_top_k -- the field
  # number is reserved -- so sending it makes the backend reject the request.
  retrieval = VertexAiRagRetrieval(
      name="rag_retrieval",
      description="rag_retrieval",
      rag_corpora=[_CORPUS],
      similarity_top_k=7,
      vector_distance_threshold=0.4,
  )
  _, client = _mock_client(mocker)

  await retrieval.run_async(args={"query": "q"}, tool_context=mocker.Mock())

  store = _retrieve_kwargs(client)["vertex_rag_store"]
  assert store.similarity_top_k is None
  assert store.rag_corpora == [_CORPUS]
  assert store.vector_distance_threshold == 0.4
  # The tool still advertises top-k to Gemini through the untrimmed store.
  assert retrieval.vertex_rag_store.similarity_top_k == 7


@pytest.mark.asyncio
async def test_run_async_passes_similarity_top_k_on_the_query(mocker):
  retrieval = VertexAiRagRetrieval(
      name="rag_retrieval",
      description="rag_retrieval",
      rag_corpora=[_CORPUS],
      similarity_top_k=7,
  )
  _, client = _mock_client(mocker)

  await retrieval.run_async(args={"query": "q"}, tool_context=mocker.Mock())

  assert _retrieve_kwargs(client)["query"].similarity_top_k == 7


@pytest.mark.asyncio
async def test_run_async_closes_the_client(mocker):
  retrieval = VertexAiRagRetrieval(
      name="rag_retrieval",
      description="rag_retrieval",
      rag_corpora=[_CORPUS],
  )
  _, client = _mock_client(mocker)

  await retrieval.run_async(args={"query": "q"}, tool_context=mocker.Mock())

  client.aio.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_async_closes_the_client_when_retrieval_fails(mocker):
  retrieval = VertexAiRagRetrieval(
      name="rag_retrieval",
      description="rag_retrieval",
      rag_corpora=[_CORPUS],
  )
  _, client = _mock_client(mocker)
  client.aio.rag.retrieve_contexts.side_effect = RuntimeError("boom")

  with pytest.raises(RuntimeError, match="boom"):
    await retrieval.run_async(args={"query": "q"}, tool_context=mocker.Mock())

  client.aio.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_async_returns_matching_contexts(mocker):
  retrieval = VertexAiRagRetrieval(
      name="rag_retrieval",
      description="rag_retrieval",
      rag_corpora=[_CORPUS],
  )
  _mock_client(
      mocker,
      contexts=[
          SimpleNamespace(text="chunk 1"),
          SimpleNamespace(text="chunk 2"),
      ],
  )

  result = await retrieval.run_async(
      args={"query": "q"}, tool_context=mocker.Mock()
  )

  assert result == ["chunk 1", "chunk 2"]


@pytest.mark.asyncio
async def test_run_async_reports_when_nothing_matches(mocker):
  retrieval = VertexAiRagRetrieval(
      name="rag_retrieval",
      description="rag_retrieval",
      rag_corpora=[_CORPUS],
  )
  _mock_client(mocker)

  result = await retrieval.run_async(
      args={"query": "q"}, tool_context=mocker.Mock()
  )

  assert isinstance(result, str)
  assert result.startswith("No matching result found")


@pytest.mark.asyncio
async def test_run_async_rejects_a_non_string_query(mocker):
  retrieval = VertexAiRagRetrieval(
      name="rag_retrieval",
      description="rag_retrieval",
      rag_corpora=[_CORPUS],
  )
  _mock_client(mocker)

  with pytest.raises(ValueError, match="string"):
    await retrieval.run_async(args={"query": 42}, tool_context=mocker.Mock())


@pytest.mark.asyncio
async def test_run_async_prefers_the_corpus_name_over_the_environment(
    mocker, monkeypatch: pytest.MonkeyPatch
):
  # A fully-qualified corpus name identifies exactly one endpoint, so an
  # unrelated GOOGLE_CLOUD_PROJECT must not redirect the request elsewhere.
  monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "env-project")
  monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "env-location")
  retrieval = VertexAiRagRetrieval(
      name="rag_retrieval",
      description="rag_retrieval",
      rag_corpora=[_CORPUS],
  )
  factory, _ = _mock_client(mocker)

  await retrieval.run_async(args={"query": "q"}, tool_context=mocker.Mock())

  factory.assert_called_once_with(project="p", location="l")


@pytest.mark.asyncio
async def test_run_async_falls_back_to_environment_variables(
    mocker, monkeypatch: pytest.MonkeyPatch
):
  monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "env-project")
  monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "env-location")
  retrieval = VertexAiRagRetrieval(
      name="rag_retrieval",
      description="rag_retrieval",
      rag_corpora=["corpus-without-prefix"],
  )
  factory, _ = _mock_client(mocker)

  await retrieval.run_async(args={"query": "q"}, tool_context=mocker.Mock())

  factory.assert_called_once_with(
      project="env-project", location="env-location"
  )


def test_vertex_rag_retrieval_for_non_gemini():
  responses = [
      "response1",
  ]
  mockModel = testing_utils.MockModel.create(responses=responses)
  mockModel.model = "claude-3-sonnet"

  # Calls the first time.
  agent = Agent(
      name="root_agent",
      model=mockModel,
      tools=[
          VertexAiRagRetrieval(
              name="rag_retrieval",
              description="rag_retrieval",
              rag_corpora=[
                  "projects/123456789/locations/us-central1/ragCorpora/1234567890"
              ],
          )
      ],
  )
  runner = testing_utils.InMemoryRunner(agent)
  events = runner.run("test1")

  # Asserts the requests.
  assert len(mockModel.requests) == 1
  assert testing_utils.simplify_contents(mockModel.requests[0].contents) == [
      ("user", "test1"),
  ]
  assert len(mockModel.requests[0].config.tools) == 1
  assert (
      mockModel.requests[0].config.tools[0].function_declarations[0].name
      == "rag_retrieval"
  )
  assert mockModel.requests[0].tools_dict["rag_retrieval"] is not None


def test_vertex_rag_retrieval_for_non_gemini_with_another_function_tool():
  responses = [
      "response1",
  ]
  mockModel = testing_utils.MockModel.create(responses=responses)
  mockModel.model = "claude-3-sonnet"

  # Calls the first time.
  agent = Agent(
      name="root_agent",
      model=mockModel,
      tools=[
          VertexAiRagRetrieval(
              name="rag_retrieval",
              description="rag_retrieval",
              rag_corpora=[
                  "projects/123456789/locations/us-central1/ragCorpora/1234567890"
              ],
          ),
          FunctionTool(func=noop_tool),
      ],
  )
  runner = testing_utils.InMemoryRunner(agent)
  events = runner.run("test1")

  # Asserts the requests.
  assert len(mockModel.requests) == 1
  assert testing_utils.simplify_contents(mockModel.requests[0].contents) == [
      ("user", "test1"),
  ]
  assert len(mockModel.requests[0].config.tools[0].function_declarations) == 2
  assert (
      mockModel.requests[0].config.tools[0].function_declarations[0].name
      == "rag_retrieval"
  )
  assert (
      mockModel.requests[0].config.tools[0].function_declarations[1].name
      == "noop_tool"
  )
  assert mockModel.requests[0].tools_dict["rag_retrieval"] is not None


def test_vertex_rag_retrieval_for_gemini_2_x():
  responses = [
      "response1",
  ]
  mockModel = testing_utils.MockModel.create(responses=responses)
  mockModel.model = "gemini-2.5-flash"

  # Calls the first time.
  agent = Agent(
      name="root_agent",
      model=mockModel,
      tools=[
          VertexAiRagRetrieval(
              name="rag_retrieval",
              description="rag_retrieval",
              rag_corpora=[
                  "projects/123456789/locations/us-central1/ragCorpora/1234567890"
              ],
          )
      ],
  )
  runner = testing_utils.InMemoryRunner(agent)
  events = runner.run("test1")

  # Asserts the requests.
  assert len(mockModel.requests) == 1
  assert testing_utils.simplify_contents(mockModel.requests[0].contents) == [
      ("user", "test1"),
  ]
  assert len(mockModel.requests[0].config.tools) == 1
  assert mockModel.requests[0].config.tools == [
      types.Tool(
          retrieval=types.Retrieval(
              vertex_rag_store=types.VertexRagStore(
                  rag_corpora=[
                      "projects/123456789/locations/us-central1/ragCorpora/1234567890"
                  ]
              )
          )
      )
  ]
  assert "rag_retrieval" not in mockModel.requests[0].tools_dict


def test_vertex_rag_retrieval_for_non_gemini_with_disabled_check(monkeypatch):
  monkeypatch.setenv("ADK_DISABLE_GEMINI_MODEL_ID_CHECK", "true")
  responses = [
      "response1",
  ]
  mockModel = testing_utils.MockModel.create(responses=responses)
  mockModel.model = "internal-model-v1"

  agent = Agent(
      name="root_agent",
      model=mockModel,
      tools=[
          VertexAiRagRetrieval(
              name="rag_retrieval",
              description="rag_retrieval",
              rag_corpora=[
                  "projects/123456789/locations/us-central1/ragCorpora/1234567890"
              ],
          )
      ],
  )
  runner = testing_utils.InMemoryRunner(agent)
  runner.run("test1")

  assert len(mockModel.requests) == 1
  assert len(mockModel.requests[0].config.tools) == 1
  assert mockModel.requests[0].config.tools == [
      types.Tool(
          retrieval=types.Retrieval(
              vertex_rag_store=types.VertexRagStore(
                  rag_corpora=[
                      "projects/123456789/locations/us-central1/ragCorpora/1234567890"
                  ]
              )
          )
      )
  ]
  assert "rag_retrieval" not in mockModel.requests[0].tools_dict
