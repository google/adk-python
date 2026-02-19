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

from unittest.mock import MagicMock

from google.adk.tools.retrieval.base_retrieval_tool import BaseRetrievalTool
from google.adk.tools.retrieval.callable_retrieval import CallableRetrieval
from google.adk.tools.tool_context import ToolContext
import pytest


@pytest.fixture
def mock_tool_context():
  return MagicMock(spec=ToolContext)


def test_isinstance_base_retrieval_tool():
  tool = CallableRetrieval(
      name="test",
      description="A test tool.",
      retriever=lambda query: [],
  )
  assert isinstance(tool, BaseRetrievalTool)


def test_get_declaration():
  tool = CallableRetrieval(
      name="my_search",
      description="Search docs.",
      retriever=lambda query: [],
  )
  declaration = tool._get_declaration()
  assert declaration.name == "my_search"
  assert declaration.description == "Search docs."


@pytest.mark.asyncio
async def test_sync_callable(mock_tool_context):
  def my_retriever(query: str):
    return [f"result for {query}"]

  tool = CallableRetrieval(
      name="sync_tool",
      description="A sync retrieval tool.",
      retriever=my_retriever,
  )
  result = await tool.run_async(
      args={"query": "hello"}, tool_context=mock_tool_context
  )
  assert result == ["result for hello"]


@pytest.mark.asyncio
async def test_async_callable(mock_tool_context):
  async def my_retriever(query: str):
    return [f"async result for {query}"]

  tool = CallableRetrieval(
      name="async_tool",
      description="An async retrieval tool.",
      retriever=my_retriever,
  )
  result = await tool.run_async(
      args={"query": "world"}, tool_context=mock_tool_context
  )
  assert result == ["async result for world"]


@pytest.mark.asyncio
async def test_tool_context_passthrough(mock_tool_context):
  received_context = {}

  def my_retriever(query: str, tool_context: ToolContext):
    received_context["ctx"] = tool_context
    return ["with context"]

  tool = CallableRetrieval(
      name="ctx_tool",
      description="Tool with context.",
      retriever=my_retriever,
  )
  result = await tool.run_async(
      args={"query": "test"}, tool_context=mock_tool_context
  )
  assert result == ["with context"]
  assert received_context["ctx"] is mock_tool_context


@pytest.mark.asyncio
async def test_tool_context_omission(mock_tool_context):
  def my_retriever(query: str):
    return ["no context needed"]

  tool = CallableRetrieval(
      name="no_ctx_tool",
      description="Tool without context.",
      retriever=my_retriever,
  )
  result = await tool.run_async(
      args={"query": "test"}, tool_context=mock_tool_context
  )
  assert result == ["no context needed"]


@pytest.mark.asyncio
async def test_async_callable_with_tool_context(mock_tool_context):
  async def my_retriever(query: str, tool_context: ToolContext):
    return [f"async {query} with context"]

  tool = CallableRetrieval(
      name="async_ctx_tool",
      description="Async tool with context.",
      retriever=my_retriever,
  )
  result = await tool.run_async(
      args={"query": "test"}, tool_context=mock_tool_context
  )
  assert result == ["async test with context"]


@pytest.mark.asyncio
async def test_sync_callable_object(mock_tool_context):

  class MyRetriever:

    def __call__(self, query: str):
      return [f"object result for {query}"]

  tool = CallableRetrieval(
      name="obj_tool",
      description="Callable object tool.",
      retriever=MyRetriever(),
  )
  result = await tool.run_async(
      args={"query": "hello"}, tool_context=mock_tool_context
  )
  assert result == ["object result for hello"]


@pytest.mark.asyncio
async def test_async_callable_object(mock_tool_context):

  class MyAsyncRetriever:

    async def __call__(self, query: str):
      return [f"async object result for {query}"]

  tool = CallableRetrieval(
      name="async_obj_tool",
      description="Async callable object tool.",
      retriever=MyAsyncRetriever(),
  )
  result = await tool.run_async(
      args={"query": "world"}, tool_context=mock_tool_context
  )
  assert result == ["async object result for world"]
