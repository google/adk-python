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

from google.adk.events.event_actions import EventActions
from google.adk.integrations.langchain import LangchainTool
from langchain_core.tools import tool
from langchain_core.tools.structured import StructuredTool
from pydantic import BaseModel
import pytest


@tool
async def async_add_with_annotation(x, y) -> int:
  """Adds two numbers"""
  return x + y


@tool
def sync_add_with_annotation(x, y) -> int:
  """Adds two numbers"""
  return x + y


@tool(return_direct=True)
def direct_add(x, y) -> int:
  """Adds two numbers"""
  return x + y


@tool(return_direct=True)
def direct_payload_with_error_key(x) -> dict:
  """Returns a payload that carries a falsy error key"""
  return {"error": None, "value": x}


async def async_add(x, y) -> int:
  return x + y


def sync_add(x, y) -> int:
  return x + y


class AddSchema(BaseModel):
  x: int
  y: int


test_langchain_async_add_tool = StructuredTool.from_function(
    async_add,
    name="add",
    description="Adds two numbers",
    args_schema=AddSchema,
)

test_langchain_sync_add_tool = StructuredTool.from_function(
    sync_add,
    name="add",
    description="Adds two numbers",
    args_schema=AddSchema,
)


@pytest.mark.asyncio
async def test_raw_async_function_works():
  """Test that passing a raw async function to LangchainTool works correctly."""
  langchain_tool = LangchainTool(tool=test_langchain_async_add_tool)
  result = await langchain_tool.run_async(
      args={"x": 1, "y": 3}, tool_context=MagicMock()
  )
  assert result == 4


@pytest.mark.asyncio
async def test_raw_sync_function_works():
  """Test that passing a raw sync function to LangchainTool works correctly."""
  langchain_tool = LangchainTool(tool=test_langchain_sync_add_tool)
  result = await langchain_tool.run_async(
      args={"x": 1, "y": 3}, tool_context=MagicMock()
  )
  assert result == 4


@pytest.mark.asyncio
async def test_raw_async_function_with_annotation_works():
  """Test that passing a raw async function to LangchainTool works correctly."""
  langchain_tool = LangchainTool(tool=async_add_with_annotation)
  result = await langchain_tool.run_async(
      args={"x": 1, "y": 3}, tool_context=MagicMock()
  )
  assert result == 4


@pytest.mark.asyncio
async def test_raw_sync_function_with_annotation_works():
  """Test that passing a raw sync function to LangchainTool works correctly."""
  langchain_tool = LangchainTool(tool=sync_add_with_annotation)
  result = await langchain_tool.run_async(
      args={"x": 1, "y": 3}, tool_context=MagicMock()
  )
  assert result == 4


@pytest.mark.asyncio
async def test_return_direct_sets_skip_summarization():
  """A tool with return_direct=True skips summarization on run."""
  langchain_tool = LangchainTool(tool=direct_add)
  assert langchain_tool._return_direct is True

  tool_context = MagicMock()
  tool_context.actions = EventActions()
  result = await langchain_tool.run_async(
      args={"x": 1, "y": 2}, tool_context=tool_context
  )

  assert result == 3
  assert tool_context.actions.skip_summarization is True


@pytest.mark.asyncio
async def test_return_direct_leaves_skip_summarization_on_error():
  """A missing-argument error stays summarizable so the model can retry."""
  langchain_tool = LangchainTool(tool=direct_add)

  tool_context = MagicMock()
  tool_context.actions = EventActions()
  result = await langchain_tool.run_async(
      args={"x": 1}, tool_context=tool_context
  )

  assert "error" in result
  assert tool_context.actions.skip_summarization is None


@pytest.mark.asyncio
async def test_return_direct_skips_summarization_for_falsy_error_key():
  """A payload whose error key is falsy is a real result, not an error."""
  langchain_tool = LangchainTool(tool=direct_payload_with_error_key)

  tool_context = MagicMock()
  tool_context.actions = EventActions()
  result = await langchain_tool.run_async(
      args={"x": 1}, tool_context=tool_context
  )

  assert result == {"error": None, "value": 1}
  assert tool_context.actions.skip_summarization is True


@pytest.mark.asyncio
async def test_return_direct_default_false_leaves_skip_summarization():
  """A tool without return_direct does not touch skip_summarization."""
  langchain_tool = LangchainTool(tool=test_langchain_sync_add_tool)
  assert langchain_tool._return_direct is False

  tool_context = MagicMock()
  tool_context.actions = EventActions()
  result = await langchain_tool.run_async(
      args={"x": 1, "y": 3}, tool_context=tool_context
  )

  assert result == 4
  assert tool_context.actions.skip_summarization is None
