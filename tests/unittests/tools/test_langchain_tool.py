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

from unittest.mock import MagicMock

from google.adk.tools.langchain_tool import LangchainTool
from langchain.tools import tool
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
