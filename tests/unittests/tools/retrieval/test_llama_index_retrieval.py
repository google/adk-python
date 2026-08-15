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

"""Tests for LlamaIndexRetrieval tool."""

from dataclasses import dataclass
from typing import Optional

from google.adk.tools.retrieval.llama_index_retrieval import LlamaIndexRetrieval
import pytest


@dataclass
class _FakeNode:
  """Stands in for a llama-index node, which exposes its content as `text`."""

  text: str


class _FakeRetriever:
  """Records the query it was asked for and replays canned nodes."""

  def __init__(self, nodes: list[_FakeNode]):
    self._nodes = nodes
    self.received_query: Optional[str] = None

  def retrieve(self, query):
    self.received_query = query
    return self._nodes


def _tool(retriever: _FakeRetriever) -> LlamaIndexRetrieval:
  return LlamaIndexRetrieval(
      name='docs',
      description='Retrieves documentation.',
      retriever=retriever,
  )


@pytest.mark.asyncio
async def test_run_async_returns_the_text_of_the_top_result():
  """Only the best-ranked node is returned, not the whole ranked list."""
  retriever = _FakeRetriever(
      [_FakeNode('best match'), _FakeNode('worse match')]
  )

  result = await _tool(retriever).run_async(
      args={'query': 'anything'}, tool_context=None
  )

  assert result == 'best match'


@pytest.mark.asyncio
async def test_run_async_passes_the_query_argument_to_the_retriever():
  """The retriever gets the query string itself, not the whole args dict."""
  retriever = _FakeRetriever([_FakeNode('a document')])

  await _tool(retriever).run_async(
      args={'query': 'how do i retrieve', 'unused': 1}, tool_context=None
  )

  assert retriever.received_query == 'how do i retrieve'


def test_name_and_description_are_forwarded_to_the_declaration():
  """The retrieval declaration is what the model sees, so it must carry both."""
  tool = _tool(_FakeRetriever([]))

  declaration = tool._get_declaration()

  assert declaration.name == 'docs'
  assert declaration.description == 'Retrieves documentation.'
