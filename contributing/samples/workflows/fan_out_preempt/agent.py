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

"""Parallel fan-out with per-branch mid-stream preemption + libuv.

Reads five data sources concurrently. Each source is analyzed by its own
streaming agent. The instant an agent's stream reveals the source is
irrelevant to the query, that branch is preempted -- the rest of its
generation is cancelled -- while the other branches keep running. A JoinNode
fans the results back in and a synthesizer drops the irrelevant ones.

Each branch runs as its own asyncio task, so the reads truly overlap;
``enable_uvloop()`` puts them on a faster libuv loop.
"""

from typing import Any

from google.adk import Agent
from google.adk import enable_uvloop
from google.adk import Event
from google.adk import Workflow
from google.adk.workflow import JoinNode
from google.adk.workflow import StreamDecision
from google.adk.workflow import StreamingRouterNode
from google.adk.workflow import StreamView

enable_uvloop()

SOURCES = ('sharepoint', 'havian', 'wiki', 'crm', 'docs')

# Sentinel the reader emits as its first line when a source does not apply.
IRRELEVANT = 'IRRELEVANT'


def stash_query(node_input: str):
  """Puts the user query in state so every reader can template it in."""
  yield Event(state={'query': node_input})


def _make_reader(source: str) -> StreamingRouterNode:
  reader = Agent(
      name=f'read_{source}',
      instruction=(
          f'You are reading the "{source}" source to answer: {{query}}.\n'
          'If this source is clearly irrelevant to the query, your FIRST'
          f' line must be exactly "{IRRELEVANT}". Otherwise, extract only the'
          ' facts from this source that help answer the query.'
      ),
      output_key=f'{source}_result',
  )

  def monitor(view: StreamView) -> StreamDecision | None:
    # As soon as the model declares irrelevance, stop reading this source.
    if view.text.lstrip().upper().startswith(IRRELEVANT):
      return StreamDecision(output={'source': source, 'relevant': False})
    # Otherwise keep streaming; a relevant read finishes normally and its
    # final text becomes this branch's output.
    return None

  return StreamingRouterNode(
      name=f'reader_{source}',
      agent=reader,
      monitor=monitor,
      # Bound the deep read so a stuck/slow source can never hold up the join.
      timeout=60,
  )


readers = tuple(_make_reader(source) for source in SOURCES)
join_sources = JoinNode(name='join_sources')


async def synthesize(node_input: dict[str, Any]):
  """Fan-in: drop the branches that preempted as irrelevant, then answer."""
  relevant = {
      name: result
      for name, result in node_input.items()
      if not (isinstance(result, dict) and result.get('relevant') is False)
  }
  skipped = sorted(set(node_input) - set(relevant))
  yield Event(
      message=(
          f'Answer synthesized from {sorted(relevant)}.\n'
          f'Skipped (irrelevant, preempted mid-read): {skipped}.'
      ),
  )


root_agent = Workflow(
    name='root_agent',
    edges=[('START', stash_query, readers, join_sources, synthesize)],
)
