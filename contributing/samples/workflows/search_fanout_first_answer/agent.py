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

"""Search fan-out that answers from the first source that has it (Recall@k).

The classic enterprise-search shape: a keyword search returns the top-k candidate
sources (SharePoint, Confluence, CRM, a drive, email...). You don't know which
one holds the answer, so you read them **in parallel** -- and you want to answer
the moment *any* one of them does, without waiting on (or paying to finish) the
rest.

Two mechanisms combine here:

  1. Per-branch preemption -- each source is read by a ``StreamingRouterNode``
     whose ``monitor`` stops that read as soon as the model can say "answer is
     here" or "not here", so a single irrelevant source never streams to the end.
  2. Cross-branch first-answer-wins -- ``FirstMatchNode`` races the branches and,
     the instant one returns ``found=True``, cancels the still-running siblings
     (tearing down their in-flight model calls) and returns that answer.

``enable_uvloop()`` puts the concurrent reads on a libuv loop.

Honest limit: branches that already started still paid to *read* their source
(prefill). This saves the losers' *generation* and the wall-clock of waiting on
them. To also avoid reading low-ranked sources, pass them in rank order and set
``max_parallel`` so a win short-circuits before the tail is ever started.
"""

from google.adk import Agent
from google.adk import enable_uvloop
from google.adk import Event
from google.adk import Workflow
from google.adk.workflow import FirstMatchNode
from google.adk.workflow import StreamDecision
from google.adk.workflow import StreamingRouterNode
from google.adk.workflow import StreamView

enable_uvloop()

# Top-k candidates as if returned by a keyword search, in rank order.
SOURCES = ('sharepoint', 'confluence', 'crm', 'gdrive', 'email')

_FOUND = 'FOUND:'
_NOT_HERE = 'NOTHERE'


def stash_query(node_input: str):
  """Puts the user query in state so every reader can template it in."""
  yield Event(state={'query': node_input})


def _make_reader(source: str) -> StreamingRouterNode:
  reader = Agent(
      name=f'read_{source}',
      instruction=(
          f'You are searching the "{source}" source to answer: {{query}}.\n'
          'Read only as far as you must. As soon as you can tell, output ONE'
          ' line and nothing after it:\n'
          f'  "{_FOUND} <the answer>"   if this source answers the question,'
          ' or\n'
          f'  "{_NOT_HERE}"             if it clearly does not.'
      ),
      output_key=f'{source}_answer',
  )

  def monitor(view: StreamView) -> StreamDecision | None:
    upper = view.text.upper()
    idx = upper.find(_FOUND)
    if idx != -1:
      line, sep, _ = view.text[idx + len(_FOUND) :].partition('\n')
      if sep or line.strip():
        # Answer found: commit it and preempt this branch's generation.
        return StreamDecision(
            output={'found': True, 'source': source, 'answer': line.strip()}
        )
    if _NOT_HERE in upper:
      # Source is irrelevant: stop reading this one early.
      return StreamDecision(output={'found': False, 'source': source})
    return None

  return StreamingRouterNode(
      name=f'reader_{source}',
      agent=reader,
      monitor=monitor,
      # Hard cap so a stuck/slow source can never hold up the race.
      timeout=60,
  )


# The complement of JoinNode: race the readers, return the first that answers,
# and cancel the losers mid-read.
first_answer = FirstMatchNode(
    name='first_answer',
    nodes=[_make_reader(source) for source in SOURCES],
    match=lambda r: isinstance(r, dict) and r.get('found'),
    no_match_output={
        'found': False,
        'answer': 'Not found in any of the top sources.',
    },
    # Read all k at once. For rank-ordered prefill savings, set e.g.
    # max_parallel=2 so a win short-circuits before lower ranks are read.
)


async def respond(node_input: dict):
  if node_input and node_input.get('found'):
    yield Event(
        message=(
            f"Answer (from {node_input['source']}): {node_input['answer']}"
        )
    )
  else:
    yield Event(message='No source contained the answer.')


root_agent = Workflow(
    name='root_agent',
    edges=[('START', stash_query, first_answer, respond)],
)
