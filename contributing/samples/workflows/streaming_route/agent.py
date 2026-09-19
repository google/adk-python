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

"""Mid-stream preemptive routing + libuv (uvloop).

The classifier only needs to emit a single category word. A plain routing
node would wait for the model to finish the turn before advancing the graph.
``StreamingRouterNode`` instead watches the streamed tokens and, the instant
the category word appears, commits the route and cancels the rest of the
generation — the graph advances mid-stream.

``enable_uvloop()`` swaps the process onto a libuv event loop for a faster
asyncio runtime. It is a no-op (with a log line) when uvloop is not installed;
install it with ``pip install "google-adk[uvloop]"``.
"""

from google.adk import Agent
from google.adk import enable_uvloop
from google.adk import Event
from google.adk import Workflow
from google.adk.workflow import StreamDecision
from google.adk.workflow import StreamingRouterNode
from google.adk.workflow import StreamView

# Put the whole process on libuv. Call once, before anything runs.
enable_uvloop()

CATEGORIES = ('billing', 'technical', 'sales')


def process_input(node_input: str):
  """Stashes the raw user message in state for downstream agents."""
  yield Event(state={'input': node_input})


classifier = Agent(
    name='classifier',
    instruction=(
        "Classify the user's request into exactly one category. Reply with"
        ' ONLY that single lowercase word and nothing else: billing,'
        ' technical, or sales.\n\nRequest: {input}'
    ),
)


def route_when_category_streams(view: StreamView) -> StreamDecision | None:
  """Advances the graph as soon as a category word appears in the stream.

  Returning a ``StreamDecision`` commits the route and (by default) cancels
  the remainder of the model call. Returning ``None`` means "keep streaming".
  """
  text = view.text.lower()
  for category in CATEGORIES:
    if category in text:
      return StreamDecision(route=category)
  return None


intent_router = StreamingRouterNode(
    name='intent_router',
    agent=classifier,
    monitor=route_when_category_streams,
)


billing_agent = Agent(
    name='billing_agent',
    instruction='Help the user with their billing issue: {input}',
)
technical_agent = Agent(
    name='technical_agent',
    instruction='Help the user with their technical issue: {input}',
)
sales_agent = Agent(
    name='sales_agent',
    instruction='Help the user with their sales question: {input}',
)


root_agent = Workflow(
    name='root_agent',
    edges=[
        ('START', process_input, intent_router),
        (
            intent_router,
            {
                'billing': billing_agent,
                'technical': technical_agent,
                'sales': sales_agent,
            },
        ),
    ],
)
