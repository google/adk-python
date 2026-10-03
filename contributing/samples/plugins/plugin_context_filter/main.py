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


"""Shows how ContextFilterPlugin trims what each turn sends to the model."""

import asyncio

from google.adk import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.plugins.context_filter_plugin import ContextFilterPlugin
from google.adk.runners import InMemoryRunner
from google.adk.tools.tool_context import ToolContext
from google.genai import types

APP_NAME = 'plugin_context_filter'

# Each of these starts a new invocation, so the third one is where the plugin
# begins dropping the oldest turn.
PROMPTS = [
    'Look up the delivery status for order A1.',
    'Now look up order B2.',
    'Now look up order C3.',
    'Which orders have I asked about so far?',
]


async def lookup_order(tool_context: ToolContext, order_id: str) -> dict:
  """Returns a canned delivery status for an order."""
  return {'order_id': order_id, 'status': 'in transit'}


class ContentCounterPlugin(BasePlugin):
  """Prints how many contents each request carries.

  Registered after ContextFilterPlugin, so it sees the trimmed request:
  plugin callbacks run in registration order.
  """

  def __init__(self) -> None:
    super().__init__(name='content_counter')
    self.turn = 0

  async def before_model_callback(
      self, *, callback_context: CallbackContext, llm_request: LlmRequest
  ) -> None:
    self.turn += 1
    contents = llm_request.contents or []
    print(f'[request {self.turn}] contents sent to the model: {len(contents)}')


root_agent = Agent(
    model='gemini-2.5-flash',
    name='order_agent',
    description='Looks up delivery status for orders.',
    instruction=(
        'Use the lookup_order tool to answer questions about an order. Answer'
        ' questions about earlier orders only from the conversation you can'
        ' still see.'
    ),
    tools=[lookup_order],
)


async def main():
  runner = InMemoryRunner(
      agent=root_agent,
      app_name=APP_NAME,
      # Keep the two most recent invocations. Without this plugin the whole
      # conversation grows on every turn, and so does the cost of each call.
      plugins=[
          ContextFilterPlugin(num_invocations_to_keep=2),
          ContentCounterPlugin(),
      ],
  )
  session = await runner.session_service.create_session(
      user_id='user', app_name=APP_NAME
  )

  for prompt in PROMPTS:
    print(f'\nuser: {prompt}')
    async for event in runner.run_async(
        user_id='user',
        session_id=session.id,
        new_message=types.Content(
            role='user', parts=[types.Part.from_text(text=prompt)]
        ),
    ):
      if event.content and event.content.parts:
        for part in event.content.parts:
          if part.text and event.author != 'user':
            print(f'agent: {part.text.strip()}')

  stored = await runner.session_service.get_session(
      app_name=APP_NAME, user_id='user', session_id=session.id
  )
  print(f'\nevents kept in the session: {len(stored.events)}')
  print('The session keeps everything; only the model request is trimmed.')


if __name__ == '__main__':
  asyncio.run(main())
