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

"""Drives a scripted live conversation and swaps the agent's context once.

Run from the repository root:

    python3 -m contributing.samples.live.context_update.main

The application owns the `LiveRequestQueue`, so the application decides when
the conversation changes phase. Here the trigger is the `identify_customer`
tool result: when it arrives, the script sends one context update that
replaces the instruction and the tools. Per-turn policy hints go to the tail
of the conversation instead, which leaves the model's cached prefix intact.
"""

from __future__ import annotations

import asyncio

from google.adk.agents.run_config import RunConfig
from google.adk.events.event import Event
from google.adk.live import LiveRequestQueue
from google.adk.runners import InMemoryRunner
from google.genai import types

from . import agent

APP_NAME = 'context_update'
USER_ID = 'user'

# A scripted caller. Each turn is sent only after the previous one is answered.
TURNS = [
    'Hi, my number is 555-0100.',
    'Where is order 42?',
    'It arrived broken. Please refund it.',
]

# Turn-specific rules keyed by a word in the caller's turn. A real app would
# retrieve these from a policy store; the point is where they are sent, not
# how they are chosen.
TURN_POLICIES = {
    'refund': (
        '[POLICY] Before refunding, read the order total back to the caller'
        ' and ask them to confirm.'
    ),
}

# How long to wait for more events after the model signals turn completion
# before sending the next scripted turn.
_QUIET_PERIOD_SECONDS = 2.0


def _policy_for(turn: str) -> str | None:
  for keyword, policy in TURN_POLICIES.items():
    if keyword in turn.lower():
      return policy
  return None


def _text_of(event: Event) -> str:
  if not event.content or not event.content.parts:
    return ''
  return ''.join(part.text or '' for part in event.content.parts)


async def main() -> None:
  runner = InMemoryRunner(agent=agent.root_agent, app_name=APP_NAME)
  session = await runner.session_service.create_session(
      app_name=APP_NAME, user_id=USER_ID
  )
  live_request_queue = LiveRequestQueue()
  run_config = RunConfig(response_modalities=[types.Modality.TEXT])

  events: asyncio.Queue[Event | None] = asyncio.Queue()

  async def pump_events() -> None:
    try:
      async for event in runner.run_live(
          user_id=USER_ID,
          session_id=session.id,
          live_request_queue=live_request_queue,
          run_config=run_config,
      ):
        await events.put(event)
    finally:
      await events.put(None)

  pump = asyncio.create_task(pump_events())
  context_updated = False

  try:
    for turn in TURNS:
      # Cache-friendly per-turn context: appended to the conversation as a
      # partial turn, so the instruction and tools, and the prefix cache
      # built on them, do not change.
      if policy := _policy_for(turn):
        print(f'[app]   tail-inject: {policy}')
        live_request_queue.send_content(
            types.Content(role='user', parts=[types.Part(text=policy)]),
            partial=True,
        )

      print(f'[user]  {turn}')
      live_request_queue.send_content(
          types.Content(role='user', parts=[types.Part(text=turn)])
      )

      model_text = ''
      turn_completed = False
      while True:
        timeout = _QUIET_PERIOD_SECONDS if turn_completed else None
        try:
          event = await asyncio.wait_for(events.get(), timeout=timeout)
        except asyncio.TimeoutError:
          break
        if event is None:
          return

        for call in event.get_function_calls():
          print(f'[tool]  {call.name}({call.args})')
        for response in event.get_function_responses():
          print(f'[tool]  {response.name} -> {response.response}')
          # Phase transition: the caller is identified, so the rest of the
          # conversation needs the order-handling persona and tools. One
          # update, one prefix-cache rebuild.
          if response.name == 'identify_customer' and not context_updated:
            context_updated = True
            instruction = agent.ORDER_INSTRUCTION.format(
                customer=response.response
            )
            print('[app]   send_context_update: order-handling phase')
            live_request_queue.send_context_update(
                system_instruction=instruction, tools=agent.ORDER_TOOLS
            )

        if event.partial:
          model_text += _text_of(event)
        elif not event.get_function_calls() and _text_of(event):
          model_text = _text_of(event)
        if event.turn_complete:
          turn_completed = True

      if model_text:
        print(f'[model] {model_text.strip()}')
  finally:
    live_request_queue.close()
    await pump


if __name__ == '__main__':
  asyncio.run(main())
