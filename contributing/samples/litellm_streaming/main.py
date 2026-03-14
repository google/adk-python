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

"""Runs the LiteLLM streaming sample with SSE enabled."""

from __future__ import annotations

import asyncio

from dotenv import load_dotenv
from google.adk.agents.run_config import RunConfig
from google.adk.agents.run_config import StreamingMode
from google.adk.cli.utils import logs
from google.adk.runners import InMemoryRunner
from google.genai import types

from . import agent

load_dotenv(override=True)
logs.log_to_tmp_folder()


async def _run_prompt(
    *,
    runner: InMemoryRunner,
    user_id: str,
    session_id: str,
    prompt: str,
) -> None:
  """Runs one prompt and prints partial chunks in real time."""
  content = types.Content(
      role='user',
      parts=[types.Part.from_text(text=prompt)],
  )

  print(f'User: {prompt}')
  print('Agent: ', end='', flush=True)
  saw_text = False
  saw_partial_text = False

  # For `adk web`, enable the `Streaming` toggle in the UI to get
  # partial SSE responses similar to this script.
  async for event in runner.run_async(
      user_id=user_id,
      session_id=session_id,
      new_message=content,
      run_config=RunConfig(streaming_mode=StreamingMode.SSE),
  ):
    if not event.content:
      continue
    text = ''.join(part.text for part in event.content.parts if part.text)
    if not text:
      continue

    if event.partial:
      print(text, end='', flush=True)
      saw_text = True
      saw_partial_text = True
      continue

    # With SSE mode, ADK emits a final aggregated event after partial chunks.
    if not saw_partial_text:
      print(text, end='', flush=True)
      saw_text = True

  if saw_text:
    print()
  else:
    print('(no text response)')
  print('------------------------------------')


async def main() -> None:
  app_name = 'litellm_streaming_demo'
  user_id = 'user_1'
  runner = InMemoryRunner(agent=agent.root_agent, app_name=app_name)
  session = await runner.session_service.create_session(
      app_name=app_name,
      user_id=user_id,
  )
  prompts = [
      'Write an essay about the roman empire',
      'Now summarize the essay into one sentence.',
  ]

  for prompt in prompts:
    await _run_prompt(
        runner=runner,
        user_id=user_id,
        session_id=session.id,
        prompt=prompt,
    )


if __name__ == '__main__':
  asyncio.run(main())
