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

import asyncio
import os
import time

import agent
from dotenv import load_dotenv
from google.adk.agents.run_config import RunConfig
from google.adk.runners import InMemoryRunner
from google.adk.sessions.session import Session
from google.genai import types
from opentelemetry import trace
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
from opentelemetry.sdk.trace import export
from opentelemetry.sdk.trace import TracerProvider

load_dotenv(override=True)


async def main():
  app_name = 'my_app'
  user_id_1 = 'user1'
  runner = InMemoryRunner(
      agent=agent.root_agent,
      app_name=app_name,
  )
  session_11 = await runner.session_service.create_session(
      app_name=app_name, user_id=user_id_1
  )

  async def run_prompt(session: Session, new_message: str):
    content = types.Content(
        role='user', parts=[types.Part.from_text(text=new_message)]
    )
    print('** User says:', content.model_dump(exclude_none=True))
    # TODO - migrate try...finally to contextlib.aclosing after Python 3.9 is
    # no longer supported.
    agen = runner.run_async(
        user_id=user_id_1,
        session_id=session.id,
        new_message=content,
    )
    try:
      async for event in agen:
        if event.content.parts and event.content.parts[0].text:
          print(f'** {event.author}: {event.content.parts[0].text}')
    finally:
      await agen.aclose()

  async def run_prompt_bytes(session: Session, new_message: str):
    content = types.Content(
        role='user',
        parts=[
            types.Part.from_bytes(
                data=str.encode(new_message), mime_type='text/plain'
            )
        ],
    )
    print('** User says:', content.model_dump(exclude_none=True))
    # TODO - migrate try...finally to contextlib.aclosing after Python 3.9 is
    # no longer supported.
    agen = runner.run_async(
        user_id=user_id_1,
        session_id=session.id,
        new_message=content,
        run_config=RunConfig(save_input_blobs_as_artifacts=True),
    )
    try:
      async for event in agen:
        if event.content.parts and event.content.parts[0].text:
          print(f'** {event.author}: {event.content.parts[0].text}')
    finally:
      await agen.aclose()

  start_time = time.time()
  print('Start time:', start_time)
  print('------------------------------------')
  await run_prompt(session_11, 'Hi')
  await run_prompt(session_11, 'Roll a die with 100 sides')
  await run_prompt(session_11, 'Roll a die again with 100 sides.')
  await run_prompt(session_11, 'What numbers did I got?')
  await run_prompt_bytes(session_11, 'Hi bytes')
  print(
      await runner.artifact_service.list_artifact_keys(
          app_name=app_name, user_id=user_id_1, session_id=session_11.id
      )
  )
  end_time = time.time()
  print('------------------------------------')
  print('End time:', end_time)
  print('Total time:', end_time - start_time)


if __name__ == '__main__':

  provider = TracerProvider()
  project_id = os.environ.get('GOOGLE_CLOUD_PROJECT')
  if not project_id:
    raise ValueError('GOOGLE_CLOUD_PROJECT environment variable is not set.')
  print('Tracing to project', project_id)
  processor = export.BatchSpanProcessor(
      CloudTraceSpanExporter(project_id=project_id)
  )
  provider.add_span_processor(processor)
  trace.set_tracer_provider(provider)

  asyncio.run(main())

  provider.force_flush()
  print('Done tracing to project', project_id)
