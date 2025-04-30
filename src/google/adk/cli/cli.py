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

from datetime import datetime
import importlib
import os
import sys
from typing import Optional

import click
from google.genai import types
from pydantic import BaseModel

from ..agents.llm_agent import LlmAgent
from ..artifacts import BaseArtifactService
from ..artifacts import InMemoryArtifactService
from ..runners import Runner
from ..sessions.base_session_service import BaseSessionService
from ..sessions.in_memory_session_service import InMemorySessionService
from ..sessions.session import Session
from .utils import envs


class InputFile(BaseModel):
  state: dict[str, object]
  queries: list[str]


async def run_input_file(
    app_name: str,
    user_id: str,
    root_agent: LlmAgent,
    artifact_service: BaseArtifactService,
    session_service: BaseSessionService,
    input_path: str,
) -> Session:
  runner = Runner(
      app_name=app_name,
      agent=root_agent,
      artifact_service=artifact_service,
      session_service=session_service,
  )
  with open(input_path, 'r', encoding='utf-8') as f:
    input_file = InputFile.model_validate_json(f.read())
  input_file.state['_time'] = datetime.now()

  session = session_service.create_session(
      app_name=app_name, user_id=user_id, state=input_file.state
  )
  for query in input_file.queries:
    click.echo(f'[user]: {query}')
    content = types.Content(role='user', parts=[types.Part(text=query)])
    async for event in runner.run_async(
        user_id=session.user_id, session_id=session.id, new_message=content
    ):
      if event.content and event.content.parts:
        if text := ''.join(part.text or '' for part in event.content.parts):
          click.echo(f'[{event.author}]: {text}')
  return session


async def run_interactively(
    root_agent: LlmAgent,
    artifact_service: BaseArtifactService,
    session: Session,
    session_service: BaseSessionService,
) -> None:
  runner = Runner(
      app_name=session.app_name,
      agent=root_agent,
      artifact_service=artifact_service,
      session_service=session_service,
  )
  while True:
    query = input('[user]: ')
    if not query or not query.strip():
      continue
    if query == 'exit':
      break
    async for event in runner.run_async(
        user_id=session.user_id,
        session_id=session.id,
        new_message=types.Content(role='user', parts=[types.Part(text=query)]),
    ):
      if event.content and event.content.parts:
        if text := ''.join(part.text or '' for part in event.content.parts):
          click.echo(f'[{event.author}]: {text}')


async def run_cli(
    *,
    agent_parent_dir: str,
    agent_folder_name: str,
    input_file: Optional[str] = None,
    saved_session_file: Optional[str] = None,
    save_session: bool,
) -> None:
  """Runs an interactive CLI for a certain agent.

  Args:
    agent_parent_dir: str, the absolute path of the parent folder of the agent
      folder.
    agent_folder_name: str, the name of the agent folder.
    input_file: Optional[str], the absolute path to the json file that contains
      the initial session state and user queries, exclusive with
      saved_session_file.
    saved_session_file: Optional[str], the absolute path to the json file that
      contains a previously saved session, exclusive with input_file.
    save_session: bool, whether to save the session on exit.
  """
  if agent_parent_dir not in sys.path:
    sys.path.append(agent_parent_dir)

  artifact_service = InMemoryArtifactService()
  session_service = InMemorySessionService()

  agent_module_path = os.path.join(agent_parent_dir, agent_folder_name)
  agent_module = importlib.import_module(agent_folder_name)
  user_id = 'test_user'
  session = session_service.create_session(
      app_name=agent_folder_name, user_id=user_id
  )
  root_agent = agent_module.agent.root_agent
  envs.load_dotenv_for_agent(agent_folder_name, agent_parent_dir)
  if input_file:
    session = await run_input_file(
        app_name=agent_folder_name,
        user_id=user_id,
        root_agent=root_agent,
        artifact_service=artifact_service,
        session_service=session_service,
        input_path=input_file,
    )
  elif saved_session_file:

    loaded_session = None
    with open(saved_session_file, 'r') as f:
      loaded_session = Session.model_validate_json(f.read())

    if loaded_session:
      for event in loaded_session.events:
        session_service.append_event(session, event)
        content = event.content
        if not content or not content.parts or not content.parts[0].text:
          continue
        if event.author == 'user':
          click.echo(f'[user]: {content.parts[0].text}')
        else:
          click.echo(f'[{event.author}]: {content.parts[0].text}')

    await run_interactively(
        root_agent,
        artifact_service,
        session,
        session_service,
    )
  else:
    click.echo(f'Running agent {root_agent.name}, type exit to exit.')
    await run_interactively(
        root_agent,
        artifact_service,
        session,
        session_service,
    )

  if save_session:
    session_id = input('Session ID to save: ')
    session_path = f'{agent_module_path}/{session_id}.session.json'

    # Fetch the session again to get all the details.
    session = session_service.get_session(
        app_name=session.app_name,
        user_id=session.user_id,
        session_id=session.id,
    )
    with open(session_path, 'w') as f:
      f.write(session.model_dump_json(indent=2, exclude_none=True))

    print('Session saved to', session_path)
