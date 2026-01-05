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

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional
from typing import Union

import click
from google.genai import types
from pydantic import BaseModel

from ..agents.llm_agent import LlmAgent
from ..apps.app import App
from ..artifacts.base_artifact_service import BaseArtifactService
from ..auth.credential_service.base_credential_service import BaseCredentialService
from ..auth.credential_service.in_memory_credential_service import InMemoryCredentialService
from ..runners import Runner
from ..sessions.base_session_service import BaseSessionService
from ..sessions.session import Session
from ..utils.context_utils import Aclosing
from ..utils.env_utils import is_env_enabled
from .service_registry import load_services_module
from .utils import envs
from .utils.agent_loader import AgentLoader
from .utils.service_factory import create_artifact_service_from_options
from .utils.service_factory import create_session_service_from_options


class InputFile(BaseModel):
  state: dict[str, object]
  queries: list[str]


async def run_input_file(
    app_name: str,
    user_id: str,
    agent_or_app: Union[LlmAgent, App],
    artifact_service: BaseArtifactService,
    session_service: BaseSessionService,
    credential_service: BaseCredentialService,
    input_path: str,
) -> Session:
  app = (
      agent_or_app
      if isinstance(agent_or_app, App)
      else App(name=app_name, root_agent=agent_or_app)
  )
  runner = Runner(
      app=app,
      artifact_service=artifact_service,
      session_service=session_service,
      credential_service=credential_service,
  )
  with open(input_path, 'r', encoding='utf-8') as f:
    input_file = InputFile.model_validate_json(f.read())
  input_file.state['_time'] = datetime.now().isoformat()

  session = await session_service.create_session(
      app_name=app_name, user_id=user_id, state=input_file.state
  )
  for query in input_file.queries:
    click.echo(f'[user]: {query}')
    content = types.Content(role='user', parts=[types.Part(text=query)])
    async with Aclosing(
        runner.run_async(
            user_id=session.user_id, session_id=session.id, new_message=content
        )
    ) as agen:
      async for event in agen:
        if event.content and event.content.parts:
          if text := ''.join(part.text or '' for part in event.content.parts):
            click.echo(f'[{event.author}]: {text}')
  return session


async def run_interactively(
    root_agent_or_app: Union[LlmAgent, App],
    artifact_service: BaseArtifactService,
    session: Session,
    session_service: BaseSessionService,
    credential_service: BaseCredentialService,
) -> None:
  app = (
      root_agent_or_app
      if isinstance(root_agent_or_app, App)
      else App(name=session.app_name, root_agent=root_agent_or_app)
  )
  runner = Runner(
      app=app,
      artifact_service=artifact_service,
      session_service=session_service,
      credential_service=credential_service,
  )
  while True:
    query = input('[user]: ')
    if not query or not query.strip():
      continue
    if query == 'exit':
      break
    async with Aclosing(
        runner.run_async(
            user_id=session.user_id,
            session_id=session.id,
            new_message=types.Content(
                role='user', parts=[types.Part(text=query)]
            ),
        )
    ) as agen:
      async for event in agen:
        if event.content and event.content.parts:
          if text := ''.join(part.text or '' for part in event.content.parts):
            click.echo(f'[{event.author}]: {text}')
  await runner.close()


async def run_cli(
    *,
    agent_parent_dir: str,
    agent_folder_name: str,
    input_file: Optional[str] = None,
    saved_session_file: Optional[str] = None,
    save_session: bool,
    session_id: Optional[str] = None,
    session_service_uri: Optional[str] = None,
    artifact_service_uri: Optional[str] = None,
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
    session_id: Optional[str], the session ID to save the session to on exit.
    session_service_uri: Optional[str], custom session service URI.
    artifact_service_uri: Optional[str], custom artifact service URI.
  """
  agent_parent_path = Path(agent_parent_dir).resolve()
  agent_root = agent_parent_path / agent_folder_name
  load_services_module(str(agent_root))
  user_id = 'test_user'

  agents_dir = str(agent_parent_path)
  agent_loader = AgentLoader(agents_dir=agents_dir)
  agent_or_app = agent_loader.load_agent(agent_folder_name)
  session_app_name = (
      agent_or_app.name if isinstance(agent_or_app, App) else agent_folder_name
  )
  app_name_to_dir = None
  if isinstance(agent_or_app, App) and agent_or_app.name != agent_folder_name:
    app_name_to_dir = {agent_or_app.name: agent_folder_name}

  # Create session and artifact services using factory functions
  # Sessions persist under <agents_dir>/<agent>/.adk/session.db by default.
  session_service = create_session_service_from_options(
      base_dir=agent_parent_path,
      session_service_uri=session_service_uri,
      app_name_to_dir=app_name_to_dir,
  )

  artifact_service = create_artifact_service_from_options(
      base_dir=agent_root,
      artifact_service_uri=artifact_service_uri,
  )

  credential_service = InMemoryCredentialService()
  if not is_env_enabled('ADK_DISABLE_LOAD_DOTENV'):
    envs.load_dotenv_for_agent(agent_folder_name, agents_dir)

  # Helper function for printing events
  def _print_event(event) -> None:
    content = event.content
    if not content or not content.parts:
      return
    text_parts = [part.text for part in content.parts if part.text]
    if not text_parts:
      return
    author = event.author or 'system'
    click.echo(f'[{author}]: {"".join(text_parts)}')

  if input_file:
    session = await run_input_file(
        app_name=session_app_name,
        user_id=user_id,
        agent_or_app=agent_or_app,
        artifact_service=artifact_service,
        session_service=session_service,
        credential_service=credential_service,
        input_path=input_file,
    )
  elif saved_session_file:
    # Load the saved session from file
    with open(saved_session_file, 'r', encoding='utf-8') as f:
      loaded_session = Session.model_validate_json(f.read())

    # Create a new session in the service, copying state from the file
    session = await session_service.create_session(
        app_name=session_app_name,
        user_id=user_id,
        state=loaded_session.state if loaded_session else None,
    )

    # Append events from the file to the new session and display them
    if loaded_session:
      for event in loaded_session.events:
        await session_service.append_event(session, event)
        _print_event(event)

    await run_interactively(
        agent_or_app,
        artifact_service,
        session,
        session_service,
        credential_service,
    )
  else:
    session = await session_service.create_session(
        app_name=session_app_name, user_id=user_id
    )
    click.echo(f'Running agent {agent_or_app.name}, type exit to exit.')
    await run_interactively(
        agent_or_app,
        artifact_service,
        session,
        session_service,
        credential_service,
    )

  if save_session:
    session_id = session_id or input('Session ID to save: ')
    session_path = agent_root / f'{session_id}.session.json'

    # Fetch the session again to get all the details.
    session = await session_service.get_session(
        app_name=session.app_name,
        user_id=session.user_id,
        session_id=session.id,
    )
    session_path.write_text(
        session.model_dump_json(indent=2, exclude_none=True, by_alias=True),
        encoding='utf-8',
    )

    print('Session saved to', session_path)
