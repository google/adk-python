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

from __future__ import annotations

import asyncio
from datetime import datetime
import json
import logging
from pathlib import Path
import re
import sys
from typing import Any
from typing import Optional
from typing import Union

import click
from google.genai import types
from pydantic import BaseModel

from ..agents.base_agent import BaseAgent
from ..agents.llm_agent import LlmAgent
from ..apps.app import App
from ..artifacts.base_artifact_service import BaseArtifactService
from ..auth.credential_service.base_credential_service import BaseCredentialService
from ..auth.credential_service.in_memory_credential_service import InMemoryCredentialService
from ..events.event import Event
from ..memory.base_memory_service import BaseMemoryService
from ..runners import Runner
from ..sessions.base_session_service import BaseSessionService
from ..sessions.session import Session
from ..utils.context_utils import Aclosing
from ..utils.env_utils import is_env_enabled
from .service_registry import load_services_module
from .utils import envs
from .utils.agent_loader import AgentLoader
from .utils.service_factory import create_artifact_service_from_options
from .utils.service_factory import create_memory_service_from_options
from .utils.service_factory import create_session_service_from_options

logger = logging.getLogger('google_adk.' + __name__)


class InputFile(BaseModel):
  state: dict[str, object]
  queries: list[str]


def _to_app(agent_or_app: Union[BaseAgent, App, Any], app_name: str) -> App:
  """Wraps a BaseAgent or BaseNode in an App if not already one."""
  if isinstance(agent_or_app, App):
    return agent_or_app
  return App(name=app_name, root_agent=agent_or_app)


async def run_input_file(
    app_name: str,
    user_id: str,
    agent_or_app: Union[LlmAgent, App],
    artifact_service: BaseArtifactService,
    session_service: BaseSessionService,
    credential_service: BaseCredentialService,
    input_path: str,
    memory_service: Optional[BaseMemoryService] = None,
) -> Session:
  app = _to_app(agent_or_app, app_name)
  runner = Runner(
      app=app,
      artifact_service=artifact_service,
      session_service=session_service,
      memory_service=memory_service,
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


_REQUEST_INPUT = 'adk_request_input'
_REQUEST_CONFIRMATION = 'adk_request_confirmation'


def _collect_pending_function_calls(
    events: list[Event],
) -> list[tuple[str, str, dict[str, Any]]]:
  """Collects pending HITL function calls from events.

  Returns a list of (function_call_id, function_name, args) tuples
  for function calls that need user input.
  """
  pending = []
  for event in events:
    lr_ids = getattr(event, 'long_running_tool_ids', None)
    if not lr_ids:
      continue
    content = getattr(event, 'content', None)
    if not content or not content.parts:
      continue
    for part in content.parts:
      fc = part.function_call
      if fc and fc.id in lr_ids:
        pending.append((fc.id, fc.name, fc.args or {}))
  return pending


def _is_positive_response(s: str) -> bool:
  """Returns True if the string is a positive response."""
  return s.strip().lower() in ('y', 'yes', 'true', 'confirm')


def _prompt_for_function_call(
    fc_id: str, fc_name: str, args: dict[str, Any]
) -> types.Content:
  """Prompts the user for a HITL function call and returns the response."""
  if fc_name == _REQUEST_INPUT:
    message = args.get('message') or 'Input requested'
    schema = args.get('response_schema')
    click.echo(f'[HITL input] {message}')
    if schema:
      click.echo(f'  Schema: {json.dumps(schema)}')
  elif fc_name == _REQUEST_CONFIRMATION:
    tool_confirmation = args.get('toolConfirmation', {})
    hint = tool_confirmation.get('hint', '')
    original_fc = args.get('originalFunctionCall', {})
    original_name = original_fc.get('name', 'unknown')
    click.echo(f'[HITL confirm] {hint or f"Confirm {original_name}?"}')
    click.echo('  Type "yes" to confirm, anything else to reject.')
  else:
    click.echo(f'[HITL] Waiting for input for {fc_name}({args})')

  user_input = input('[user]: ')

  # Build the FunctionResponse.
  if fc_name == _REQUEST_CONFIRMATION:
    confirmed = _is_positive_response(user_input)
    response = {'confirmed': confirmed}
  else:
    # Try to parse as JSON, fall back to wrapping as {"result": value}.
    try:
      parsed = json.loads(user_input)
      response = parsed if isinstance(parsed, dict) else {'result': parsed}
    except (json.JSONDecodeError, ValueError):
      response = {'result': user_input}

  return types.Content(
      role='user',
      parts=[
          types.Part(
              function_response=types.FunctionResponse(
                  id=fc_id,
                  name=fc_name,
                  response=response,
              )
          )
      ],
  )


async def run_interactively(
    root_agent_or_app: Union[LlmAgent, App],
    artifact_service: BaseArtifactService,
    session: Session,
    session_service: BaseSessionService,
    credential_service: BaseCredentialService,
    memory_service: Optional[BaseMemoryService] = None,
    timeout: Optional[str] = None,
    jsonl: bool = False,
) -> None:
  app = _to_app(root_agent_or_app, session.app_name)
  runner = Runner(
      app=app,
      artifact_service=artifact_service,
      session_service=session_service,
      memory_service=memory_service,
      credential_service=credential_service,
  )

  next_message = None
  resume_invocation_id = None
  while True:
    if next_message is None:
      query = input('[user]: ')
      if not query or not query.strip():
        continue
      if query == 'exit':
        break
      next_message = types.Content(role='user', parts=[types.Part(text=query)])

    collected_events = []
    invocation_id = None

    async def run_and_print():
      nonlocal invocation_id
      async with Aclosing(
          runner.run_async(
              user_id=session.user_id,
              session_id=session.id,
              new_message=next_message,
              invocation_id=resume_invocation_id,
          )
      ) as agen:
        async for event in agen:
          collected_events.append(event)
          if getattr(event, 'invocation_id', None):
            invocation_id = event.invocation_id
          _print_event(event, jsonl=jsonl, session_id=session.id)

    try:
      if timeout:
        seconds = _parse_timeout(timeout)
        await asyncio.wait_for(run_and_print(), timeout=seconds)
      else:
        await run_and_print()
    except asyncio.TimeoutError:
      click.secho(
          f'Error: Command timed out after {timeout}', fg='red', err=True
      )
      next_message = None
      resume_invocation_id = None
      continue

    next_message = None
    resume_invocation_id = None

    # Check for pending HITL function calls that need user input.
    pending = _collect_pending_function_calls(collected_events)
    if pending:
      # Handle each pending function call. If there are multiple,
      # collect all responses into a single Content with multiple parts.
      parts = []
      for fc_id, fc_name, args in pending:
        response_content = _prompt_for_function_call(fc_id, fc_name, args)
        parts.extend(response_content.parts)
      next_message = types.Content(role='user', parts=parts)
      resume_invocation_id = invocation_id

  await runner.close()


def _override_default_llm_model(default_llm_model: str):
  """Overrides the default LLM model for LlmAgent."""
  logger.info('Overriding default model to %s', default_llm_model)
  LlmAgent.set_default_model(default_llm_model)


def _setup_runner_context(
    *,
    agent_parent_dir: str,
    agent_folder_name: str,
    in_memory: bool = False,
    session_service_uri: Optional[str] = None,
    artifact_service_uri: Optional[str] = None,
    memory_service_uri: Optional[str] = None,
    use_local_storage: bool = True,
    default_llm_model: Optional[str] = None,
):
  """Sets up the agent, services, and environment for running.

  Returns a tuple containing the loaded agent/app, services, and other
  contextual information needed for execution.
  """
  agent_parent_path = Path(agent_parent_dir).resolve()
  agent_root = agent_parent_path / agent_folder_name
  load_services_module(str(agent_root))
  user_id = 'test_user'

  agents_dir = str(agent_parent_path)
  agent_loader = AgentLoader(agents_dir=agents_dir)
  agent_or_app = agent_loader.load_agent(agent_folder_name)

  if default_llm_model:
    _override_default_llm_model(default_llm_model)
  session_app_name = (
      agent_or_app.name if isinstance(agent_or_app, App) else agent_folder_name
  )
  app_name_to_dir = None
  if isinstance(agent_or_app, App) and agent_or_app.name != agent_folder_name:
    app_name_to_dir = {agent_or_app.name: agent_folder_name}

  if not is_env_enabled('ADK_DISABLE_LOAD_DOTENV'):
    envs.load_dotenv_for_agent(agent_folder_name, agents_dir)

  if in_memory:
    session_service_uri = 'memory://'
    artifact_service_uri = 'memory://'
    use_local_storage = False

  session_service = create_session_service_from_options(
      base_dir=agent_parent_path,
      session_service_uri=session_service_uri,
      app_name_to_dir=app_name_to_dir,
      use_local_storage=use_local_storage,
  )

  artifact_service = create_artifact_service_from_options(
      base_dir=agent_parent_path,
      artifact_service_uri=artifact_service_uri,
      app_name_to_dir=app_name_to_dir,
      use_local_storage=use_local_storage,
  )
  memory_service = create_memory_service_from_options(
      base_dir=agent_parent_path,
      memory_service_uri=memory_service_uri,
  )

  credential_service = InMemoryCredentialService()

  return (
      agent_or_app,
      session_service,
      artifact_service,
      memory_service,
      credential_service,
      user_id,
      session_app_name,
      agent_root,
  )


def _print_event(
    event: Event, jsonl: bool = False, session_id: Optional[str] = None
):
  """Prints an event to the console.

  Args:
    event: The Event object to print.
    jsonl: If True, outputs structured JSONL to stdout. Otherwise, outputs
      human-readable text.
    session_id: Optional session ID to inject into the JSONL output.
  """
  if jsonl:
    event_dict = event.model_dump(mode='json', by_alias=True, exclude_none=True)
    if session_id:
      event_dict['session_id'] = session_id
    if event.node_info and event.node_info.path:
      event_dict['node_path'] = event.node_info.path

    # Filter out empty dictionaries in 'actions' (e.g., empty state delta) to
    # reduce noise
    if 'actions' in event_dict and isinstance(event_dict['actions'], dict):
      event_dict['actions'] = {
          k: v for k, v in event_dict['actions'].items() if v != {}
      }
      if not event_dict['actions']:
        del event_dict['actions']

    # Optimize key order for human readability in JSONL viewers
    ordered_dict = {}
    for k in ['author', 'session_id', 'node_path', 'id']:
      if k in event_dict:
        ordered_dict[k] = event_dict[k]
    for k, v in event_dict.items():
      if k not in ordered_dict:
        ordered_dict[k] = v
    click.echo(json.dumps(ordered_dict))
  else:
    # Human readable mode
    author = event.author or 'unknown'
    text_parts = (
        [p.text for p in event.content.parts if p.text] if event.content else []
    )
    if text_parts:
      text = ''.join(text_parts)
      click.echo(f'[{author}]: {text}')
    elif event.long_running_tool_ids:
      click.secho(f'[{author}]: (Paused for input...)', fg='yellow')


async def run_cli(
    *,
    agent_parent_dir: str,
    agent_folder_name: str,
    input_file: Optional[str] = None,
    saved_session_file: Optional[str] = None,
    save_session: bool,
    session_id: Optional[str] = None,
    state_str: Optional[str] = None,
    timeout: Optional[str] = None,
    in_memory: bool = False,
    jsonl: bool = False,
    session_service_uri: Optional[str] = None,
    artifact_service_uri: Optional[str] = None,
    memory_service_uri: Optional[str] = None,
    use_local_storage: bool = True,
    default_llm_model: Optional[str] = None,
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
    memory_service_uri: Optional[str], custom memory service URI.
    use_local_storage: bool, whether to use local .adk storage by default.
  """
  (
      agent_or_app,
      session_service,
      artifact_service,
      memory_service,
      credential_service,
      user_id,
      session_app_name,
      agent_root,
  ) = _setup_runner_context(
      agent_parent_dir=agent_parent_dir,
      agent_folder_name=agent_folder_name,
      in_memory=in_memory,
      session_service_uri=session_service_uri,
      artifact_service_uri=artifact_service_uri,
      memory_service_uri=memory_service_uri,
      use_local_storage=use_local_storage,
      default_llm_model=default_llm_model,
  )

  # Helper function for printing events
  if input_file:
    session = await run_input_file(
        app_name=session_app_name,
        user_id=user_id,
        agent_or_app=agent_or_app,
        artifact_service=artifact_service,
        session_service=session_service,
        memory_service=memory_service,
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
        _print_event(event, jsonl=jsonl, session_id=session.id)

    await run_interactively(
        agent_or_app,
        artifact_service,
        session,
        session_service,
        credential_service,
        memory_service=memory_service,
        timeout=timeout,
        jsonl=jsonl,
    )
  else:
    initial_state = None
    if state_str:
      try:
        initial_state = json.loads(state_str)
      except json.JSONDecodeError as e:
        click.secho(f'Error: Invalid JSON for --state: {e}', fg='red', err=True)
        return
    session = await session_service.create_session(
        app_name=session_app_name, user_id=user_id, state=initial_state
    )
    click.echo(f'Running agent {agent_or_app.name}, type exit to exit.')
    await run_interactively(
        agent_or_app,
        artifact_service,
        session,
        session_service,
        credential_service,
        memory_service=memory_service,
        timeout=timeout,
        jsonl=jsonl,
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


def _parse_timeout(timeout_str: str) -> float:
  """Parses a timeout string like '30s', '5m' into seconds."""
  match = re.match(r'^(\d+)([sm])?$', timeout_str)
  if not match:
    raise ValueError(f'Invalid timeout format: {timeout_str}')
  val, unit = match.groups()
  seconds = float(val)
  if unit == 'm':
    seconds *= 60
  return seconds


async def run_once_cli(
    *,
    agent_parent_dir: str,
    agent_folder_name: str,
    query: Optional[str] = None,
    state_str: Optional[str] = None,
    session_id: Optional[str] = None,
    replay: Optional[str] = None,
    timeout: Optional[str] = None,
    in_memory: bool = False,
    jsonl: bool = False,
    session_service_uri: Optional[str] = None,
    artifact_service_uri: Optional[str] = None,
    memory_service_uri: Optional[str] = None,
    use_local_storage: bool = True,
    default_llm_model: Optional[str] = None,
) -> int:
  """Runs an agent in query/automated mode."""
  (
      agent_or_app,
      session_service,
      artifact_service,
      memory_service,
      credential_service,
      user_id,
      session_app_name,
      agent_root,
  ) = _setup_runner_context(
      agent_parent_dir=agent_parent_dir,
      agent_folder_name=agent_folder_name,
      in_memory=in_memory,
      session_service_uri=session_service_uri,
      artifact_service_uri=artifact_service_uri,
      memory_service_uri=memory_service_uri,
      use_local_storage=use_local_storage,
      default_llm_model=default_llm_model,
  )

  parsed_state = None
  if state_str:
    try:
      parsed_state = json.loads(state_str)
    except json.JSONDecodeError as e:
      click.secho(f'Error: Invalid JSON for --state: {e}', fg='red', err=True)
      return 1

  if query and replay:
    click.secho(
        'Error: Cannot provide both query and --replay.', fg='red', err=True
    )
    return 1

  if not query and not replay:
    if not sys.stdin.isatty():
      query = sys.stdin.read().strip()
    else:
      click.secho(
          'Error: Missing query argument or stdin input.', fg='red', err=True
      )
      return 1

  app = _to_app(agent_or_app, session_app_name)
  runner = Runner(
      app=app,
      artifact_service=artifact_service,
      session_service=session_service,
      memory_service=memory_service,
      credential_service=credential_service,
  )

  if replay:
    with open(replay, 'r', encoding='utf-8') as f:
      input_file = InputFile.model_validate_json(f.read())
    session = await session_service.create_session(
        app_name=session_app_name,
        user_id=user_id,
        state=input_file.state,
        session_id=session_id,
    )
    queries = input_file.queries
  else:
    if session_id:
      session = await session_service.get_session(
          app_name=session_app_name, user_id=user_id, session_id=session_id
      )
      if not session:
        session = await session_service.create_session(
            app_name=session_app_name,
            user_id=user_id,
            state=parsed_state,
            session_id=session_id,
        )
    else:
      session = await session_service.create_session(
          app_name=session_app_name, user_id=user_id, state=parsed_state
      )
    queries = [query] if query else []

  # Output session ID once per run to stderr for humans
  if not jsonl:
    click.secho(f'Session ID: {session.id}', fg='yellow', err=True)

  exit_code = 0

  async def execute_query(query: str):
    nonlocal exit_code

    # Auto-resume magic: Check if the last event in the session indicates an
    # active interrupt (Human-In-The-Loop suspension). If so, we automatically
    # map the user's text query to the required function response instead of
    # treating it as a new user message.
    # Find the last event with active interrupts
    interrupt_event = None
    for e in reversed(session.events):
      if e.long_running_tool_ids:
        interrupt_event = e
        break

    if interrupt_event:
      # Assume the first active interrupt is the one we want to answer
      interrupt_id = list(interrupt_event.long_running_tool_ids)[0]
      if not jsonl:
        click.secho(
            f'Auto-resuming interrupt {interrupt_id} with input: {query}',
            fg='cyan',
            err=True,
        )

      # Construct a FunctionResponse pointing back to the interrupt ID.
      # We check the synthetic function name to handle different interrupt types.
      # TODO: We still need to handle 'adk_request_credential' (auth).
      # TODO: Support batch HITL or interactive selection when multiple
      # interrupts are active.
      fc = next(
          (
              c
              for c in interrupt_event.get_function_calls()
              if c.id == interrupt_id
          ),
          None,
      )

      if fc and fc.name == 'adk_request_confirmation':
        # Try to parse as JSON to support passing custom payload or explicit confirmed flag.
        try:
          parsed = json.loads(query)
          if isinstance(parsed, dict):
            response = parsed
          else:
            response = {'confirmed': _is_positive_response(query)}
        except (json.JSONDecodeError, ValueError):
          response = {'confirmed': _is_positive_response(query)}

        content = types.Content(
            role='user',
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        id=interrupt_id,
                        name='adk_request_confirmation',
                        response=response,
                    )
                )
            ],
        )
      else:
        # Fallback to adk_request_input or default behavior
        content = types.Content(
            role='user',
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        id=interrupt_id,
                        name='adk_request_input',
                        response={'result': query},
                    )
                )
            ],
        )
    else:
      # Standard flow: Treat the query as a new text message from the user
      content = types.Content(role='user', parts=[types.Part(text=query)])

    async with Aclosing(
        runner.run_async(
            user_id=session.user_id,
            session_id=session.id,
            invocation_id=interrupt_event.invocation_id
            if interrupt_event
            else None,
            new_message=content,
        )
    ) as agen:
      async for event in agen:
        _print_event(event, jsonl=jsonl, session_id=session.id)
        if event.long_running_tool_ids:
          exit_code = 2

      if exit_code == 2 and not jsonl:
        click.secho(
            '\n'
            + '=' * 60
            + '\n'
            '🚨 [PAUSED] Workflow is waiting for human input! 🚨\n\n'
            'To resume, run the command again with:\n'
            f'  --session_id {session.id}\n'
            'And provide your input as the query.\n'
            + '=' * 60
            + '\n',
            fg='yellow',
            bold=True,
            err=True,
        )

  try:
    for q in queries:
      if timeout:
        seconds = _parse_timeout(timeout)
        await asyncio.wait_for(execute_query(q), timeout=seconds)
      else:
        await execute_query(q)
  except asyncio.TimeoutError:
    click.secho(f'Error: Command timed out after {timeout}', fg='red', err=True)
    return 1
  except Exception as e:
    click.secho(f'Error: {e}', fg='red', err=True)
    return 1
  finally:
    await runner.close()

  return exit_code
