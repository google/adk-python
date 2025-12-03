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
import json
import os
import shutil
import subprocess
from typing import Final
from typing import Optional

import click
from packaging.version import parse

_IS_WINDOWS = os.name == 'nt'
_GCLOUD_CMD = 'gcloud.cmd' if _IS_WINDOWS else 'gcloud'

_DOCKERFILE_TEMPLATE: Final[str] = """
FROM python:3.11-slim
WORKDIR /app

# Create a non-root user
RUN adduser --disabled-password --gecos "" myuser

# Switch to the non-root user
USER myuser

# Set up environment variables - Start
ENV PATH="/home/myuser/.local/bin:$PATH"

ENV GOOGLE_GENAI_USE_VERTEXAI=1
ENV GOOGLE_CLOUD_PROJECT={gcp_project_id}
ENV GOOGLE_CLOUD_LOCATION={gcp_region}

# Set up environment variables - End

# Install ADK - Start
RUN pip install google-adk=={adk_version}
# Install ADK - End

# Copy agent - Start

# Set permission
COPY --chown=myuser:myuser "agents/{app_name}/" "/app/agents/{app_name}/"

# Copy agent - End

# Install Agent Deps - Start
{install_agent_deps}
# Install Agent Deps - End

EXPOSE {port}

CMD adk {command} --port={port} {host_option} {service_option} {trace_to_cloud_option} {allow_origins_option} {a2a_option} "/app/agents"
"""

_AGENT_ENGINE_APP_TEMPLATE: Final[str] = """
import os
import vertexai
from vertexai.agent_engines import AdkApp

if {is_config_agent}:
  from google.adk.agents import config_agent_utils
  try:
    # This path is for local loading.
    root_agent = config_agent_utils.from_config("{agent_folder}/root_agent.yaml")
  except FileNotFoundError:
    # This path is used to support the file structure in Agent Engine.
    root_agent = config_agent_utils.from_config("./{temp_folder}/{app_name}/root_agent.yaml")
else:
  from .agent import {adk_app_object}

if {express_mode}: # Whether or not to use Express Mode
  vertexai.init(api_key=os.environ.get("GOOGLE_API_KEY"))
else:
  vertexai.init(
    project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
    location=os.environ.get("GOOGLE_CLOUD_LOCATION"),
  )

adk_app = AdkApp(
    {adk_app_type}={adk_app_object},
    enable_tracing={trace_to_cloud_option},
)
"""

_AGENT_ENGINE_CLASS_METHODS = [
    {
        'name': 'get_session',
        'description': (
            'Deprecated. Use async_get_session instead.\n\n        Get a'
            ' session for the given user.\n        '
        ),
        'parameters': {
            'properties': {
                'user_id': {'type': 'string'},
                'session_id': {'type': 'string'},
            },
            'required': ['user_id', 'session_id'],
            'type': 'object',
        },
        'api_mode': '',
    },
    {
        'name': 'list_sessions',
        'description': (
            'Deprecated. Use async_list_sessions instead.\n\n        List'
            ' sessions for the given user.\n        '
        ),
        'parameters': {
            'properties': {'user_id': {'type': 'string'}},
            'required': ['user_id'],
            'type': 'object',
        },
        'api_mode': '',
    },
    {
        'name': 'create_session',
        'description': (
            'Deprecated. Use async_create_session instead.\n\n        Creates a'
            ' new session.\n        '
        ),
        'parameters': {
            'properties': {
                'user_id': {'type': 'string'},
                'session_id': {'type': 'string', 'nullable': True},
                'state': {'type': 'object', 'nullable': True},
            },
            'required': ['user_id'],
            'type': 'object',
        },
        'api_mode': '',
    },
    {
        'name': 'delete_session',
        'description': (
            'Deprecated. Use async_delete_session instead.\n\n        Deletes a'
            ' session for the given user.\n        '
        ),
        'parameters': {
            'properties': {
                'user_id': {'type': 'string'},
                'session_id': {'type': 'string'},
            },
            'required': ['user_id', 'session_id'],
            'type': 'object',
        },
        'api_mode': '',
    },
    {
        'name': 'async_get_session',
        'description': (
            'Get a session for the given user.\n\n        Args:\n           '
            ' user_id (str):\n                Required. The ID of the user.\n  '
            '          session_id (str):\n                Required. The ID of'
            ' the session.\n            **kwargs (dict[str, Any]):\n           '
            '     Optional. Additional keyword arguments to pass to the\n      '
            '          session service.\n\n        Returns:\n           '
            ' Session: The session instance (if any). It returns None if the\n '
            '           session is not found.\n\n        Raises:\n           '
            ' RuntimeError: If the session is not found.\n        '
        ),
        'parameters': {
            'properties': {
                'user_id': {'type': 'string'},
                'session_id': {'type': 'string'},
            },
            'required': ['user_id', 'session_id'],
            'type': 'object',
        },
        'api_mode': 'async',
    },
    {
        'name': 'async_list_sessions',
        'description': (
            'List sessions for the given user.\n\n        Args:\n           '
            ' user_id (str):\n                Required. The ID of the user.\n  '
            '          **kwargs (dict[str, Any]):\n                Optional.'
            ' Additional keyword arguments to pass to the\n               '
            ' session service.\n\n        Returns:\n           '
            ' ListSessionsResponse: The list of sessions.\n        '
        ),
        'parameters': {
            'properties': {'user_id': {'type': 'string'}},
            'required': ['user_id'],
            'type': 'object',
        },
        'api_mode': 'async',
    },
    {
        'name': 'async_create_session',
        'description': (
            'Creates a new session.\n\n        Args:\n            user_id'
            ' (str):\n                Required. The ID of the user.\n          '
            '  session_id (str):\n                Optional. The ID of the'
            ' session. If not provided, an ID\n                will be be'
            ' generated for the session.\n            state (dict[str, Any]):\n'
            '                Optional. The initial state of the session.\n     '
            '       **kwargs (dict[str, Any]):\n                Optional.'
            ' Additional keyword arguments to pass to the\n               '
            ' session service.\n\n        Returns:\n            Session: The'
            ' newly created session instance.\n        '
        ),
        'parameters': {
            'properties': {
                'user_id': {'type': 'string'},
                'session_id': {'type': 'string', 'nullable': True},
                'state': {'type': 'object', 'nullable': True},
            },
            'required': ['user_id'],
            'type': 'object',
        },
        'api_mode': 'async',
    },
    {
        'name': 'async_delete_session',
        'description': (
            'Deletes a session for the given user.\n\n        Args:\n          '
            '  user_id (str):\n                Required. The ID of the user.\n '
            '           session_id (str):\n                Required. The ID of'
            ' the session.\n            **kwargs (dict[str, Any]):\n           '
            '     Optional. Additional keyword arguments to pass to the\n      '
            '          session service.\n        '
        ),
        'parameters': {
            'properties': {
                'user_id': {'type': 'string'},
                'session_id': {'type': 'string'},
            },
            'required': ['user_id', 'session_id'],
            'type': 'object',
        },
        'api_mode': 'async',
    },
    {
        'name': 'async_add_session_to_memory',
        'description': (
            'Generates memories.\n\n        Args:\n            session'
            ' (Dict[str, Any]):\n                Required. The session to use'
            ' for generating memories. It should\n                be a'
            ' dictionary representing an ADK Session object, e.g.\n            '
            '    session.model_dump(mode="json").\n        '
        ),
        'parameters': {
            'properties': {
                'session': {'additionalProperties': True, 'type': 'object'}
            },
            'required': ['session'],
            'type': 'object',
        },
        'api_mode': 'async',
    },
    {
        'name': 'async_search_memory',
        'description': (
            'Searches memories for the given user.\n\n        Args:\n          '
            '  user_id: The id of the user.\n            query: The query to'
            ' match the memories on.\n\n        Returns:\n            A'
            ' SearchMemoryResponse containing the matching memories.\n        '
        ),
        'parameters': {
            'properties': {
                'user_id': {'type': 'string'},
                'query': {'type': 'string'},
            },
            'required': ['user_id', 'query'],
            'type': 'object',
        },
        'api_mode': 'async',
    },
    {
        'name': 'stream_query',
        'description': (
            'Deprecated. Use async_stream_query instead.\n\n        Streams'
            ' responses from the ADK application in response to a message.\n\n '
            '       Args:\n            message (Union[str, Dict[str, Any]]):\n '
            '               Required. The message to stream responses for.\n   '
            '         user_id (str):\n                Required. The ID of the'
            ' user.\n            session_id (str):\n                Optional.'
            ' The ID of the session. If not provided, a new\n               '
            ' session will be created for the user.\n            run_config'
            ' (Optional[Dict[str, Any]]):\n                Optional. The run'
            ' config to use for the query. If you want to\n                pass'
            ' in a `run_config` pydantic object, you can pass in a dict\n      '
            '          representing it as'
            ' `run_config.model_dump(mode="json")`.\n            **kwargs'
            ' (dict[str, Any]):\n                Optional. Additional keyword'
            ' arguments to pass to the\n                runner.\n\n       '
            ' Yields:\n            The output of querying the ADK'
            ' application.\n        '
        ),
        'parameters': {
            'properties': {
                'message': {
                    'anyOf': [
                        {'type': 'string'},
                        {'additionalProperties': True, 'type': 'object'},
                    ]
                },
                'user_id': {'type': 'string'},
                'session_id': {'type': 'string', 'nullable': True},
                'run_config': {'type': 'object', 'nullable': True},
            },
            'required': ['message', 'user_id'],
            'type': 'object',
        },
        'api_mode': 'stream',
    },
    {
        'name': 'async_stream_query',
        'description': (
            'Streams responses asynchronously from the ADK application.\n\n    '
            '    Args:\n            message (str):\n                Required.'
            ' The message to stream responses for.\n            user_id'
            ' (str):\n                Required. The ID of the user.\n          '
            '  session_id (str):\n                Optional. The ID of the'
            ' session. If not provided, a new\n                session will be'
            ' created for the user.\n            run_config (Optional[Dict[str,'
            ' Any]]):\n                Optional. The run config to use for the'
            ' query. If you want to\n                pass in a `run_config`'
            ' pydantic object, you can pass in a dict\n               '
            ' representing it as `run_config.model_dump(mode="json")`.\n       '
            '     **kwargs (dict[str, Any]):\n                Optional.'
            ' Additional keyword arguments to pass to the\n               '
            ' runner.\n\n        Yields:\n            Event dictionaries'
            ' asynchronously.\n        '
        ),
        'parameters': {
            'properties': {
                'message': {
                    'anyOf': [
                        {'type': 'string'},
                        {'additionalProperties': True, 'type': 'object'},
                    ]
                },
                'user_id': {'type': 'string'},
                'session_id': {'type': 'string', 'nullable': True},
                'run_config': {'type': 'object', 'nullable': True},
            },
            'required': ['message', 'user_id'],
            'type': 'object',
        },
        'api_mode': 'async_stream',
    },
    {
        'name': 'streaming_agent_run_with_events',
        'description': (
            'Streams responses asynchronously from the ADK application.\n\n    '
            '    In general, you should use `async_stream_query` instead, as it'
            ' has a\n        more structured API and works with the respective'
            ' ADK services that\n        you have defined for the AdkApp. This'
            ' method is primarily meant for\n        invocation from'
            ' AgentSpace.\n\n        Args:\n            request_json (str):\n  '
            '              Required. The request to stream responses for.\n   '
            '     '
        ),
        'parameters': {
            'properties': {'request_json': {'type': 'string'}},
            'required': ['request_json'],
            'type': 'object',
        },
        'api_mode': 'async_stream',
    },
]


def _resolve_project(project_in_option: Optional[str]) -> str:
  if project_in_option:
    return project_in_option

  result = subprocess.run(
      [_GCLOUD_CMD, 'config', 'get-value', 'project'],
      check=True,
      capture_output=True,
      text=True,
  )
  project = result.stdout.strip()
  click.echo(f'Use default project: {project}')
  return project


def _validate_gcloud_extra_args(
    extra_gcloud_args: Optional[tuple[str, ...]], adk_managed_args: set[str]
) -> None:
  """Validates that extra gcloud args don't conflict with ADK-managed args.

  This function dynamically checks for conflicts based on the actual args
  that ADK will set, rather than using a hardcoded list.

  Args:
    extra_gcloud_args: User-provided extra arguments for gcloud.
    adk_managed_args: Set of argument names that ADK will set automatically.
                     Should include '--' prefix (e.g., '--project').

  Raises:
    click.ClickException: If any conflicts are found.
  """
  if not extra_gcloud_args:
    return

  # Parse user arguments into a set of argument names for faster lookup
  user_arg_names = set()
  for arg in extra_gcloud_args:
    if arg.startswith('--'):
      # Handle both '--arg=value' and '--arg value' formats
      arg_name = arg.split('=')[0]
      user_arg_names.add(arg_name)

  # Check for conflicts with ADK-managed args
  conflicts = user_arg_names.intersection(adk_managed_args)

  if conflicts:
    conflict_list = ', '.join(f"'{arg}'" for arg in sorted(conflicts))
    if len(conflicts) == 1:
      raise click.ClickException(
          f"The argument {conflict_list} conflicts with ADK's automatic"
          ' configuration. ADK will set this argument automatically, so please'
          ' remove it from your command.'
      )
    else:
      raise click.ClickException(
          f"The arguments {conflict_list} conflict with ADK's automatic"
          ' configuration. ADK will set these arguments automatically, so'
          ' please remove them from your command.'
      )


def _get_service_option_by_adk_version(
    adk_version: str,
    session_uri: Optional[str],
    artifact_uri: Optional[str],
    memory_uri: Optional[str],
) -> str:
  """Returns service option string based on adk_version."""
  parsed_version = parse(adk_version)
  if parsed_version >= parse('1.3.0'):
    session_option = (
        f'--session_service_uri={session_uri}' if session_uri else ''
    )
    artifact_option = (
        f'--artifact_service_uri={artifact_uri}' if artifact_uri else ''
    )
    memory_option = f'--memory_service_uri={memory_uri}' if memory_uri else ''
    return f'{session_option} {artifact_option} {memory_option}'
  elif parsed_version >= parse('1.2.0'):
    session_option = f'--session_db_url={session_uri}' if session_uri else ''
    artifact_option = (
        f'--artifact_storage_uri={artifact_uri}' if artifact_uri else ''
    )
    return f'{session_option} {artifact_option}'
  else:
    return f'--session_db_url={session_uri}' if session_uri else ''


def to_cloud_run(
    *,
    agent_folder: str,
    project: Optional[str],
    region: Optional[str],
    service_name: str,
    app_name: str,
    temp_folder: str,
    port: int,
    trace_to_cloud: bool,
    with_ui: bool,
    log_level: str,
    verbosity: str,
    adk_version: str,
    allow_origins: Optional[list[str]] = None,
    session_service_uri: Optional[str] = None,
    artifact_service_uri: Optional[str] = None,
    memory_service_uri: Optional[str] = None,
    a2a: bool = False,
    extra_gcloud_args: Optional[tuple[str, ...]] = None,
):
  """Deploys an agent to Google Cloud Run.

  `agent_folder` should contain the following files:

  - __init__.py
  - agent.py
  - requirements.txt (optional, for additional dependencies)
  - ... (other required source files)

  The folder structure of temp_folder will be

  * dist/[google_adk wheel file]
  * agents/[app_name]/
    * agent source code from `agent_folder`

  Args:
    agent_folder: The folder (absolute path) containing the agent source code.
    project: Google Cloud project id.
    region: Google Cloud region.
    service_name: The service name in Cloud Run.
    app_name: The name of the app, by default, it's basename of `agent_folder`.
    temp_folder: The temp folder for the generated Cloud Run source files.
    port: The port of the ADK api server.
    trace_to_cloud: Whether to enable Cloud Trace.
    with_ui: Whether to deploy with UI.
    verbosity: The verbosity level of the CLI.
    adk_version: The ADK version to use in Cloud Run.
    allow_origins: The list of allowed origins for the ADK api server.
    session_service_uri: The URI of the session service.
    artifact_service_uri: The URI of the artifact service.
    memory_service_uri: The URI of the memory service.
  """
  app_name = app_name or os.path.basename(agent_folder)

  click.echo(f'Start generating Cloud Run source files in {temp_folder}')

  # remove temp_folder if exists
  if os.path.exists(temp_folder):
    click.echo('Removing existing files')
    shutil.rmtree(temp_folder)

  try:
    # copy agent source code
    click.echo('Copying agent source code...')
    agent_src_path = os.path.join(temp_folder, 'agents', app_name)
    shutil.copytree(agent_folder, agent_src_path)
    requirements_txt_path = os.path.join(agent_src_path, 'requirements.txt')
    install_agent_deps = (
        f'RUN pip install -r "/app/agents/{app_name}/requirements.txt"'
        if os.path.exists(requirements_txt_path)
        else '# No requirements.txt found.'
    )
    click.echo('Copying agent source code completed.')

    # create Dockerfile
    click.echo('Creating Dockerfile...')
    host_option = '--host=0.0.0.0' if adk_version > '0.5.0' else ''
    allow_origins_option = (
        f'--allow_origins={",".join(allow_origins)}' if allow_origins else ''
    )
    a2a_option = '--a2a' if a2a else ''
    dockerfile_content = _DOCKERFILE_TEMPLATE.format(
        gcp_project_id=project,
        gcp_region=region,
        app_name=app_name,
        port=port,
        command='web' if with_ui else 'api_server',
        install_agent_deps=install_agent_deps,
        service_option=_get_service_option_by_adk_version(
            adk_version,
            session_service_uri,
            artifact_service_uri,
            memory_service_uri,
        ),
        trace_to_cloud_option='--trace_to_cloud' if trace_to_cloud else '',
        allow_origins_option=allow_origins_option,
        adk_version=adk_version,
        host_option=host_option,
        a2a_option=a2a_option,
    )
    dockerfile_path = os.path.join(temp_folder, 'Dockerfile')
    os.makedirs(temp_folder, exist_ok=True)
    with open(dockerfile_path, 'w', encoding='utf-8') as f:
      f.write(
          dockerfile_content,
      )
    click.echo(f'Creating Dockerfile complete: {dockerfile_path}')

    # Deploy to Cloud Run
    click.echo('Deploying to Cloud Run...')
    region_options = ['--region', region] if region else []
    project = _resolve_project(project)

    # Build the set of args that ADK will manage
    adk_managed_args = {'--source', '--project', '--port', '--verbosity'}
    if region:
      adk_managed_args.add('--region')

    # Validate that extra gcloud args don't conflict with ADK-managed args
    _validate_gcloud_extra_args(extra_gcloud_args, adk_managed_args)

    # Build the command with extra gcloud args
    gcloud_cmd = [
        _GCLOUD_CMD,
        'run',
        'deploy',
        service_name,
        '--source',
        temp_folder,
        '--project',
        project,
        *region_options,
        '--port',
        str(port),
        '--verbosity',
        log_level.lower() if log_level else verbosity,
    ]

    # Handle labels specially - merge user labels with ADK label
    user_labels = []
    extra_args_without_labels = []

    if extra_gcloud_args:
      for arg in extra_gcloud_args:
        if arg.startswith('--labels='):
          # Extract user-provided labels
          user_labels_value = arg[9:]  # Remove '--labels=' prefix
          user_labels.append(user_labels_value)
        else:
          extra_args_without_labels.append(arg)

    # Combine ADK label with user labels
    all_labels = ['created-by=adk']
    all_labels.extend(user_labels)
    labels_arg = ','.join(all_labels)

    gcloud_cmd.extend(['--labels', labels_arg])

    # Add any remaining extra passthrough args
    gcloud_cmd.extend(extra_args_without_labels)

    subprocess.run(gcloud_cmd, check=True)
  finally:
    click.echo(f'Cleaning up the temp folder: {temp_folder}')
    shutil.rmtree(temp_folder)


def to_agent_engine(
    *,
    agent_folder: str,
    temp_folder: Optional[str] = None,
    adk_app: str,
    staging_bucket: str,
    trace_to_cloud: Optional[bool] = None,
    api_key: Optional[str] = None,
    adk_app_object: Optional[str] = None,
    agent_engine_id: Optional[str] = None,
    absolutize_imports: bool = True,
    project: Optional[str] = None,
    region: Optional[str] = None,
    display_name: Optional[str] = None,
    description: Optional[str] = None,
    requirements_file: Optional[str] = None,
    env_file: Optional[str] = None,
    agent_engine_config_file: Optional[str] = None,
):
  """Deploys an agent to Vertex AI Agent Engine.

  `agent_folder` should contain the following files:

  - __init__.py
  - agent.py
  - <adk_app>.py (optional, for customization; will be autogenerated otherwise)
  - requirements.txt (optional, for additional dependencies)
  - .env (optional, for environment variables)
  - ... (other required source files)

  The contents of `adk_app` should look something like:

  ```
  from agent import <adk_app_object>
  from vertexai.agent_engines import AdkApp

  adk_app = AdkApp(
    agent=<adk_app_object>,  # or `app=<adk_app_object>`
  )
  ```

  Args:
    agent_folder (str): The folder (absolute path) containing the agent source
      code.
    temp_folder (str): The temp folder for the generated Agent Engine source
      files. It will be replaced with the generated files if it already exists.
    adk_app (str): The name of the file (without .py) containing the AdkApp
      instance.
    staging_bucket (str): The GCS bucket for staging the deployment artifacts.
    trace_to_cloud (bool): Whether to enable Cloud Trace.
    api_key (str): Optional. The API key to use for Express Mode.
      If not provided, the API key from the GOOGLE_API_KEY environment variable
      will be used. It will only be used if GOOGLE_GENAI_USE_VERTEXAI is true.
    adk_app_object (str): Optional. The Python object corresponding to the root
      ADK agent or app. Defaults to `root_agent` if not specified.
    agent_engine_id (str): Optional. The ID of the Agent Engine instance to
      update. If not specified, a new Agent Engine instance will be created.
    absolutize_imports (bool): Optional. Default is True. Whether to absolutize
      imports. If True, all relative imports will be converted to absolute
      import statements.
    project (str): Optional. Google Cloud project id.
    region (str): Optional. Google Cloud region.
    display_name (str): Optional. The display name of the Agent Engine.
    description (str): Optional. The description of the Agent Engine.
    requirements_file (str): Optional. The filepath to the `requirements.txt`
      file to use. If not specified, the `requirements.txt` file in the
      `agent_folder` will be used.
    env_file (str): Optional. The filepath to the `.env` file for environment
      variables. If not specified, the `.env` file in the `agent_folder` will be
      used. The values of `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION`
      will be overridden by `project` and `region` if they are specified.
    agent_engine_config_file (str): The filepath to the agent engine config file
      to use. If not specified, the `.agent_engine_config.json` file in the
      `agent_folder` will be used.
  """
  app_name = os.path.basename(agent_folder)
  display_name = display_name or app_name
  parent_folder = os.path.dirname(agent_folder)
  if parent_folder != os.getcwd():
    click.echo(f'Please deploy from the project dir: {parent_folder}')
    return
  tmp_app_name = app_name + '_tmp' + datetime.now().strftime('%Y%m%d_%H%M%S')
  temp_folder = temp_folder or tmp_app_name
  agent_src_path = os.path.join(parent_folder, temp_folder)
  click.echo(f'Staging all files in: {agent_src_path}')
  adk_app_object = adk_app_object or 'root_agent'
  if adk_app_object not in ['root_agent', 'app']:
    click.echo(
        f'Invalid adk_app_object: {adk_app_object}. Please use "root_agent"'
        ' or "app".'
    )
    return
  # remove agent_src_path if it exists
  if os.path.exists(agent_src_path):
    click.echo('Removing existing files')
    shutil.rmtree(agent_src_path)

  try:
    ignore_patterns = None
    ae_ignore_path = os.path.join(agent_folder, '.ae_ignore')
    if os.path.exists(ae_ignore_path):
      click.echo(f'Ignoring files matching the patterns in {ae_ignore_path}')
      with open(ae_ignore_path, 'r') as f:
        patterns = [pattern.strip() for pattern in f.readlines()]
        ignore_patterns = shutil.ignore_patterns(*patterns)
    click.echo('Copying agent source code...')
    shutil.copytree(agent_folder, agent_src_path, ignore=ignore_patterns)
    click.echo('Copying agent source code complete.')

    project = _resolve_project(project)

    click.echo('Resolving files and dependencies...')
    agent_config = {}
    if staging_bucket:
      agent_config['staging_bucket'] = staging_bucket
    if not agent_engine_config_file:
      # Attempt to read the agent engine config from .agent_engine_config.json in the dir (if any).
      agent_engine_config_file = os.path.join(
          agent_folder, '.agent_engine_config.json'
      )
    if os.path.exists(agent_engine_config_file):
      click.echo(f'Reading agent engine config from {agent_engine_config_file}')
      with open(agent_engine_config_file, 'r') as f:
        agent_config = json.load(f)
    if display_name:
      if 'display_name' in agent_config:
        click.echo(
            'Overriding display_name in agent engine config with'
            f' {display_name}'
        )
      agent_config['display_name'] = display_name
    if description:
      if 'description' in agent_config:
        click.echo(
            f'Overriding description in agent engine config with {description}'
        )
      agent_config['description'] = description

    if not requirements_file:
      # Attempt to read requirements from requirements.txt in the dir (if any).
      requirements_txt_path = os.path.join(agent_src_path, 'requirements.txt')
      if not os.path.exists(requirements_txt_path):
        click.echo(f'Creating {requirements_txt_path}...')
        with open(requirements_txt_path, 'w', encoding='utf-8') as f:
          f.write('google-cloud-aiplatform[adk,agent_engines]')
        click.echo(f'Created {requirements_txt_path}')
      agent_config['requirements_file'] = agent_config.get(
          'requirements',
          requirements_txt_path,
      )
    else:
      if 'requirements_file' in agent_config:
        click.echo(
            'Overriding requirements in agent engine config with '
            f'{requirements_file}'
        )
      agent_config['requirements_file'] = requirements_file
    agent_config['requirements_file'] = f'{temp_folder}/requirements.txt'

    env_vars = {}
    if not env_file:
      # Attempt to read the env variables from .env in the dir (if any).
      env_file = os.path.join(agent_folder, '.env')
    if os.path.exists(env_file):
      from dotenv import dotenv_values

      click.echo(f'Reading environment variables from {env_file}')
      env_vars = dotenv_values(env_file)
      if 'GOOGLE_CLOUD_PROJECT' in env_vars:
        env_project = env_vars.pop('GOOGLE_CLOUD_PROJECT')
        if env_project:
          if project:
            click.secho(
                'Ignoring GOOGLE_CLOUD_PROJECT in .env as `--project` was'
                ' explicitly passed and takes precedence',
                fg='yellow',
            )
          else:
            project = env_project
            click.echo(f'{project=} set by GOOGLE_CLOUD_PROJECT in {env_file}')
      if 'GOOGLE_CLOUD_LOCATION' in env_vars:
        env_region = env_vars.pop('GOOGLE_CLOUD_LOCATION')
        if env_region:
          if region:
            click.secho(
                'Ignoring GOOGLE_CLOUD_LOCATION in .env as `--region` was'
                ' explicitly passed and takes precedence',
                fg='yellow',
            )
          else:
            region = env_region
            click.echo(f'{region=} set by GOOGLE_CLOUD_LOCATION in {env_file}')
    if api_key:
      if 'GOOGLE_API_KEY' in env_vars:
        click.secho(
            'Ignoring GOOGLE_API_KEY in .env as `--api_key` was'
            ' explicitly passed and takes precedence',
            fg='yellow',
        )
      else:
        env_vars['GOOGLE_GENAI_USE_VERTEXAI'] = '1'
        env_vars['GOOGLE_API_KEY'] = api_key
    elif not project:
      if 'GOOGLE_API_KEY' in env_vars:
        api_key = env_vars['GOOGLE_API_KEY']
        click.echo(f'api_key set by GOOGLE_API_KEY in {env_file}')
    if env_vars:
      if 'env_vars' in agent_config:
        click.echo(
            f'Overriding env_vars in agent engine config with {env_vars}'
        )
      agent_config['env_vars'] = env_vars
    # Set env_vars in agent_config to None if it is not set.
    agent_config['env_vars'] = agent_config.get('env_vars', env_vars)

    import vertexai

    if project and region:
      click.echo('Initializing Vertex AI...')
      client = vertexai.Client(project=project, location=region)
    elif api_key:
      click.echo('Initializing Vertex AI in Express Mode with API key...')
      client = vertexai.Client(api_key=api_key)
    else:
      click.echo(
          'No project/region or api_key provided. '
          'Please specify either project/region or api_key.'
      )
      return
    click.echo('Vertex AI initialized.')

    is_config_agent = False
    config_root_agent_file = os.path.join(agent_src_path, 'root_agent.yaml')
    if os.path.exists(config_root_agent_file):
      click.echo(f'Config agent detected: {config_root_agent_file}')
      is_config_agent = True

    adk_app_file = os.path.join(temp_folder, f'{adk_app}.py')
    if adk_app_object == 'root_agent':
      adk_app_type = 'agent'
    elif adk_app_object == 'app':
      adk_app_type = 'app'
    else:
      click.echo(
          f'Invalid adk_app_object: {adk_app_object}. Please use "root_agent"'
          ' or "app".'
      )
      return
    with open(adk_app_file, 'w', encoding='utf-8') as f:
      f.write(
          _AGENT_ENGINE_APP_TEMPLATE.format(
              app_name=app_name,
              trace_to_cloud_option=trace_to_cloud,
              is_config_agent=is_config_agent,
              temp_folder=temp_folder,
              agent_folder=agent_folder,
              adk_app_object=adk_app_object,
              adk_app_type=adk_app_type,
              express_mode=api_key is not None,
          )
      )
    click.echo(f'Created {adk_app_file}')
    click.echo('Files and dependencies resolved')
    if absolutize_imports:
      click.echo(
          'Agent Engine deployments have switched to source-based deployment, '
          'so it is no longer necessary to absolutize imports.'
      )
    click.echo('Deploying to agent engine...')
    agent_config['entrypoint_module'] = f'{temp_folder}.{adk_app}'
    agent_config['entrypoint_object'] = 'adk_app'
    agent_config['source_packages'] = [temp_folder]
    agent_config['class_methods'] = _AGENT_ENGINE_CLASS_METHODS
    agent_config['agent_framework'] = 'google-adk'

    if not agent_engine_id:
      agent_engine = client.agent_engines.create(config=agent_config)
      click.secho(
          f'✅ Created agent engine: {agent_engine.api_resource.name}',
          fg='green',
      )
    else:
      if project and region and not agent_engine_id.startswith('projects/'):
        agent_engine_id = f'projects/{project}/locations/{region}/reasoningEngines/{agent_engine_id}'
      client.agent_engines.update(name=agent_engine_id, config=agent_config)
      click.secho(f'✅ Updated agent engine: {agent_engine_id}', fg='green')
  finally:
    click.echo(f'Cleaning up the temp folder: {temp_folder}')
    shutil.rmtree(temp_folder)


def to_gke(
    *,
    agent_folder: str,
    project: Optional[str],
    region: Optional[str],
    cluster_name: str,
    service_name: str,
    app_name: str,
    temp_folder: str,
    port: int,
    trace_to_cloud: bool,
    with_ui: bool,
    log_level: str,
    adk_version: str,
    allow_origins: Optional[list[str]] = None,
    session_service_uri: Optional[str] = None,
    artifact_service_uri: Optional[str] = None,
    memory_service_uri: Optional[str] = None,
    a2a: bool = False,
):
  """Deploys an agent to Google Kubernetes Engine(GKE).

  Args:
    agent_folder: The folder (absolute path) containing the agent source code.
    project: Google Cloud project id.
    region: Google Cloud region.
    cluster_name: The name of the GKE cluster.
    service_name: The service name in GKE.
    app_name: The name of the app, by default, it's basename of `agent_folder`.
    temp_folder: The local directory to use as a temporary workspace for
      preparing deployment artifacts. The tool populates this folder with a copy
      of the agent's source code and auto-generates necessary files like a
      Dockerfile and deployment.yaml.
    port: The port of the ADK api server.
    trace_to_cloud: Whether to enable Cloud Trace.
    with_ui: Whether to deploy with UI.
    log_level: The logging level.
    adk_version: The ADK version to use in GKE.
    allow_origins: The list of allowed origins for the ADK api server.
    session_service_uri: The URI of the session service.
    artifact_service_uri: The URI of the artifact service.
    memory_service_uri: The URI of the memory service.
  """
  click.secho(
      '\n🚀 Starting ADK Agent Deployment to GKE...', fg='cyan', bold=True
  )
  click.echo('--------------------------------------------------')
  # Resolve project early to show the user which one is being used
  project = _resolve_project(project)
  click.echo(f'  Project:         {project}')
  click.echo(f'  Region:          {region}')
  click.echo(f'  Cluster:         {cluster_name}')
  click.echo('--------------------------------------------------\n')

  app_name = app_name or os.path.basename(agent_folder)

  click.secho('STEP 1: Preparing build environment...', bold=True)
  click.echo(f'  - Using temporary directory: {temp_folder}')

  # remove temp_folder if exists
  if os.path.exists(temp_folder):
    click.echo('  - Removing existing temporary directory...')
    shutil.rmtree(temp_folder)

  try:
    # copy agent source code
    click.echo('  - Copying agent source code...')
    agent_src_path = os.path.join(temp_folder, 'agents', app_name)
    shutil.copytree(agent_folder, agent_src_path)
    requirements_txt_path = os.path.join(agent_src_path, 'requirements.txt')
    install_agent_deps = (
        f'RUN pip install -r "/app/agents/{app_name}/requirements.txt"'
        if os.path.exists(requirements_txt_path)
        else ''
    )
    click.secho('✅ Environment prepared.', fg='green')

    allow_origins_option = (
        f'--allow_origins={",".join(allow_origins)}' if allow_origins else ''
    )

    # create Dockerfile
    click.secho('\nSTEP 2: Generating deployment files...', bold=True)
    click.echo('  - Creating Dockerfile...')
    host_option = '--host=0.0.0.0' if adk_version > '0.5.0' else ''
    dockerfile_content = _DOCKERFILE_TEMPLATE.format(
        gcp_project_id=project,
        gcp_region=region,
        app_name=app_name,
        port=port,
        command='web' if with_ui else 'api_server',
        install_agent_deps=install_agent_deps,
        service_option=_get_service_option_by_adk_version(
            adk_version,
            session_service_uri,
            artifact_service_uri,
            memory_service_uri,
        ),
        trace_to_cloud_option='--trace_to_cloud' if trace_to_cloud else '',
        allow_origins_option=allow_origins_option,
        adk_version=adk_version,
        host_option=host_option,
        a2a_option='--a2a' if a2a else '',
    )
    dockerfile_path = os.path.join(temp_folder, 'Dockerfile')
    os.makedirs(temp_folder, exist_ok=True)
    with open(dockerfile_path, 'w', encoding='utf-8') as f:
      f.write(
          dockerfile_content,
      )
    click.secho(f'✅ Dockerfile generated: {dockerfile_path}', fg='green')

    # Build and push the Docker image
    click.secho(
        '\nSTEP 3: Building container image with Cloud Build...', bold=True
    )
    click.echo(
        '  (This may take a few minutes. Raw logs from gcloud will be shown'
        ' below.)'
    )
    project = _resolve_project(project)
    image_name = f'gcr.io/{project}/{service_name}'
    subprocess.run(
        [
            'gcloud',
            'builds',
            'submit',
            '--tag',
            image_name,
            '--verbosity',
            log_level.lower(),
            temp_folder,
        ],
        check=True,
    )
    click.secho('✅ Container image built and pushed successfully.', fg='green')

    # Create a Kubernetes deployment
    click.echo('  - Creating Kubernetes deployment.yaml...')
    deployment_yaml = f"""
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {service_name}
  labels:
    app.kubernetes.io/name: adk-agent
    app.kubernetes.io/version: {adk_version}
    app.kubernetes.io/instance: {service_name}
    app.kubernetes.io/managed-by: adk-cli
spec:
  replicas: 1
  selector:
    matchLabels:
      app: {service_name}
  template:
    metadata:
      labels:
        app: {service_name}
        app.kubernetes.io/name: adk-agent
        app.kubernetes.io/version: {adk_version}
        app.kubernetes.io/instance: {service_name}
        app.kubernetes.io/managed-by: adk-cli
    spec:
      containers:
      - name: {service_name}
        image: {image_name}
        ports:
        - containerPort: {port}
---
apiVersion: v1
kind: Service
metadata:
  name: {service_name}
spec:
  type: LoadBalancer
  selector:
    app: {service_name}
  ports:
  - port: 80
    targetPort: {port}
"""
    deployment_yaml_path = os.path.join(temp_folder, 'deployment.yaml')
    with open(deployment_yaml_path, 'w', encoding='utf-8') as f:
      f.write(deployment_yaml)
    click.secho(
        f'✅ Kubernetes deployment manifest generated: {deployment_yaml_path}',
        fg='green',
    )

    # Apply the deployment
    click.secho('\nSTEP 4: Applying deployment to GKE cluster...', bold=True)
    click.echo('  - Getting cluster credentials...')
    subprocess.run(
        [
            'gcloud',
            'container',
            'clusters',
            'get-credentials',
            cluster_name,
            '--region',
            region,
            '--project',
            project,
        ],
        check=True,
    )
    click.echo('  - Applying Kubernetes manifest...')
    result = subprocess.run(
        ['kubectl', 'apply', '-f', temp_folder],
        check=True,
        capture_output=True,  # <-- Add this
        text=True,  # <-- Add this
    )

    # 2. Print the captured output line by line
    click.secho(
        '  - The following resources were applied to the cluster:', fg='green'
    )
    for line in result.stdout.strip().split('\n'):
      click.echo(f'    - {line}')

  finally:
    click.secho('\nSTEP 5: Cleaning up...', bold=True)
    click.echo(f'  - Removing temporary directory: {temp_folder}')
    shutil.rmtree(temp_folder)
  click.secho(
      '\n🎉 Deployment to GKE finished successfully!', fg='cyan', bold=True
  )
