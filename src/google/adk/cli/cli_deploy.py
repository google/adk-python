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

from datetime import datetime
import importlib
import json
import os
import shutil
import subprocess
import sys
import traceback
from typing import Final
from typing import Literal
from typing import Optional
import warnings

import click
from packaging.version import parse

from ..version import __version__
from .utils import _onboarding

_IS_WINDOWS = os.name == 'nt'
_GCLOUD_CMD = 'gcloud.cmd' if _IS_WINDOWS else 'gcloud'
_LOCAL_STORAGE_FLAG_MIN_VERSION: Final[str] = '1.21.0'
_AGENT_ENGINE_REQUIREMENT: Final[str] = (
    'google-cloud-aiplatform[adk,agent_engines]'
)


def _ensure_agent_engine_dependency(requirements_txt_path: str) -> None:
  """Ensures staged requirements include Agent Platform dependencies."""
  if not os.path.exists(requirements_txt_path):
    raise FileNotFoundError(
        f'requirements.txt not found at: {requirements_txt_path}'
    )

  requirements = ''
  with open(requirements_txt_path, 'r', encoding='utf-8') as f:
    requirements = f.read()

  for line in requirements.splitlines():
    stripped = line.strip()
    if (
        stripped
        and not stripped.startswith('#')
        and stripped.startswith('google-cloud-aiplatform')
    ):
      return

  with open(requirements_txt_path, 'a', encoding='utf-8') as f:
    if requirements and not requirements.endswith('\n'):
      f.write('\n')
    f.write(f'{_AGENT_ENGINE_REQUIREMENT}\n')
    f.write(f'google-adk[a2a]=={__version__}\n')


_DOCKERFILE_TEMPLATE: Final[str] = """
FROM python:3.11-slim
WORKDIR /app

# Create a non-root user
RUN adduser --disabled-password --gecos "" myuser

# Switch to the non-root user
USER myuser

# Set up environment variables - Start
ENV PATH="/home/myuser/.local/bin:$PATH"

ENV GOOGLE_GENAI_USE_ENTERPRISE=1
ENV GOOGLE_CLOUD_PROJECT={gcp_project_id}
ENV GOOGLE_CLOUD_LOCATION={gcp_region}

# Set up environment variables - End

# Install ADK - Start
RUN pip install "google-adk[a2a]=={adk_version}"
# Install ADK - End

# Copy agent - Start

# Set permission
COPY --chown=myuser:myuser "agents/{app_name}/" "/app/agents/{app_name}/"

# Copy agent - End

# Install Agent Deps - Start
{install_agent_deps}
# Install Agent Deps - End

EXPOSE {port}

CMD adk {command} --port={port} {host_option} {service_option} {trace_to_cloud_option} {otel_to_cloud_option} {allow_origins_option} {a2a_option} {trigger_sources_option} {gemini_enterprise_option}{express_mode_option} "/app/agents"
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
                'ttl': {'type': 'string', 'nullable': True},
                'expire_time': {'type': 'string', 'nullable': True},
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
            ' session. If not provided, an ID\n                will be'
            ' generated for the session.\n            state (dict[str, Any]):\n'
            '                Optional. The initial state of the session.\n     '
            '       ttl (str):\n                Optional. The time-to-live for'
            ' the session.\n            expire_time (str):\n               '
            ' Optional. The expiration time for the session.\n           '
            ' **kwargs (dict[str, Any]):\n                Optional. Additional'
            ' keyword arguments to pass to the\n                session'
            ' service.\n\n        Returns:\n            Session: The newly'
            ' created session instance.\n        '
        ),
        'parameters': {
            'properties': {
                'user_id': {'type': 'string'},
                'session_id': {'type': 'string', 'nullable': True},
                'state': {'type': 'object', 'nullable': True},
                'ttl': {'type': 'string', 'nullable': True},
                'expire_time': {'type': 'string', 'nullable': True},
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


def _resolve_adk_version() -> str:
  """Returns the default ADK version."""
  from google.adk.version import __version__

  return __version__


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


def _validate_agent_import(
    agent_src_path: str,
    adk_app_object: str,
    is_config_agent: bool,
) -> None:
  """Validates that the agent module can be imported successfully.

  This pre-deployment validation catches common issues like missing
  dependencies or import errors in custom BaseLlm implementations before
  the agent is deployed to Agent Engine. This provides clearer error
  messages and prevents deployments that would fail at runtime.

  Args:
    agent_src_path: Path to the staged agent source code.
    adk_app_object: The Python object name to import ('root_agent' or 'app').
    is_config_agent: Whether this is a config-based agent.

  Raises:
    click.ClickException: If the agent module cannot be imported.
  """
  if is_config_agent:
    # Config agents are loaded from YAML, skip Python import validation
    return

  agent_module_path = os.path.join(agent_src_path, 'agent.py')
  if not os.path.exists(agent_module_path):
    raise click.ClickException(
        f'Agent module not found at {agent_module_path}. '
        'Please ensure your agent folder contains an agent.py file.'
    )

  # Add the parent directory to sys.path temporarily for import resolution
  parent_dir = os.path.dirname(agent_src_path)
  module_name = os.path.basename(agent_src_path)

  original_sys_path = sys.path.copy()
  original_sys_modules_keys = set(sys.modules.keys())
  try:
    # Add parent directory to path so imports work correctly
    if parent_dir not in sys.path:
      sys.path.insert(0, parent_dir)
    try:
      module = importlib.import_module(f'{module_name}.agent')
    except ImportError as e:
      error_msg = str(e)
      tb = traceback.format_exc()

      # Check for common issues
      if 'BaseLlm' in tb or 'base_llm' in tb.lower():
        raise click.ClickException(
            'Failed to import agent module due to a BaseLlm-related error:\n'
            f'{error_msg}\n\n'
            'This error often occurs when deploying agents with custom LLM '
            'implementations. Please ensure:\n'
            '1. All custom LLM classes are defined in files within your agent '
            'folder\n'
            '2. All required dependencies are listed in requirements.txt\n'
            '3. Import paths use relative imports (e.g., "from .my_llm import '
            'MyLlm")\n'
            '4. Your custom BaseLlm class and its dependencies are installed\n'
            '\n'
            'If this failure is expected (e.g., missing local dependencies), '
            'disable agent import validation by omitting '
            '--validate-agent-import (default) or passing '
            '--skip-agent-import-validation (or --no-validate-agent-import).'
        ) from e
      else:
        raise click.ClickException(
            f'Failed to import agent module:\n{error_msg}\n\n'
            'Please ensure all dependencies are listed in requirements.txt '
            'and all imports are resolvable.\n\n'
            f'Full traceback:\n{tb}\n\n'
            'If this failure is expected (e.g., missing local dependencies), '
            'disable agent import validation by omitting '
            '--validate-agent-import (default) or passing '
            '--skip-agent-import-validation (or --no-validate-agent-import).'
        ) from e
    except Exception as e:
      tb = traceback.format_exc()
      raise click.ClickException(
          f'Error while loading agent module:\n{e}\n\n'
          'Please check your agent code for errors.\n\n'
          f'Full traceback:\n{tb}\n\n'
          'If this failure is expected (e.g., missing local dependencies), '
          'disable agent import validation by omitting '
          '--validate-agent-import (default) or passing '
          '--skip-agent-import-validation (or --no-validate-agent-import).'
      ) from e

    # Check that the expected object exists
    if not hasattr(module, adk_app_object):
      available_attrs = [
          attr for attr in dir(module) if not attr.startswith('_')
      ]
      raise click.ClickException(
          f"Agent module does not export '{adk_app_object}'. "
          f'Available exports: {available_attrs}\n\n'
          'Please ensure your agent.py exports either "root_agent" or "app".'
      )

    click.echo(
        'Agent module validation successful: '
        f'found "{adk_app_object}" in agent.py'
    )

  finally:
    # Restore original sys.path
    sys.path[:] = original_sys_path
    # Clean up modules introduced by validation.
    for key in list(sys.modules.keys()):
      if key in original_sys_modules_keys:
        continue
      if key == module_name or key.startswith(f'{module_name}.'):
        sys.modules.pop(key, None)


def _get_service_option_by_adk_version(
    adk_version: str,
    session_uri: Optional[str],
    artifact_uri: Optional[str],
    memory_uri: Optional[str],
    use_local_storage: Optional[bool] = None,
) -> str:
  """Returns service option string based on adk_version."""
  parsed_version = parse(adk_version)
  options: list[str] = []

  if session_uri:
    options.append(f'--session_service_uri={session_uri}')
  if artifact_uri:
    options.append(f'--artifact_service_uri={artifact_uri}')
  if memory_uri:
    options.append(f'--memory_service_uri={memory_uri}')

  if use_local_storage is not None and parsed_version >= parse(
      _LOCAL_STORAGE_FLAG_MIN_VERSION
  ):
    # Only valid when session/artifact URIs are unset; otherwise the CLI
    # rejects the combination to avoid confusing precedence.
    if session_uri is None and artifact_uri is None:
      options.append((
          '--use_local_storage'
          if use_local_storage
          else '--no_use_local_storage'
      ))

  return ' '.join(options)


def _get_ignore_patterns_func(agent_folder: str):
  """Returns a shutil.ignore_patterns function with combined patterns from .gitignore, .gcloudignore and .ae_ignore."""
  patterns = set()

  for filename in ['.gitignore', '.gcloudignore', '.ae_ignore']:
    filepath = os.path.join(agent_folder, filename)
    if os.path.exists(filepath):
      click.echo(f'Reading ignore patterns from {filename}...')
      try:
        with open(filepath, 'r') as f:
          for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
              # If it ends with /, remove it for fnmatch compatibility
              if line.endswith('/'):
                line = line[:-1]
              # Strip leading / from root-anchored patterns; shutil.ignore_patterns
              # matches basenames via fnmatch, so '/venv' would match nothing.
              if line.startswith('/'):
                line = line[1:]
              if line:
                patterns.add(line)
      except Exception as e:
        click.secho(f'Warning: Failed to read {filename}: {e}', fg='yellow')

  return shutil.ignore_patterns(*patterns)


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
    otel_to_cloud: bool,
    with_ui: bool,
    log_level: str,
    verbosity: str,
    adk_version: str,
    allow_origins: Optional[list[str]] = None,
    session_service_uri: Optional[str] = None,
    artifact_service_uri: Optional[str] = None,
    memory_service_uri: Optional[str] = None,
    use_local_storage: bool = False,
    a2a: bool = False,
    trigger_sources: Optional[str] = None,
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
    otel_to_cloud: Whether to enable exporting OpenTelemetry signals
      to Google Cloud.
    with_ui: Whether to deploy with UI.
    verbosity: The verbosity level of the CLI.
    adk_version: The ADK version to use in Cloud Run.
    allow_origins: Origins to allow for CORS. Can be literal origins or regex
      patterns prefixed with 'regex:'.
    session_service_uri: The URI of the session service.
    artifact_service_uri: The URI of the artifact service.
    memory_service_uri: The URI of the memory service.
    use_local_storage: Whether to use local .adk storage in the container.
  """
  app_name = app_name or os.path.basename(agent_folder)
  if parse(adk_version) >= parse('1.3.0') and not use_local_storage:
    session_service_uri = session_service_uri or 'memory://'
    artifact_service_uri = artifact_service_uri or 'memory://'

  click.echo(f'Start generating Cloud Run source files in {temp_folder}')

  # remove temp_folder if exists
  if os.path.exists(temp_folder):
    click.echo('Removing existing files')
    shutil.rmtree(temp_folder)

  try:
    # copy agent source code
    click.echo('Copying agent source code...')
    agent_src_path = os.path.join(temp_folder, 'agents', app_name)
    ignore_func = _get_ignore_patterns_func(agent_folder)
    shutil.copytree(agent_folder, agent_src_path, ignore=ignore_func)
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
    trigger_sources_option = (
        f'--trigger_sources={trigger_sources}' if trigger_sources else ''
    )
    dockerfile_content = _DOCKERFILE_TEMPLATE.format(
        gcp_project_id=project,
        gcp_region=region,
        app_name=app_name,
        port=port,
        command='api_server --with_ui' if with_ui else 'api_server',
        install_agent_deps=install_agent_deps,
        service_option=_get_service_option_by_adk_version(
            adk_version,
            session_service_uri,
            artifact_service_uri,
            memory_service_uri,
            use_local_storage,
        ),
        trace_to_cloud_option='--trace_to_cloud' if trace_to_cloud else '',
        otel_to_cloud_option='--otel_to_cloud' if otel_to_cloud else '',
        allow_origins_option=allow_origins_option,
        adk_version=adk_version,
        host_option=host_option,
        a2a_option=a2a_option,
        trigger_sources_option=trigger_sources_option,
        gemini_enterprise_option='',
        express_mode_option='',
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


def _print_agent_engine_url(resource_name: str) -> None:
  """Prints the Google Cloud Console URL for the deployed agent."""
  parts = resource_name.split('/')
  if len(parts) >= 6 and parts[0] == 'projects' and parts[2] == 'locations':
    project_id = parts[1]
    region = parts[3]
    engine_id = parts[5]

    url = (
        'https://console.cloud.google.com/vertex-ai/agents/agent-engines'
        f'/locations/{region}/agent-engines/{engine_id}/playground'
        f'?project={project_id}'
    )
    click.secho(
        f'\n🎉 View your deployed agent here:\n{url}\n', fg='cyan', bold=True
    )


def to_agent_engine(
    *,
    agent_folder: str,
    temp_folder: Optional[str] = None,
    adk_app: Optional[str] = None,
    staging_bucket: Optional[str] = None,
    trace_to_cloud: Optional[bool] = None,
    otel_to_cloud: Optional[bool] = None,
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
    skip_agent_import_validation: bool = True,
    trigger_sources: Optional[str] = None,
    memory_service_uri: Optional[str] = None,
    session_service_uri: Optional[str] = None,
    artifact_service_uri: Optional[str] = None,
    adk_version: Optional[str] = None,
):
  """Deploys an agent to Gemini Enterprise Agent Platform.

  `agent_folder` should contain the following files:

  - __init__.py
  - agent.py
  - requirements.txt (optional, for additional dependencies)
  - .env (optional, for environment variables)
  - ... (other required source files)

  Args:
    agent_folder (str): The folder (absolute path) containing the agent source
      code.
    temp_folder (str): The temp folder for the generated Agent Platform source
      files. It will be replaced with the generated files if it already exists.
    adk_app (str): Deprecated. This argument is no longer required or used.
    staging_bucket (str): Deprecated. This argument is no longer required or
      used.
    trace_to_cloud (bool): Deprecated. This argument is no longer required or
      used.
    otel_to_cloud (bool): Whether to enable exporting OpenTelemetry signals to
      Google Cloud.
    api_key (str): Optional. The API key to use for Express Mode. If not
      provided, the API key from the GOOGLE_API_KEY environment variable will be
      used. It will only be used if GOOGLE_GENAI_USE_ENTERPRISE is true.
    adk_app_object (str): Deprecated. This argument is no longer required or
      used.
    agent_engine_id (str): Optional. The ID of the Agent Runtime instance to
      update. If not specified, a new Agent Runtime instance will be created.
    absolutize_imports (bool): Deprecated. This argument is no longer required
      or used.
    project (str): Optional. Google Cloud project id for the deployed agent. If
      not specified, the project from the `GOOGLE_CLOUD_PROJECT` environment
      variable will be used. It will be ignored if `api_key` is specified.
    region (str): Optional. Google Cloud region for the deployed agent. If not
      specified, the region from the `GOOGLE_CLOUD_LOCATION` environment
      variable will be used. It will be ignored if `api_key` is specified.
    display_name (str): Optional. The display name of the Agent Runtime.
    description (str): Optional. The description of the Agent Runtime.
    requirements_file (str): Deprecated. This argument is no longer required or
      used.
    env_file (str): Optional. The filepath to the `.env` file for environment
      variables. If not specified, the `.env` file in the `agent_folder` will be
      used. The values of `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION`
      will be overridden by `project` and `region` if they are specified.
    agent_engine_config_file (str): The filepath to the agent platform config
      file to use. If not specified, the `.agent_engine_config.json` file in the
      `agent_folder` will be used.
    skip_agent_import_validation (bool): Deprecated. This argument is no longer
      required or used.
    trigger_sources (str): Optional. Comma-separated list of trigger sources to
      enable (e.g., 'pubsub,eventarc'). Registers /trigger/* endpoints for batch
      and event-driven agent invocations.
    memory_service_uri (str): Optional. The URI of the memory service. If not
      specified, the memory service will be deployed to the same parent resource
      as the runtime.
    session_service_uri (str): Optional. The URI of the session service. If not
      specified, the session service will be deployed to the same parent
      resource as the runtime.
    artifact_service_uri (str): Optional. The URI of the artifact service.
    adk_version (str): Optional. The ADK version to use in Agent Platform
      deployment. If not specified, the version in the dev environment will be
      used.
  """
  app_name = os.path.basename(agent_folder)
  display_name = display_name or app_name
  parent_folder = os.path.dirname(agent_folder)
  if adk_app_object:
    warnings.warn(
        'WARNING: `--adk_app_object` is deprecated and will be removed in the'
        ' future. Please drop it from the list of arguments.',
        DeprecationWarning,
        stacklevel=2,
    )
  if adk_app:
    warnings.warn(
        'WARNING: `adk_app` is deprecated and will be removed in a future'
        ' release. Please drop it from the list of arguments.',
        DeprecationWarning,
        stacklevel=2,
    )
  if staging_bucket:
    warnings.warn(
        'WARNING: `staging_bucket` is deprecated and will be removed in a'
        ' future release. Please drop it from the list of arguments.',
        DeprecationWarning,
        stacklevel=2,
    )
  if not adk_version:
    adk_version = _resolve_adk_version()
    click.echo(f'Using default ADK version: {adk_version}')

  original_cwd = os.getcwd()
  did_change_cwd = False
  if parent_folder != original_cwd:
    click.echo(
        'Agent Runtime deployment uses relative paths; temporarily switching '
        f'working directory to: {parent_folder}'
    )
    os.chdir(parent_folder)
    did_change_cwd = True
  tmp_app_name = app_name + '_tmp' + datetime.now().strftime('%Y%m%d_%H%M%S')
  temp_folder = temp_folder or tmp_app_name
  agent_src_path = os.path.join(parent_folder, temp_folder, 'agents', app_name)
  temp_folder_path = os.path.join(parent_folder, temp_folder)
  if os.path.exists(temp_folder_path):
    click.echo('Removing existing files')
    shutil.rmtree(temp_folder_path)

  try:
    ignore_func = _get_ignore_patterns_func(agent_folder)
    click.echo('Copying agent source code...')
    shutil.copytree(
        agent_folder,
        agent_src_path,
        ignore=ignore_func,
        dirs_exist_ok=True,
    )
    os.chdir(temp_folder_path)
    click.echo('Copying agent source code complete.')

    project = _resolve_project(project)

    click.echo('Resolving files and dependencies...')
    agent_config = {}
    if agent_engine_config_file and not os.path.exists(
        agent_engine_config_file
    ):
      raise click.ClickException(
          'Agent Platform config file not found: '
          f'{parent_folder}/{agent_engine_config_file}'
      )
    if not agent_engine_config_file:
      # Attempt to read the agent platform config from .agent_engine_config.json
      # in the dir (if any).
      agent_engine_config_file = os.path.join(
          agent_folder, '.agent_engine_config.json'
      )
    if os.path.exists(agent_engine_config_file):
      click.echo(
          f'Reading agent platform config from {agent_engine_config_file}'
      )
      with open(agent_engine_config_file, 'r') as f:
        agent_config = json.load(f)
    if display_name:
      if 'display_name' in agent_config:
        click.echo(
            'Overriding display_name in agent platform config with'
            f' {display_name}'
        )
      agent_config['display_name'] = display_name
    if description:
      if 'description' in agent_config:
        click.echo(
            'Overriding description in agent platform config with'
            f' {description}'
        )
      agent_config['description'] = description

    requirements_txt_path = os.path.join(agent_src_path, 'requirements.txt')
    if requirements_file:
      warnings.warn(
          'WARNING: `--requirements_file` is deprecated and will be removed in'
          ' the future. Please define `requirements.txt` in the agent folder.',
          DeprecationWarning,
          stacklevel=2,
      )
    if trace_to_cloud:
      warnings.warn(
          'WARNING: `--trace_to_cloud` is deprecated and will be removed in the'
          ' future. Please use `--otel_to_cloud` instead.',
          DeprecationWarning,
          stacklevel=2,
      )
    if not os.path.exists(requirements_txt_path):
      click.echo(f'Creating {requirements_txt_path}...')
      with open(requirements_txt_path, 'w', encoding='utf-8') as f:
        f.write(f'{_AGENT_ENGINE_REQUIREMENT}\n')
        f.write(f'google-adk[a2a]=={__version__}\n')
        click.echo(f'Using google-adk[a2a]=={__version__} in requirements')
      click.echo(f'Created {requirements_txt_path}')
    _ensure_agent_engine_dependency(requirements_txt_path)

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
        env_region = env_vars.get('GOOGLE_CLOUD_LOCATION')
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
        env_vars['GOOGLE_GENAI_USE_ENTERPRISE'] = '1'
        env_vars['GOOGLE_API_KEY'] = api_key
    elif not project:
      if 'GOOGLE_API_KEY' in env_vars:
        api_key = env_vars['GOOGLE_API_KEY']
        click.echo(f'api_key set by GOOGLE_API_KEY in {env_file}')
    if otel_to_cloud:
      if 'GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY' in env_vars:
        click.secho(
            'Ignoring GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY in .env'
            ' as `--otel_to_cloud` was explicitly passed and takes precedence',
            fg='yellow',
        )
      env_vars['GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY'] = 'true'
      if 'ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS' not in env_vars:
        env_vars['ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS'] = 'false'
    else:
      enable_telemetry = env_vars.get(
          'GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY',
      )
      if enable_telemetry in ['true', '1']:
        otel_to_cloud = True
        click.echo(
            '`--otel_to_cloud` is set to True by'
            f' GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY in {env_file}'
        )
    if env_vars:
      if 'env_vars' in agent_config:
        click.echo(
            f'Overriding env_vars in agent platform config with {env_vars}'
        )
      agent_config['env_vars'] = env_vars
    # Set env_vars in agent_config to None if it is not set.
    agent_config['env_vars'] = agent_config.get('env_vars', env_vars)

    import vertexai

    from ..utils._google_client_headers import get_tracking_headers

    if not (api_key or project or region):
      click.echo(
          'No api_key/project/region provided. Starting onboarding flow...'
      )
      auth_info = _onboarding.handle_login_with_google()
      project = auth_info.project_id
      region = auth_info.region

    click.echo('Initializing Agent Platform client...')
    if project and region:
      client = vertexai.Client(
          project=project,
          location=region,
          http_options={'headers': get_tracking_headers()},
      )
      click.echo('Agent Platform client initialized with project and region.')
    elif api_key:
      client = vertexai.Client(
          api_key=api_key,
          http_options={'headers': get_tracking_headers()},
      )
      click.echo('Agent Platform client initialized with ExpressMode API Key.')
    else:
      click.echo(
          'Failed to initialize Agent Platform client. Please provide an API'
          'key or project and region.'
      )
      return

    if skip_agent_import_validation:
      warnings.warn(
          'WARNING: `--skip-agent-import-validation` is deprecated and will be'
          ' removed in the future. Please drop it from the list of arguments.',
          DeprecationWarning,
          stacklevel=2,
      )

    def create_dockerfile_for_agent_engine(resource_name: str):
      requirements_txt_path = os.path.join(agent_src_path, 'requirements.txt')
      install_agent_deps = (
          f'RUN pip install -r "/app/agents/{app_name}/requirements.txt"'
          if os.path.exists(requirements_txt_path)
          else '# No requirements.txt found.'
      )
      trigger_sources_option = (
          f'--trigger_sources={trigger_sources}' if trigger_sources else ''
      )
      agent_engine_uri = f'agentengine://{resource_name}'
      dockerfile_content = _DOCKERFILE_TEMPLATE.format(
          gcp_project_id=project,
          gcp_region=region,
          app_name=app_name,
          port=8080,
          command='api_server',
          install_agent_deps=install_agent_deps,
          service_option=_get_service_option_by_adk_version(
              adk_version,
              session_service_uri or agent_engine_uri,
              artifact_service_uri,
              memory_service_uri or agent_engine_uri,
              False,  # use_local_storage
          ),
          trace_to_cloud_option='--trace_to_cloud' if trace_to_cloud else '',
          otel_to_cloud_option='--otel_to_cloud' if otel_to_cloud else '',
          allow_origins_option='',  # Not supported for now.
          adk_version=adk_version,
          host_option='--host=0.0.0.0',
          a2a_option='--a2a',
          trigger_sources_option=trigger_sources_option,
          gemini_enterprise_option=f'--gemini_enterprise_app_name={app_name}',
          express_mode_option=(
              ' --express_mode' if api_key and not project else ''
          ),
      )
      with open('Dockerfile', 'w', encoding='utf-8') as f:
        f.write(dockerfile_content)

    if absolutize_imports:
      warnings.warn(
          'WARNING: `--absolutize_imports` is deprecated and will be removed'
          ' in the future. Please drop it from the list of arguments.',
          DeprecationWarning,
          stacklevel=2,
      )
    click.echo('Deploying to Agent Platform...')
    agent_config['source_packages'] = [f'agents/{app_name}', 'Dockerfile']
    agent_config['image_spec'] = {}  # Use the Dockerfile
    agent_config['class_methods'] = _AGENT_ENGINE_CLASS_METHODS
    agent_config['agent_framework'] = 'google-adk'

    resource_name = agent_engine_id
    if not resource_name:
      agent_engine = client.agent_engines.create()
      resource_name = agent_engine.api_resource.name
      click.secho(f'Created a new instance: {resource_name}', fg='green')
    elif project and region and not resource_name.startswith('projects/'):
      resource_name = f'projects/{project}/locations/{region}/reasoningEngines/{agent_engine_id}'
    click.echo('Creating Dockerfile...')
    create_dockerfile_for_agent_engine(resource_name)
    click.echo(f'Dockerfile created at {os.getcwd()}/Dockerfile.')
    try:
      client.agent_engines.update(name=resource_name, config=agent_config)
      click.secho(f'Deployed to Agent Platform: {resource_name}', fg='green')
    except Exception as e:
      click.secho(f'Failed to deploy to Agent Platform: {e}', fg='red')
      # Only delete the instance if it was newly created in this function.
      if agent_engine_id is None:
        client.agent_engines.delete(name=resource_name)
        click.secho(f'Cleaned up the instance: {resource_name}', fg='green')
      raise e
    _print_agent_engine_url(resource_name)
  finally:
    temp_folder_path = os.path.join(parent_folder, temp_folder)
    click.echo(f'Cleaning up the temp folder: {temp_folder_path}')
    os.chdir(original_cwd)
    shutil.rmtree(temp_folder_path)


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
    otel_to_cloud: bool,
    with_ui: bool,
    log_level: str,
    adk_version: str,
    allow_origins: Optional[list[str]] = None,
    session_service_uri: Optional[str] = None,
    artifact_service_uri: Optional[str] = None,
    memory_service_uri: Optional[str] = None,
    use_local_storage: bool = False,
    a2a: bool = False,
    trigger_sources: Optional[str] = None,
    service_type: Literal[
        'ClusterIP', 'NodePort', 'LoadBalancer'
    ] = 'ClusterIP',
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
    otel_to_cloud: Whether to enable exporting OpenTelemetry signals
      to Google Cloud.
    with_ui: Whether to deploy with UI.
    log_level: The logging level.
    adk_version: The ADK version to use in GKE.
    allow_origins: Origins to allow for CORS. Can be literal origins or regex
      patterns prefixed with 'regex:'.
    session_service_uri: The URI of the session service.
    artifact_service_uri: The URI of the artifact service.
    memory_service_uri: The URI of the memory service.
    use_local_storage: Whether to use local .adk storage in the container.
    service_type: The Kubernetes Service type (default: ClusterIP).
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
  if parse(adk_version) >= parse('1.3.0') and not use_local_storage:
    session_service_uri = session_service_uri or 'memory://'
    artifact_service_uri = artifact_service_uri or 'memory://'

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
    ignore_func = _get_ignore_patterns_func(agent_folder)
    shutil.copytree(agent_folder, agent_src_path, ignore=ignore_func)
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
        command='api_server --with_ui' if with_ui else 'api_server',
        install_agent_deps=install_agent_deps,
        service_option=_get_service_option_by_adk_version(
            adk_version,
            session_service_uri,
            artifact_service_uri,
            memory_service_uri,
            use_local_storage,
        ),
        trace_to_cloud_option='--trace_to_cloud' if trace_to_cloud else '',
        otel_to_cloud_option='--otel_to_cloud' if otel_to_cloud else '',
        allow_origins_option=allow_origins_option,
        adk_version=adk_version,
        host_option=host_option,
        a2a_option='--a2a' if a2a else '',
        trigger_sources_option=(
            f'--trigger_sources={trigger_sources}' if trigger_sources else ''
        ),
        gemini_enterprise_option='',
        express_mode_option='',
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
  type: {service_type}
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
  if service_type == 'ClusterIP':
    click.echo(
        '\nThe service is only reachable from within the cluster.'
        ' To access it locally, run:'
        f'\n  kubectl port-forward svc/{service_name} {port}:{port}'
        '\n\nTo expose the service externally, add a Gateway or'
        ' re-deploy with --service_type=LoadBalancer.'
    )
