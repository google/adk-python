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

from collections.abc import Sequence
from datetime import datetime
import importlib
import json
import logging
import os
import re
import shutil
import stat
import subprocess
import sys
import traceback
from typing import Any
from typing import Callable
from typing import Final
from typing import Literal
from typing import Optional
import warnings

import click
from packaging.requirements import InvalidRequirement
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.utils import NormalizedName
from packaging.version import parse

from ..version import __version__
from .deployers import DeployerFactory
from .deployers._dockerfile_template import _render_dockerfile
from .deployers._dockerfile_template import _render_install_agent_deps
from .deployers._dockerfile_template import _validate_app_name
from .utils import _onboarding

logger = logging.getLogger('google_adk.' + __name__)

_IS_WINDOWS = os.name == 'nt'
_GCLOUD_CMD = 'gcloud.cmd' if _IS_WINDOWS else 'gcloud'
_LOCAL_STORAGE_FLAG_MIN_VERSION: Final[str] = '1.21.0'
_GEMINI_ENTERPRISE_FLAG_MIN_VERSION: Final[str] = '2.2.0'
# The deployed image runs `adk api_server`, which imports `agentplatform`, so
# the staged requirements must resolve to the v2 SDK and not to a v1 release
# that only ships the legacy `vertexai` surface.
_AGENT_ENGINE_REQUIREMENT: Final[str] = (
    'google-cloud-aiplatform[adk,agent_engines]>=2.2,<3'
)
_AGENT_ENGINE_MIN_VERSION: Final[str] = '2.2'
# Either distribution provides the top-level `agentplatform` package: the
# bundled google-cloud-aiplatform, and the standalone package it was split
# into. An agent may legitimately pin either one. Held in canonical form,
# because `Requirement.name` preserves whatever spelling the agent wrote and
# `google_cloud_aiplatform` is the same distribution as `google-cloud-aiplatform`.
_AGENT_PLATFORM_DISTRIBUTIONS: Final[frozenset[str]] = frozenset({
    canonicalize_name('google-cloud-aiplatform'),
    canonicalize_name('google-cloud-agentplatform'),
})
# Full Cloud Build private worker pool resource name, e.g.
# projects/my-project/locations/us-central1/workerPools/my-private-pool
_WORKER_POOL_RESOURCE_RE: Final[re.Pattern[str]] = re.compile(
    r'^projects/[^/]+/locations/[^/]+/workerPools/[^/]+$'
)


def _validate_trigger_options(
    trigger_sources: str | None,
    trigger_oidc_audience: str | None,
    trigger_oidc_service_accounts: str | None,
) -> None:
  if trigger_sources and not trigger_oidc_audience:
    raise click.UsageError(
        '--trigger_oidc_audience is required when --trigger_sources is set'
    )
  if trigger_oidc_service_accounts and not trigger_oidc_audience:
    raise click.UsageError(
        '--trigger_oidc_service_accounts requires --trigger_oidc_audience to'
        ' be set'
    )
  if (
      trigger_sources
      and trigger_oidc_audience
      and not trigger_oidc_service_accounts
  ):
    logger.warning(
        '--trigger_oidc_audience is set without'
        ' --trigger_oidc_service_accounts; any Google account can obtain a'
        ' token for this audience. Set --trigger_oidc_service_accounts to'
        ' restrict caller identity.'
    )


def _validate_worker_pool(worker_pool: str) -> str:
  """Validates a Cloud Build worker pool resource name.

  Args:
    worker_pool: Full resource name of the form
      `projects/{project}/locations/{location}/workerPools/{pool}`.

  Returns:
    The validated worker pool resource name.

  Raises:
    click.ClickException: If the resource name is empty or malformed.
  """
  worker_pool = worker_pool.strip()
  if not worker_pool:
    raise click.ClickException('worker_pool must be a non-empty resource name.')
  if not _WORKER_POOL_RESOURCE_RE.fullmatch(worker_pool):
    raise click.ClickException(
        'Invalid worker_pool resource name. Expected format:'
        ' projects/{project}/locations/{location}/workerPools/{pool}.'
        f' Got: {worker_pool}'
    )
  return worker_pool


def _apply_worker_pool_to_agent_config(
    agent_config: dict[str, Any],
    worker_pool: Optional[str],
) -> None:
  """Nests worker_pool into agent_config['build_config'].

  Supports three sources, in increasing precedence:

  1. Existing ``build_config.worker_pool`` already in ``agent_config``.
  2. Top-level ``worker_pool`` convenience key in ``.agent_engine_config.json``
     (popped so it is not forwarded as an unknown top-level field).
  3. Explicit ``worker_pool`` argument (CLI flag), which overrides both.

  The Vertex Agent Engine SDK reads Cloud Build private pools from
  ``config.build_config.worker_pool`` and maps them onto
  ``spec.build_spec.worker_pool``.
  """
  build_config = agent_config.get('build_config')
  if build_config is None:
    build_config = {}
  elif not isinstance(build_config, dict):
    raise click.ClickException(
        'build_config in agent platform config must be a JSON object.'
    )
  else:
    # Copy so we do not mutate a shared structure unexpectedly.
    build_config = dict(build_config)

  config_worker_pool = agent_config.pop('worker_pool', None)
  if config_worker_pool is not None:
    if not isinstance(config_worker_pool, str):
      raise click.ClickException(
          'worker_pool in agent platform config must be a string resource name.'
      )
    build_config['worker_pool'] = _validate_worker_pool(config_worker_pool)

  if worker_pool is not None:
    if build_config.get('worker_pool'):
      click.echo(
          'Overriding build_config.worker_pool in agent platform config with'
          f' {worker_pool}'
      )
    build_config['worker_pool'] = _validate_worker_pool(worker_pool)

  # Validate any worker_pool that was already nested under build_config.
  if 'worker_pool' in build_config and build_config['worker_pool'] is not None:
    build_config['worker_pool'] = _validate_worker_pool(
        str(build_config['worker_pool'])
    )

  if build_config:
    agent_config['build_config'] = build_config
  else:
    agent_config.pop('build_config', None)


# Runtime service account email for Agent Engine, e.g.
# my-agent@my-project.iam.gserviceaccount.com
_SERVICE_ACCOUNT_EMAIL_RE: Final[re.Pattern[str]] = re.compile(
    r'^[a-zA-Z0-9._+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
)


def _validate_service_account(service_account: str) -> str:
  """Validates an Agent Engine runtime service account email.

  Args:
    service_account: Google Cloud service account email.

  Returns:
    The validated service account email.

  Raises:
    click.ClickException: If the email is empty or malformed.
  """
  service_account = service_account.strip()
  if not service_account:
    raise click.ClickException(
        'service_account must be a non-empty service account email.'
    )
  if not _SERVICE_ACCOUNT_EMAIL_RE.fullmatch(service_account):
    raise click.ClickException(
        'Invalid service_account email. Expected a Google Cloud service'
        ' account email such as'
        ' my-agent@my-project.iam.gserviceaccount.com.'
        f' Got: {service_account}'
    )
  return service_account


def _apply_service_account_to_agent_config(
    agent_config: dict[str, Any],
    service_account: Optional[str],
) -> None:
  """Sets top-level ``service_account`` on the Agent Engine update config.

  Precedence (highest last):

  1. Existing ``service_account`` in ``.agent_engine_config.json``.
  2. Explicit ``service_account`` argument (CLI / resolved from
     ``GOOGLE_CLOUD_SERVICE_ACCOUNT``), which overrides the config file.

  The Vertex Agent Engine SDK maps ``config.service_account`` onto
  ``spec.service_account`` (runtime identity). This is distinct from
  ``build_config.service_account`` (Cloud Build identity).
  """
  if service_account is not None:
    validated = _validate_service_account(service_account)
    existing = agent_config.get('service_account')
    if existing and existing != validated:
      click.echo(
          'Overriding service_account in agent platform config with'
          f' {validated}'
      )
    agent_config['service_account'] = validated
    return

  existing = agent_config.get('service_account')
  if existing is None:
    return
  if not isinstance(existing, str):
    raise click.ClickException(
        'service_account in agent platform config must be a string email.'
    )
  agent_config['service_account'] = _validate_service_account(existing)


def _on_rm_error(func: Callable[..., Any], path: str, exc_info: Any) -> None:
  """Error handler for shutil.rmtree to handle read-only files on Windows."""
  os.chmod(path, stat.S_IWRITE)
  func(path)


def _robust_rmtree(path: str) -> None:
  """Remove a directory tree, handling read-only files on Windows."""
  if _IS_WINDOWS:
    if sys.version_info >= (3, 12):
      shutil.rmtree(path, onexc=lambda fn, p, exc: _on_rm_error(fn, p, None))
    else:
      shutil.rmtree(path, onerror=_on_rm_error)
  else:
    shutil.rmtree(path)


def _ensure_agent_engine_dependency(requirements_txt_path: str) -> None:
  """Ensures staged requirements include Agent Platform dependencies."""
  if not os.path.exists(requirements_txt_path):
    raise FileNotFoundError(
        f'requirements.txt not found at: {requirements_txt_path}'
    )

  requirements = ''
  with open(requirements_txt_path, 'r', encoding='utf-8') as f:
    requirements = f.read()

  # Canonical names, in first-seen order, so the appended floors are stable and
  # each distribution gets exactly one however many lines name it.
  pinned_distributions: dict[NormalizedName, None] = {}
  hash_checking = False
  # A backslash at end of line continues the requirement onto the next one,
  # which is how `uv export` and `pip-compile --generate-hashes` lay out each
  # pin and its `--hash` options.
  for line in re.sub(r'\\\r?\n', ' ', requirements).splitlines():
    # A requirements file comment runs from an unquoted `#` to end of line, and
    # `Requirement()` rejects one left in place, so a pin carrying a trailing
    # comment would otherwise look like no pin at all. The `#` has to be
    # preceded by whitespace or start the line, so that the fragment in a
    # direct URL reference survives.
    stripped = re.split(r'(?:^|\s)#', line, maxsplit=1)[0].strip()
    if not stripped:
      continue
    # pip checks hashes for the whole file once any requirement carries
    # `--hash`, or when `--require-hashes` is given.
    if '--hash' in stripped or stripped.startswith('--require-hashes'):
      hash_checking = True
    # Per-requirement options such as `--hash` follow the specifier, and
    # `Requirement()` rejects them, so parse only what comes before.
    specifier = re.split(r'\s+--', stripped, maxsplit=1)[0]
    if specifier.startswith('-'):
      continue
    try:
      existing = Requirement(specifier)
    except InvalidRequirement:
      continue
    name = canonicalize_name(existing.name)
    if name in _AGENT_PLATFORM_DISTRIBUTIONS:
      pinned_distributions[name] = None

  if pinned_distributions and hash_checking:
    # A hash-locked file already pins each distribution to one exact, hashed
    # release, and in hash-checking mode pip rejects any requirement without a
    # hash, so appending a floor would fail the image build outright. Leave the
    # agent's lock as written, as this function did before it stated floors.
    return

  with open(requirements_txt_path, 'a', encoding='utf-8') as f:
    if requirements and not requirements.endswith('\n'):
      f.write('\n')
    if pinned_distributions:
      # The agent already asks for an Agent Platform distribution. Rather than
      # judging whether its specifier can reach the v2 surface, state the floor
      # as a second requirement on the same distribution and let pip reconcile
      # the two. pip is the authority on what a specifier set allows, so a pin
      # that cannot reach the floor fails the image build with a resolver error
      # naming both lines, instead of this function having to reimplement -- and
      # get wrong -- the rules for `<2`, `>=1.0,!=2.2`, `~=2.3` and the rest.
      # The deployed image runs `adk api_server`, which reaches for
      # `Client.runtimes`; a v1 release only has `Client.agent_engines`, so
      # failing the build is the point. Every pinned distribution gets its own
      # floor: both ship the `agentplatform` package, so a v1 pin on either one
      # can put the v1 surface on disk.
      for name in pinned_distributions:
        f.write(f'{name}>={_AGENT_ENGINE_MIN_VERSION},<3\n')
    else:
      f.write(f'{_AGENT_ENGINE_REQUIREMENT}\n')
      f.write(f'google-adk[a2a]=={__version__}\n')


# What a deployment advertises to Agent Platform: one entry per operation the
# AdkApp template registers, mirroring the method's own documentation and
# signature. A unit test checks the operation names and the parameters each
# entry declares against the installed template, because a catalogue that
# disagrees with the method makes the deployed resource advertise an API it
# does not serve. The descriptions are not checked: wording drift is cosmetic,
# and it varies with whichever Agent Platform SDK the version floor resolves
# to.
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
        'name': 'async_save_artifact',
        'description': (
            'Saves an artifact to the artifact service storage.\n\n       '
            ' Args:\n            user_id (str):\n                Required. The'
            ' ID of the user.\n            filename (str):\n               '
            ' Required. The filename of the artifact.\n            artifact'
            ' (Union[types.Part, Dict[str, Any], str]):\n               '
            ' Required. The artifact to save.\n            session_id'
            ' (Optional[str]):\n                Optional. The ID of the'
            ' session.\n            custom_metadata (Optional[Dict[str,'
            ' Any]]):\n                Optional. Custom metadata to associate'
            ' with the artifact.\n            **kwargs (dict[str, Any]):\n     '
            '           Optional. Additional keyword arguments to pass to the\n'
            '                artifact service.\n\n        Returns:\n           '
            ' int: The revision ID.\n        '
        ),
        'parameters': {
            'properties': {
                'user_id': {'type': 'string'},
                'filename': {'type': 'string'},
                'artifact': {
                    'anyOf': [
                        {'additionalProperties': True, 'type': 'object'},
                        {'type': 'string'},
                    ]
                },
                'session_id': {'type': 'string', 'nullable': True},
                'custom_metadata': {'type': 'object', 'nullable': True},
            },
            'required': ['user_id', 'filename', 'artifact'],
            'type': 'object',
        },
        'api_mode': 'async',
    },
    {
        'name': 'async_load_artifact',
        'description': (
            'Gets an artifact from the artifact service storage.\n\n       '
            ' Args:\n            user_id (str):\n                Required. The'
            ' ID of the user.\n            filename (str):\n               '
            ' Required. The filename of the artifact.\n            session_id'
            ' (Optional[str]):\n                Optional. The ID of the'
            ' session.\n            version (Optional[int]):\n               '
            ' Optional. The version of the artifact.\n            **kwargs'
            ' (dict[str, Any]):\n                Optional. Additional keyword'
            ' arguments to pass to the\n                artifact service.\n\n  '
            '      Returns:\n            Optional[types.Part]: The artifact or'
            ' None if not found.\n        '
        ),
        'parameters': {
            'properties': {
                'user_id': {'type': 'string'},
                'filename': {'type': 'string'},
                'session_id': {'type': 'string', 'nullable': True},
                'version': {'type': 'integer', 'nullable': True},
            },
            'required': ['user_id', 'filename'],
            'type': 'object',
        },
        'api_mode': 'async',
    },
    {
        'name': 'async_list_artifact_keys',
        'description': (
            'Lists all the artifact filenames within a session.\n\n       '
            ' Args:\n            user_id (str):\n                Required. The'
            ' ID of the user.\n            session_id (Optional[str]):\n       '
            '         Optional. The ID of the session.\n            **kwargs'
            ' (dict[str, Any]):\n                Optional. Additional keyword'
            ' arguments to pass to the\n                artifact service.\n\n  '
            '      Returns:\n            list[str]: A list of artifact'
            ' filenames.\n        '
        ),
        'parameters': {
            'properties': {
                'user_id': {'type': 'string'},
                'session_id': {'type': 'string', 'nullable': True},
            },
            'required': ['user_id'],
            'type': 'object',
        },
        'api_mode': 'async',
    },
    {
        'name': 'async_delete_artifact',
        'description': (
            'Deletes an artifact.\n\n        Args:\n            user_id'
            ' (str):\n                Required. The ID of the user.\n          '
            '  filename (str):\n                Required. The filename of the'
            ' artifact.\n            session_id (Optional[str]):\n             '
            '   Optional. The ID of the session.\n            **kwargs'
            ' (dict[str, Any]):\n                Optional. Additional keyword'
            ' arguments to pass to the\n                artifact service.\n    '
            '    '
        ),
        'parameters': {
            'properties': {
                'user_id': {'type': 'string'},
                'filename': {'type': 'string'},
                'session_id': {'type': 'string', 'nullable': True},
            },
            'required': ['user_id', 'filename'],
            'type': 'object',
        },
        'api_mode': 'async',
    },
    {
        'name': 'async_list_versions',
        'description': (
            'Lists all versions of an artifact.\n\n        Args:\n           '
            ' user_id (str):\n                Required. The ID of the user.\n  '
            '          filename (str):\n                Required. The filename'
            ' of the artifact.\n            session_id (Optional[str]):\n      '
            '          Optional. The ID of the session.\n            **kwargs'
            ' (dict[str, Any]):\n                Optional. Additional keyword'
            ' arguments to pass to the\n                artifact service.\n\n  '
            '      Returns:\n            list[int]: A list of all available'
            ' versions of the artifact.\n        '
        ),
        'parameters': {
            'properties': {
                'user_id': {'type': 'string'},
                'filename': {'type': 'string'},
                'session_id': {'type': 'string', 'nullable': True},
            },
            'required': ['user_id', 'filename'],
            'type': 'object',
        },
        'api_mode': 'async',
    },
    {
        'name': 'async_list_artifact_versions',
        'description': (
            'Lists all versions and their metadata for a specific'
            ' artifact.\n\n        Args:\n            user_id (str):\n         '
            '       Required. The ID of the user.\n            filename'
            ' (str):\n                Required. The filename of the'
            ' artifact.\n            session_id (Optional[str]):\n             '
            '   Optional. The ID of the session.\n            **kwargs'
            ' (dict[str, Any]):\n                Optional. Additional keyword'
            ' arguments to pass to the\n                artifact service.\n\n  '
            '      Returns:\n            list[ArtifactVersion]: A list of'
            ' ArtifactVersion objects.\n        '
        ),
        'parameters': {
            'properties': {
                'user_id': {'type': 'string'},
                'filename': {'type': 'string'},
                'session_id': {'type': 'string', 'nullable': True},
            },
            'required': ['user_id', 'filename'],
            'type': 'object',
        },
        'api_mode': 'async',
    },
    {
        'name': 'async_get_artifact_version',
        'description': (
            'Gets the metadata for a specific version of an artifact.\n\n      '
            '  Args:\n            user_id (str):\n                Required. The'
            ' ID of the user.\n            filename (str):\n               '
            ' Required. The filename of the artifact.\n            session_id'
            ' (Optional[str]):\n                Optional. The ID of the'
            ' session.\n            version (Optional[int]):\n               '
            ' Optional. The version number of the artifact.\n            '
            '**kwargs (dict[str, Any]):\n                Optional. Additional'
            ' keyword arguments to pass to the\n                artifact'
            ' service.\n\n        Returns:\n            Optional['
            'ArtifactVersion]: An ArtifactVersion object or None.\n        '
        ),
        'parameters': {
            'properties': {
                'user_id': {'type': 'string'},
                'filename': {'type': 'string'},
                'session_id': {'type': 'string', 'nullable': True},
                'version': {'type': 'integer', 'nullable': True},
            },
            'required': ['user_id', 'filename'],
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
            ' created for the user. If this is specified, then\n               '
            ' `session_events` will be ignored.\n            session_events'
            ' (Optional[List[Dict[str, Any]]]):\n                Optional. The'
            ' session events to use for the query. This will be\n             '
            '   used to initialize the session if `session_id` is not'
            ' provided.\n            run_config (Optional[Dict[str,'
            ' Any]]):\n                Optional. The run config to use for the'
            ' query. If you want to\n                pass in a `run_config`'
            ' pydantic object, you can pass in a dict\n               '
            ' representing it as `run_config.model_dump(mode="json")`.\n       '
            '     **kwargs (dict[str, Any]):\n                Optional.'
            ' Additional keyword arguments to pass to the\n               '
            ' runner.\n\n        Yields:\n            Event dictionaries'
            ' asynchronously.\n\n        Raises:\n            TypeError: If'
            ' message is not a string or a dictionary representing\n           '
            ' a Content object.\n            ValueError: If both session_id and'
            ' session_events are specified.\n        '
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
                'session_events': {'type': 'array', 'nullable': True},
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


def _validate_dockerfile_env_value(name: str, value: Optional[str]) -> None:
  """Validates a value before it is written into a Dockerfile ENV instruction.

  Args:
    name: The environment variable name, used in the error message. The value
      itself is never echoed, because it can come from the agent folder's `.env`
      file.
    value: The value to write.

  Raises:
    click.ClickException: If the value spans more than one line.
  """
  if value is not None and ('\n' in value or '\r' in value):
    raise click.ClickException(
        f'Invalid value for {name}. The value is written into the generated'
        ' Dockerfile and must not span multiple lines.'
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
    importlib.invalidate_caches()
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


def _get_service_options_by_adk_version(
    adk_version: str,
    session_uri: Optional[str],
    artifact_uri: Optional[str],
    memory_uri: Optional[str],
    use_local_storage: Optional[bool] = None,
) -> list[str]:
  """Returns the service options based on adk_version, one per argv entry."""
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

  return options


def _get_ignore_patterns_func(
    agent_folder: str,
) -> Callable[[Any, list[str]], set[str]]:
  """Returns a shutil.ignore_patterns function that excludes the local .adk folder along with the combined patterns from .gitignore, .gcloudignore and .ae_ignore."""
  # .adk holds the developer's own sessions and artifacts.
  patterns = {'.adk'}

  for filename in ['.gitignore', '.gcloudignore', '.ae_ignore']:
    filepath = os.path.join(agent_folder, filename)
    if os.path.exists(filepath):
      click.echo(f'Reading ignore patterns from {filename}...')
      try:
        with open(filepath, 'r', encoding='utf-8') as f:
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


def _stage_extra_packages(
    requested_extra_packages: Sequence[tuple[str, str]],
    build_context: str,
    *,
    reserved_names: Sequence[str],
) -> list[str]:
  """Copies extra packages into the container build context.

  Args:
    requested_extra_packages: (path, base_dir) pairs. A relative path is
      resolved against its base_dir; an absolute path is taken as is.
    build_context: The folder the container image is built from.
    reserved_names: Names the deployment generates in the build context after
      this point, and which an entry must therefore not take.

  Returns:
    The names of the staged entries, relative to the build context.

  Raises:
    click.ClickException: If a path does not exist, or its name collides with
      something else in the build context.
  """
  staged_extra_packages: list[str] = []
  for pkg, base_dir in requested_extra_packages:
    pkg_src = pkg if os.path.isabs(pkg) else os.path.join(base_dir, pkg)
    pkg_src = os.path.abspath(pkg_src)
    if not os.path.exists(pkg_src):
      raise click.ClickException(f'extra_packages path not found: {pkg}')
    base = os.path.basename(os.path.normpath(pkg_src))
    dst = os.path.join(build_context, base)
    if os.path.exists(dst) or base in reserved_names:
      raise click.ClickException(
          f'extra_packages entry has a conflicting name: {base}'
      )
    if os.path.isdir(pkg_src):
      shutil.copytree(pkg_src, dst, dirs_exist_ok=True)
    else:
      shutil.copy2(pkg_src, dst)
    staged_extra_packages.append(base)
  return staged_extra_packages


def _get_extra_packages_copy(
    staged_extra_packages: Sequence[str], build_context: str
) -> str:
  """Returns the Dockerfile lines that copy staged extra packages."""
  if not staged_extra_packages:
    return ''
  copy_lines = [
      f'COPY --chown=myuser:myuser "{base}/" "/app/{base}/"'
      if os.path.isdir(os.path.join(build_context, base))
      else f'COPY --chown=myuser:myuser "{base}" "/app/{base}"'
      for base in staged_extra_packages
  ]
  copy_lines.append('ENV PYTHONPATH="/app:$PYTHONPATH"')
  return '\n'.join(copy_lines)


def run(
    *,
    agent_folder: str,
    provider: str,
    project: str | None = None,
    region: str | None = None,
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
    allow_origins: list[str] | None = None,
    session_service_uri: str | None = None,
    artifact_service_uri: str | None = None,
    memory_service_uri: str | None = None,
    use_local_storage: bool = False,
    a2a: bool = False,
    trigger_sources: str | None = None,
    trigger_oidc_audience: str | None = None,
    trigger_oidc_service_accounts: str | None = None,
    provider_args: tuple[str, ...] = (),
    env: tuple[str, ...] = (),
    extra_gcloud_args: tuple[str, ...] | None = None,
    with_cloud_run_sandbox: bool = False,
    extra_packages: list[str] | None = None,
) -> None:
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
    with_cloud_run_sandbox: Whether to enable the Cloud Run sandbox for code
      execution.
    extra_packages: Additional local file or directory paths to stage alongside
      the agent and make importable in the image. A relative path is resolved
      against the current working directory.
  """
  _validate_trigger_options(
      trigger_sources, trigger_oidc_audience, trigger_oidc_service_accounts
  )
  app_name = _validate_app_name(
      app_name or os.path.basename(os.path.normpath(agent_folder))
  )
  if parse(adk_version) >= parse('1.3.0') and not use_local_storage:
    session_service_uri = session_service_uri or 'memory://'
    artifact_service_uri = artifact_service_uri or 'memory://'

  click.echo(f'Start generating {provider} source files in {temp_folder}')

  # remove temp_folder if exists
  if os.path.exists(temp_folder):
    click.echo('Removing existing files')
    _robust_rmtree(temp_folder)

  try:
    # copy agent source code
    click.echo('Copying agent source code...')
    agent_src_path = os.path.join(temp_folder, 'agents', app_name)
    ignore_func = _get_ignore_patterns_func(agent_folder)
    shutil.copytree(agent_folder, agent_src_path, ignore=ignore_func)
    requirements_txt_path = os.path.join(agent_src_path, 'requirements.txt')
    install_agent_deps = _render_install_agent_deps(
        app_name, requirements_txt_path, '# No requirements.txt found.'
    )
    click.echo('Copying agent source code completed.')

    staged_extra_packages = _stage_extra_packages(
        [(pkg, os.getcwd()) for pkg in extra_packages or []],
        temp_folder,
        reserved_names=('Dockerfile',),
    )

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
    trigger_oidc_audience_option = (
        f'--trigger_oidc_audience={trigger_oidc_audience}'
        if trigger_oidc_audience
        else ''
    )
    trigger_oidc_service_accounts_option = (
        f'--trigger_oidc_service_accounts={trigger_oidc_service_accounts}'
        if trigger_oidc_service_accounts
        else ''
    )
    dockerfile_content = _render_dockerfile(
        app_name=app_name,
        port=port,
        command='api_server --with_ui' if with_ui else 'api_server',
        install_agent_deps=install_agent_deps,
        service_options=_get_service_options_by_adk_version(
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
        trigger_oidc_audience_option=trigger_oidc_audience_option,
        trigger_oidc_service_accounts_option=trigger_oidc_service_accounts_option,
        gemini_enterprise_option='',
        express_mode_option='',
        extra_packages_copy=_get_extra_packages_copy(
            staged_extra_packages, temp_folder
        ),
        extra_env_vars='',
    )
    dockerfile_path = os.path.join(temp_folder, 'Dockerfile')
    os.makedirs(temp_folder, exist_ok=True)
    with open(dockerfile_path, 'w', encoding='utf-8') as f:
      f.write(
          dockerfile_content,
      )
    click.echo(f'Creating Dockerfile complete: {dockerfile_path}')

    # Deploy
    click.echo(f'Deploying to {provider}...')

    deployer = DeployerFactory.get_deployer(provider)
    deployer.deploy(
        agent_folder=agent_folder,
        temp_folder=temp_folder,
        service_name=service_name,
        provider_args=provider_args,
        env_vars=env,
        project=project,
        region=region,
        port=port,
        verbosity=verbosity,
        extra_gcloud_args=extra_gcloud_args,
        log_level=log_level,
        with_cloud_run_sandbox=with_cloud_run_sandbox,
    )
  finally:
    click.echo(f'Cleaning up the temp folder: {temp_folder}')
    _robust_rmtree(temp_folder)


def to_cloud_run(
    *,
    agent_folder: str,
    project: str | None = None,
    region: str | None = None,
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
    allow_origins: list[str] | None = None,
    session_service_uri: str | None = None,
    artifact_service_uri: str | None = None,
    memory_service_uri: str | None = None,
    use_local_storage: bool = False,
    a2a: bool = False,
    trigger_sources: str | None = None,
    trigger_oidc_audience: str | None = None,
    trigger_oidc_service_accounts: str | None = None,
    extra_gcloud_args: tuple[str, ...] | None = None,
    with_cloud_run_sandbox: bool = False,
) -> None:
  """Deploys an agent to Google Cloud Run (deprecated)."""
  warnings.warn(
      'to_cloud_run is deprecated, use run instead.',
      DeprecationWarning,
      stacklevel=2,
  )
  run(
      agent_folder=agent_folder,
      provider='cloud_run',
      project=project,
      region=region,
      service_name=service_name,
      app_name=app_name,
      temp_folder=temp_folder,
      port=port,
      trace_to_cloud=trace_to_cloud,
      otel_to_cloud=otel_to_cloud,
      with_ui=with_ui,
      log_level=log_level,
      verbosity=verbosity,
      adk_version=adk_version,
      allow_origins=allow_origins,
      session_service_uri=session_service_uri,
      artifact_service_uri=artifact_service_uri,
      memory_service_uri=memory_service_uri,
      use_local_storage=use_local_storage,
      a2a=a2a,
      trigger_sources=trigger_sources,
      trigger_oidc_audience=trigger_oidc_audience,
      trigger_oidc_service_accounts=trigger_oidc_service_accounts,
      provider_args=(),
      env=(),
      extra_gcloud_args=extra_gcloud_args,
      with_cloud_run_sandbox=with_cloud_run_sandbox,
  )


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


def _print_gemini_enterprise_hint() -> None:
  """Prints a pointer to the Gemini Enterprise registration docs."""
  click.secho(
      'To make this agent available in Gemini Enterprise, register it by'
      ' following:\nhttps://docs.cloud.google.com/gemini/enterprise/docs/register-and-manage-an-adk-agent\n',
      fg='cyan',
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
    trigger_oidc_audience: Optional[str] = None,
    trigger_oidc_service_accounts: Optional[str] = None,
    memory_service_uri: Optional[str] = None,
    session_service_uri: Optional[str] = None,
    artifact_service_uri: Optional[str] = None,
    adk_version: Optional[str] = None,
    extra_packages: Optional[list[str]] = None,
    worker_pool: Optional[str] = None,
    service_account: Optional[str] = None,
) -> None:
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
    extra_packages (list[str]): Optional. Additional local file or directory
      paths to stage alongside the agent and make importable in the image.
    worker_pool (str): Optional. Full Cloud Build private worker pool resource
      name
      (`projects/{project}/locations/{location}/workerPools/{pool}`).
      When set, Agent Engine builds the container image on that pool so
      deploys can reach private networks / comply with org build policies.
      Overrides `worker_pool` / `build_config.worker_pool` from
      `.agent_engine_config.json` when both are present.
    service_account (str): Optional. Google Cloud service account email used
      as the Agent Engine runtime identity. Overrides
      ``GOOGLE_CLOUD_SERVICE_ACCOUNT`` in the ``.env`` file and
      ``service_account`` in ``.agent_engine_config.json`` when both are
      present. When omitted, Agent Engine uses its default service agent.
  """
  _validate_trigger_options(
      trigger_sources, trigger_oidc_audience, trigger_oidc_service_accounts
  )
  app_name = os.path.basename(os.path.normpath(agent_folder))
  _validate_app_name(app_name)
  display_name = display_name or app_name
  parent_folder = os.path.dirname(os.path.normpath(agent_folder))
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
  agent_folder_abs = os.path.abspath(agent_folder)
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
    _robust_rmtree(temp_folder_path)

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
      with open(agent_engine_config_file, 'r', encoding='utf-8') as f:
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

    _apply_worker_pool_to_agent_config(agent_config, worker_pool)

    config_extra_packages = agent_config.pop('extra_packages', None) or []
    # CLI entries resolve against the invocation dir; config-file entries
    # against the agent folder that declared them.
    requested_extra_packages = [
        (pkg, original_cwd) for pkg in extra_packages or []
    ] + [(pkg, agent_folder_abs) for pkg in config_extra_packages]
    staged_extra_packages = _stage_extra_packages(
        requested_extra_packages,
        temp_folder_path,
        reserved_names=('Dockerfile',),
    )

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
      # Pop so the SA email is not forwarded as a runtime env var.
      if 'GOOGLE_CLOUD_SERVICE_ACCOUNT' in env_vars:
        env_service_account = env_vars.pop('GOOGLE_CLOUD_SERVICE_ACCOUNT')
        if env_service_account:
          if service_account:
            click.secho(
                'Ignoring GOOGLE_CLOUD_SERVICE_ACCOUNT in .env as'
                ' `--service_account` was explicitly passed and takes'
                ' precedence',
                fg='yellow',
            )
          else:
            service_account = env_service_account
            click.echo(
                f'{service_account=} set by GOOGLE_CLOUD_SERVICE_ACCOUNT in'
                f' {env_file}'
            )
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
        # sorted() over a dict iterates its keys, so this prints the variable
        # names only and never their values.
        click.echo(
            'Overriding env_vars in agent platform config with'
            f' {sorted(env_vars)}'
        )
      agent_config['env_vars'] = env_vars
    _apply_service_account_to_agent_config(agent_config, service_account)
    # Set env_vars in agent_config to None if it is not set.
    agent_config['env_vars'] = agent_config.get('env_vars', env_vars)

    from ..dependencies._agentplatform import agentplatform
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
      client = agentplatform.Client(
          project=project,
          location=region,
          http_options={'headers': get_tracking_headers()},
      )
      click.echo('Agent Platform client initialized with project and region.')
    elif api_key:
      client = agentplatform.Client(
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

    # A v1 google-cloud-aiplatform also ships an importable `agentplatform`,
    # whose client exposes `agent_engines` where v2 exposes `runtimes`. The
    # import above therefore succeeds against either, and without this check
    # the mismatch surfaces only as a bare AttributeError from the create call
    # further down, after the deploy already looks under way.
    if getattr(client, 'runtimes', None) is None:
      raise click.ClickException(
          'The installed Agent Platform SDK exposes `Client.agent_engines`'
          ' rather than `Client.runtimes`, so it predates the surface this'
          ' deployment uses. Install google-cloud-agentplatform>=2.2, or'
          ' google-cloud-aiplatform>=2.2,<3 if you depend on the bundled'
          ' distribution.'
      )

    if skip_agent_import_validation:
      warnings.warn(
          'WARNING: `--skip-agent-import-validation` is deprecated and will be'
          ' removed in the future. Please drop it from the list of arguments.',
          DeprecationWarning,
          stacklevel=2,
      )

    # Validated before the instance is created, so a failure cannot leak one.
    enterprise_val = env_vars.get('GOOGLE_GENAI_USE_ENTERPRISE', '1')
    _validate_dockerfile_env_value(
        'GOOGLE_GENAI_USE_ENTERPRISE', enterprise_val
    )
    _validate_dockerfile_env_value('GOOGLE_CLOUD_PROJECT', project)
    _validate_dockerfile_env_value('GOOGLE_CLOUD_LOCATION', region)

    def create_dockerfile_for_agent_engine(resource_name: str) -> None:
      requirements_txt_path = os.path.join(agent_src_path, 'requirements.txt')
      install_agent_deps = _render_install_agent_deps(
          app_name, requirements_txt_path, '# No requirements.txt found.'
      )
      trigger_sources_option = (
          f'--trigger_sources={trigger_sources}' if trigger_sources else ''
      )
      trigger_oidc_audience_option = (
          f'--trigger_oidc_audience={trigger_oidc_audience}'
          if trigger_oidc_audience
          else ''
      )
      trigger_oidc_service_accounts_option = (
          f'--trigger_oidc_service_accounts={trigger_oidc_service_accounts}'
          if trigger_oidc_service_accounts
          else ''
      )
      agent_engine_uri = f'agentengine://{resource_name}'
      supports_gemini_enterprise_flag = parse(adk_version) >= parse(
          _GEMINI_ENTERPRISE_FLAG_MIN_VERSION
      )
      if not supports_gemini_enterprise_flag:
        click.secho(
            'Omitting --gemini_enterprise_app_name as it requires adk_version'
            f' {_GEMINI_ENTERPRISE_FLAG_MIN_VERSION} or later, and'
            f' {adk_version} was requested',
            fg='yellow',
        )
      gcp_env_lines = [f'ENV GOOGLE_GENAI_USE_ENTERPRISE={enterprise_val}']
      if project:
        gcp_env_lines.append(f'ENV GOOGLE_CLOUD_PROJECT={project}')
      if region:
        gcp_env_lines.append(f'ENV GOOGLE_CLOUD_LOCATION={region}')
      extra_env_vars = (
          '\n' + '\n'.join(gcp_env_lines) + '\n' if gcp_env_lines else ''
      )
      dockerfile_content = _render_dockerfile(
          app_name=app_name,
          port=8080,
          command='api_server',
          install_agent_deps=install_agent_deps,
          service_options=_get_service_options_by_adk_version(
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
          trigger_oidc_audience_option=trigger_oidc_audience_option,
          trigger_oidc_service_accounts_option=trigger_oidc_service_accounts_option,
          gemini_enterprise_option=(
              f'--gemini_enterprise_app_name={app_name}'
              if supports_gemini_enterprise_flag
              else ''
          ),
          express_mode_option=(
              '--express_mode' if api_key and not project else ''
          ),
          extra_packages_copy=_get_extra_packages_copy(
              staged_extra_packages, temp_folder_path
          ),
          extra_env_vars=extra_env_vars,
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
    agent_config['source_packages'] = [
        f'agents/{app_name}',
        'Dockerfile',
        *staged_extra_packages,
    ]
    agent_config['image_spec'] = {}  # Use the Dockerfile
    agent_config['class_methods'] = _AGENT_ENGINE_CLASS_METHODS
    agent_config['agent_framework'] = 'google-adk'

    resource_name = agent_engine_id
    if not resource_name:
      runtime = client.runtimes.create()
      resource_name = runtime.api_resource.name
      click.secho(f'Created a new instance: {resource_name}', fg='green')
    elif project and region and not resource_name.startswith('projects/'):
      resource_name = f'projects/{project}/locations/{region}/reasoningEngines/{agent_engine_id}'
    click.echo('Creating Dockerfile...')
    create_dockerfile_for_agent_engine(resource_name)
    click.echo(f'Dockerfile created at {os.getcwd()}/Dockerfile.')
    try:
      client.runtimes.update(name=resource_name, config=agent_config)
      click.secho(f'Deployed to Agent Platform: {resource_name}', fg='green')
    except Exception as e:
      click.secho(f'Failed to deploy to Agent Platform: {e}', fg='red')
      # Only delete the instance if it was newly created in this function.
      if agent_engine_id is None:
        client.runtimes.delete(name=resource_name)
        click.secho(f'Cleaned up the instance: {resource_name}', fg='green')
      raise e
    _print_agent_engine_url(resource_name)
    _print_gemini_enterprise_hint()
  finally:
    temp_folder_path = os.path.join(parent_folder, temp_folder)
    click.echo(f'Cleaning up the temp folder: {temp_folder_path}')
    os.chdir(original_cwd)
    _robust_rmtree(temp_folder_path)


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
    trigger_oidc_audience: Optional[str] = None,
    trigger_oidc_service_accounts: Optional[str] = None,
    service_type: Literal[
        'ClusterIP', 'NodePort', 'LoadBalancer'
    ] = 'ClusterIP',
    extra_packages: Optional[list[str]] = None,
) -> None:
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
    extra_packages: Additional local file or directory paths to stage alongside
      the agent and make importable in the image. A relative path is resolved
      against the current working directory.
  """
  _validate_trigger_options(
      trigger_sources, trigger_oidc_audience, trigger_oidc_service_accounts
  )
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

  app_name = app_name or os.path.basename(os.path.normpath(agent_folder))
  _validate_app_name(app_name)
  if parse(adk_version) >= parse('1.3.0') and not use_local_storage:
    session_service_uri = session_service_uri or 'memory://'
    artifact_service_uri = artifact_service_uri or 'memory://'

  click.secho('STEP 1: Preparing build environment...', bold=True)
  click.echo(f'  - Using temporary directory: {temp_folder}')

  # remove temp_folder if exists
  if os.path.exists(temp_folder):
    click.echo('  - Removing existing temporary directory...')
    _robust_rmtree(temp_folder)

  try:
    # copy agent source code
    click.echo('  - Copying agent source code...')
    agent_src_path = os.path.join(temp_folder, 'agents', app_name)
    ignore_func = _get_ignore_patterns_func(agent_folder)
    shutil.copytree(agent_folder, agent_src_path, ignore=ignore_func)
    requirements_txt_path = os.path.join(agent_src_path, 'requirements.txt')
    install_agent_deps = _render_install_agent_deps(
        app_name, requirements_txt_path, ''
    )
    staged_extra_packages = _stage_extra_packages(
        [(pkg, os.getcwd()) for pkg in extra_packages or []],
        temp_folder,
        reserved_names=('Dockerfile', 'deployment.yaml'),
    )
    click.secho('✅ Environment prepared.', fg='green')

    allow_origins_option = (
        f'--allow_origins={",".join(allow_origins)}' if allow_origins else ''
    )

    # create Dockerfile
    click.secho('\nSTEP 2: Generating deployment files...', bold=True)
    click.echo('  - Creating Dockerfile...')
    host_option = '--host=0.0.0.0' if adk_version > '0.5.0' else ''
    trigger_oidc_audience_option = (
        f'--trigger_oidc_audience={trigger_oidc_audience}'
        if trigger_oidc_audience
        else ''
    )
    trigger_oidc_service_accounts_option = (
        f'--trigger_oidc_service_accounts={trigger_oidc_service_accounts}'
        if trigger_oidc_service_accounts
        else ''
    )
    dockerfile_content = _render_dockerfile(
        app_name=app_name,
        port=port,
        command='api_server --with_ui' if with_ui else 'api_server',
        install_agent_deps=install_agent_deps,
        service_options=_get_service_options_by_adk_version(
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
        trigger_oidc_audience_option=trigger_oidc_audience_option,
        trigger_oidc_service_accounts_option=trigger_oidc_service_accounts_option,
        gemini_enterprise_option='',
        express_mode_option='',
        extra_packages_copy=_get_extra_packages_copy(
            staged_extra_packages, temp_folder
        ),
        extra_env_vars='',
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
    # Tag each build uniquely rather than overwriting one floating tag.
    image_tag = datetime.now().strftime('%Y%m%d-%H%M%S')
    image_name = f'gcr.io/{project}/{service_name}:{image_tag}'
    subprocess.run(
        [
            _GCLOUD_CMD,
            'builds',
            'submit',
            '--tag',
            image_name,
            '--project',
            project,
            '--verbosity',
            log_level.lower(),
            temp_folder,
        ],
        check=True,
    )
    click.secho('✅ Container image built and pushed successfully.', fg='green')

    # Create a Kubernetes deployment
    click.echo('  - Creating Kubernetes deployment.yaml...')
    env_entries = [
        '        - name: GOOGLE_GENAI_USE_ENTERPRISE\n          value: "1"',
        f'        - name: GOOGLE_CLOUD_PROJECT\n          value: "{project}"',
    ]
    if region:
      env_entries.append(
          f'        - name: GOOGLE_CLOUD_LOCATION\n          value: "{region}"'
      )
    env_yaml = '        env:\n' + '\n'.join(env_entries)

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
{env_yaml}
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
    region_options = ['--region', region] if region else []
    subprocess.run(
        [
            _GCLOUD_CMD,
            'container',
            'clusters',
            'get-credentials',
            cluster_name,
            *region_options,
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
    _robust_rmtree(temp_folder)
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
