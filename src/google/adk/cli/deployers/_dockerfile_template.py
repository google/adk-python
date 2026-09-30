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

import json
import os
import re
from typing import Any
from typing import Final

import click

# app_name is interpolated into the generated Dockerfile (COPY/RUN instructions
# and CMD) by _DOCKERFILE_TEMPLATE. It defaults to the basename of the agent
# source folder, so its value can come from a directory name the deploying
# developer did not choose (a cloned or shared agent template). Restrict it to a
# plain identifier before it reaches the template so it cannot break out of a
# Dockerfile instruction or container path.
_APP_NAME_PATTERN: Final[re.Pattern[str]] = re.compile(
    r'^[A-Za-z0-9_-][A-Za-z0-9_.-]{0,62}$'
)


def _validate_app_name(app_name: str) -> str:
  """Validates the deploy app name before it is written into a Dockerfile.

  Args:
    app_name: The app name, either passed via --app_name or derived from the
      agent source folder basename.

  Returns:
    The validated app name, unchanged.

  Raises:
    click.ClickException: If the app name is not a plain identifier.
  """
  if not _APP_NAME_PATTERN.fullmatch(app_name):
    raise click.ClickException(
        f'Invalid app name {app_name!r}. The app name is used in the generated'
        ' Dockerfile and must contain only letters, digits, hyphens,'
        ' underscores, and periods (1-63 characters, starting with a letter,'
        ' digit, hyphen, or underscore).'
    )
  return app_name


def _validate_dockerfile_literal(name: str, value: Any) -> str:
  """Validates a value interpolated into a single-line Dockerfile instruction.

  A value containing a newline would end the instruction it is embedded in and
  let the remainder be parsed as additional Dockerfile instructions (for
  example an extra `RUN`).

  Args:
    name: The name of the value, used in the error message.
    value: The value to validate.

  Returns:
    The value as a string, unchanged.

  Raises:
    click.ClickException: If the value spans more than one line.
  """
  text = str(value)
  if '\n' in text or '\r' in text:
    raise click.ClickException(
        f'Invalid {name}: {text!r}. Value must not contain line breaks.'
    )
  return text


def _to_exec_form(instruction: str, args: list[str]) -> str:
  """Renders a Dockerfile instruction in exec (JSON array) form.

  Exec form is passed straight to `execve` rather than to `/bin/sh -c`, so no
  shell metacharacter in `args` (`$(...)`, backticks, `;`, `&&`, ...) is ever
  interpreted.

  Args:
    instruction: The Dockerfile instruction, e.g. `RUN` or `CMD`.
    args: The argv to run.

  Returns:
    The rendered instruction, e.g. `RUN ["pip", "install", "wheel"]`.
  """
  return f'{instruction} {json.dumps(args)}'


_DOCKERFILE_TEMPLATE: Final[str] = """
FROM python:3.11-slim
WORKDIR /app

# Create a non-root user
RUN adduser --disabled-password --gecos "" myuser

# Switch to the non-root user
USER myuser

# Set up environment variables - Start
ENV PATH="/home/myuser/.local/bin:$PATH"
{extra_env_vars}
# Set up environment variables - End

# Install ADK - Start
{install_adk}
# Remove dev_server.py to ensure production-safe endpoints only (disabling dev endpoints in production)
# Shell form on purpose: this instruction interpolates nothing, and `|| true`
# is shell syntax. Do not add a `{{...}}` placeholder to it without moving it to
# exec form first.
RUN python -c "import os, glob, google.adk.cli as cli; d = os.path.dirname(cli.__file__); [os.remove(f) for f in glob.glob(os.path.join(d, 'dev_server*'))]; [os.remove(f) for f in glob.glob(os.path.join(d, '__pycache__', 'dev_server*'))]" || true
# Install ADK - End

# Copy agent - Start

# Set permission
COPY --chown=myuser:myuser "agents/{app_name}/" "/app/agents/{app_name}/"
{extra_packages_copy}
# Copy agent - End

# Install Agent Deps - Start
{install_agent_deps}
# Install Agent Deps - End

EXPOSE {port}

{start_command}
"""


def _render_install_agent_deps(
    app_name: str, requirements_txt_path: str, fallback: str
) -> str:
  """Renders the instruction installing the agent's own requirements.

  Args:
    app_name: The name of the app; must pass `_validate_app_name`.
    requirements_txt_path: Path to the staged `requirements.txt`.
    fallback: What to emit when the agent ships no `requirements.txt`.

  Returns:
    A `RUN` instruction in exec form, or `fallback`.
  """
  if not os.path.exists(requirements_txt_path):
    return fallback
  _validate_app_name(app_name)
  return _to_exec_form(
      'RUN',
      ['pip', 'install', '-r', f'/app/agents/{app_name}/requirements.txt'],
  )


def _render_dockerfile(
    *,
    app_name: str,
    port: Any,
    command: str,
    install_agent_deps: str,
    service_options: list[str],
    trace_to_cloud_option: str,
    otel_to_cloud_option: str,
    allow_origins_option: str,
    adk_version: str,
    host_option: str,
    a2a_option: str,
    trigger_sources_option: str,
    trigger_oidc_audience_option: str = '',
    trigger_oidc_service_accounts_option: str = '',
    gemini_enterprise_option: str = '',
    express_mode_option: str = '',
    extra_packages_copy: str = '',
    extra_env_vars: str = '',
) -> str:
  """Renders the Dockerfile used by every `adk deploy` target.

  `RUN` and `CMD` are emitted in exec form so that the generated Dockerfile
  never hands an interpolated value to a shell, neither at build time nor at
  container start time.

  Args:
    app_name: The name of the app; must pass `_validate_app_name`.
    port: The port the ADK api server listens on.
    command: The `adk` subcommand plus its own flags, space separated. This is
      an ADK-internal literal, never a user supplied value.
    install_agent_deps: The already-rendered instruction that installs the
      agent's own requirements, or a comment if there are none.
    service_options: Session/artifact/memory service flags, one per entry.
    trace_to_cloud_option: `--trace_to_cloud` or an empty string.
    otel_to_cloud_option: `--otel_to_cloud` or an empty string.
    allow_origins_option: `--allow_origins=...` or an empty string.
    adk_version: The ADK version to install in the image.
    host_option: `--host=...` or an empty string.
    a2a_option: `--a2a` or an empty string.
    trigger_sources_option: `--trigger_sources=...` or an empty string.
    trigger_oidc_audience_option: `--trigger_oidc_audience=...` or an empty
      string.
    trigger_oidc_service_accounts_option: `--trigger_oidc_service_accounts=...`
      or an empty string.
    gemini_enterprise_option: `--gemini_enterprise_app_name=...` or an empty
      string.
    express_mode_option: `--express_mode` or an empty string.
    extra_packages_copy: Extra `COPY`/`ENV` instructions, or an empty string.
    extra_env_vars: Extra `ENV` lines, or an empty string.

  Returns:
    The Dockerfile contents.
  """
  _validate_app_name(app_name)
  adk_version = _validate_dockerfile_literal('adk_version', adk_version)
  port = _validate_dockerfile_literal('port', port)

  # Every entry below is one argv entry. Values are never split on whitespace
  # and never quoted: exec form passes them to the process verbatim.
  start_command_args = ['adk', *command.split(), f'--port={port}']
  start_command_args.extend(
      option.strip()
      for option in (
          host_option,
          *service_options,
          trace_to_cloud_option,
          otel_to_cloud_option,
          allow_origins_option,
          a2a_option,
          trigger_sources_option,
          trigger_oidc_audience_option,
          trigger_oidc_service_accounts_option,
          gemini_enterprise_option,
          express_mode_option,
      )
      if option and option.strip()
  )
  start_command_args.append('/app/agents')

  start_command = _to_exec_form('CMD', start_command_args)
  install_adk = _to_exec_form(
      'RUN', ['pip', 'install', f'google-adk[a2a]=={adk_version}']
  )

  return _DOCKERFILE_TEMPLATE.format(
      app_name=app_name,
      port=port,
      install_adk=install_adk,
      install_agent_deps=install_agent_deps,
      extra_packages_copy=extra_packages_copy,
      extra_env_vars=extra_env_vars,
      start_command=start_command,
  )
