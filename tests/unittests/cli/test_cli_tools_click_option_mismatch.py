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

"""Unit tests to check if any Click options and method parameters mismatch."""

import inspect
from typing import MutableMapping
from typing import Optional

import click
from google.adk.cli.cli_tools_click import cli_api_server
from google.adk.cli.cli_tools_click import cli_create_cmd
from google.adk.cli.cli_tools_click import cli_deploy_agent_engine
from google.adk.cli.cli_tools_click import cli_deploy_cloud_run
from google.adk.cli.cli_tools_click import cli_deploy_gke
from google.adk.cli.cli_tools_click import cli_eval
from google.adk.cli.cli_tools_click import cli_run
from google.adk.cli.cli_tools_click import cli_web
from google.adk.cli.cli_tools_click import deploy
from google.adk.cli.cli_tools_click import main


def _get_command_by_name(
    commands: MutableMapping[str, click.Command], name
) -> Optional[click.Command]:
  """Return the command object with the given name from a commands dict."""
  return next((cmd for cmd in commands.values() if cmd.name == name), None)


def _get_click_options(command) -> set[str]:
  """Extract Click option names from a command."""
  options = []
  for param in command.params:
    if isinstance(param, (click.Option, click.Argument)):
      options.append(param.name)
  return set(options)


def _get_method_parameters(func) -> set[str]:
  """Extract parameter names from a method signature."""
  sig = inspect.signature(func)
  return set(sig.parameters.keys())


def _check_options_in_parameters(
    command,
    func,
    command_name,
    ignore_params: Optional[set[str]] = None,
):
  """Check if all Click options are present in method parameters."""
  click_options = _get_click_options(command)
  method_params = _get_method_parameters(func)

  if ignore_params:
    click_options -= ignore_params
    method_params -= ignore_params

  option_only = click_options - method_params
  parameter_only = method_params - click_options

  assert click_options == method_params, f"""\
Click options and method parameters do not match for command: `{command_name}`.
Click options: {click_options}
Method parameters: {method_params}
Options only: {option_only}
Parameters only: {parameter_only}
"""


def test_adk_create():
  """Test that cli_create_cmd has all required parameters."""
  create_command = _get_command_by_name(main.commands, "create")

  assert create_command is not None, "Create command not found"
  _check_options_in_parameters(
      create_command, cli_create_cmd.callback, "create"
  )


def test_adk_run():
  """Test that cli_run has all required parameters."""
  run_command = _get_command_by_name(main.commands, "run")

  assert run_command is not None, "Run command not found"
  _check_options_in_parameters(run_command, cli_run.callback, "run")


def test_adk_eval():
  """Test that cli_eval has all required parameters."""
  eval_command = _get_command_by_name(main.commands, "eval")

  assert eval_command is not None, "Eval command not found"
  _check_options_in_parameters(eval_command, cli_eval.callback, "eval")


def test_adk_web():
  """Test that cli_web has all required parameters."""
  web_command = _get_command_by_name(main.commands, "web")

  assert web_command is not None, "Web command not found"
  _check_options_in_parameters(
      web_command, cli_web.callback, "web", ignore_params={"verbose"}
  )


def test_adk_api_server():
  """Test that cli_api_server has all required parameters."""
  api_server_command = _get_command_by_name(main.commands, "api_server")

  assert api_server_command is not None, "API server command not found"
  _check_options_in_parameters(
      api_server_command,
      cli_api_server.callback,
      "api_server",
      ignore_params={"verbose"},
  )


def test_adk_deploy_cloud_run():
  """Test that cli_deploy_cloud_run has all required parameters."""
  cloud_run_command = _get_command_by_name(deploy.commands, "cloud_run")

  assert cloud_run_command is not None, "Cloud Run deploy command not found"
  _check_options_in_parameters(
      cloud_run_command,
      cli_deploy_cloud_run.callback,
      "deploy cloud_run",
      ignore_params={"verbose"},
  )


def test_adk_deploy_agent_engine():
  """Test that cli_deploy_agent_engine has all required parameters."""
  agent_engine_command = _get_command_by_name(deploy.commands, "agent_engine")

  assert (
      agent_engine_command is not None
  ), "Agent Engine deploy command not found"
  _check_options_in_parameters(
      agent_engine_command,
      cli_deploy_agent_engine.callback,
      "deploy agent_engine",
  )


def test_adk_deploy_gke():
  """Test that cli_deploy_gke has all required parameters."""
  gke_command = _get_command_by_name(deploy.commands, "gke")

  assert gke_command is not None, "GKE deploy command not found"
  _check_options_in_parameters(
      gke_command, cli_deploy_gke.callback, "deploy gke"
  )
