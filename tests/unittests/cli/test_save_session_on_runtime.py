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

"""Unit tests for the save_session_on_runtime feature."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest import mock

from click.testing import CliRunner
from google.adk.cli.cli import run_cli
from google.adk.cli.cli_tools_click import main
from google.adk.sessions.session import Session


def test_cli_run_has_save_session_on_runtime_option():
  """Test that the run command has the --save_session_on_runtime option."""
  runner = CliRunner()
  result = runner.invoke(main, ["run", "--help"])
  assert "--save_session_on_runtime" in result.output


@mock.patch("google.adk.cli.cli_tools_click.run_cli")  # patch where it's USED
def test_cli_run_passes_save_session_on_runtime_flag(mock_run_cli):
  """Test that the run command passes the save_session_on_runtime flag to run_cli."""
  runner = CliRunner()
  result = runner.invoke(
      main,
      ["run", "contributing/samples/hello_world", "--save_session_on_runtime"],
      input="exit\n",
  )

  print(f"Exit code: {result.exit_code}")
  print(f"Output: {result.output}")
  if result.exception:
    print(f"Exception: {result.exception}")
    import traceback

    traceback.print_exception(
        type(result.exception), result.exception, result.exception.__traceback__
    )

  assert mock_run_cli.called, (
      f"run_cli was not called. Exit code: {result.exit_code}, Output:"
      f" {result.output}"
  )
  call_args = mock_run_cli.call_args
  assert call_args is not None
  assert call_args.kwargs.get("save_session_on_runtime") is True


@mock.patch("google.adk.cli.cli.create_session_service_from_options")
@mock.patch("google.adk.cli.cli.create_artifact_service_from_options")
@mock.patch("google.adk.cli.cli.create_memory_service_from_options")
@mock.patch("google.adk.cli.cli.InMemoryCredentialService")
@mock.patch("google.adk.cli.cli.AgentLoader")
@mock.patch("google.adk.cli.cli.load_services_module")
@mock.patch("google.adk.cli.cli.envs.load_dotenv_for_agent")
@mock.patch("google.adk.cli.cli.run_interactively")
def test_run_cli_saves_session_periodically(
    mock_run_interactively,
    mock_load_dotenv,
    mock_load_services_module,
    mock_agent_loader,
    mock_credential_service,
    mock_memory_service,
    mock_artifact_service,
    mock_session_service,
):
  """Test that run_cli calls run_interactively with save_session_on_runtime=True when flag is set."""
  mock_session_service_instance = mock_session_service.return_value
  mock_session_service_instance.create_session = mock.AsyncMock(
      return_value=mock.Mock(spec=Session)
  )
  mock_session_service_instance.get_session = mock.AsyncMock(
      return_value=mock.Mock(spec=Session)
  )

  mock_agent_loader_instance = mock_agent_loader.return_value
  mock_agent_loader_instance.load_agent = mock.Mock(return_value=mock.Mock())

  asyncio.run(
      run_cli(
          agent_parent_dir="/fake/parent",
          agent_folder_name="fake_agent",
          save_session=False,
          save_session_on_runtime=True,
          interval=60,
          session_id=None,
          session_service_uri=None,
          artifact_service_uri=None,
          memory_service_uri=None,
          use_local_storage=True,
      )
  )

  mock_run_interactively.assert_called_once()
  call_args = mock_run_interactively.call_args
  assert call_args is not None
  assert call_args.kwargs.get("save_session_on_runtime") is True
  assert call_args.kwargs.get("agent_root") is not None
