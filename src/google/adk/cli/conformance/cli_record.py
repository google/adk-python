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

"""CLI commands for ADK conformance testing."""

from __future__ import annotations

from pathlib import Path

import click
from google.genai import types

from ...utils.yaml_utils import dump_pydantic_to_yaml
from ..adk_web_server import RunAgentRequest
from ._generated_file_utils import load_test_case
from .adk_web_server_client import AdkWebServerClient
from .test_case import TestCase


async def _create_conformance_test_files(
    test_case: TestCase,
    user_id: str = "adk_conformance_test_user",
) -> Path:
  """Generate conformance test files from TestCase."""
  # Clean existing generated files
  test_case_dir = test_case.dir

  # Remove existing generated files to ensure clean state
  generated_session_file = test_case_dir / "generated-session.yaml"
  generated_recordings_file = test_case_dir / "generated-recordings.yaml"

  generated_session_file.unlink(missing_ok=True)
  generated_recordings_file.unlink(missing_ok=True)

  async with AdkWebServerClient() as client:
    # Create a new session for the test
    session = await client.create_session(
        app_name=test_case.test_spec.agent,
        user_id=user_id,
        state=test_case.test_spec.initial_state,
    )

    # Run the agent with the user messages
    for user_message_index, user_message in enumerate(
        test_case.test_spec.user_messages
    ):
      # Create content from UserMessage object
      if user_message.content is not None:
        content = user_message.content
      elif user_message.text is not None:
        content = types.UserContent(parts=[types.Part(text=user_message.text)])
      else:
        raise ValueError(
            f"UserMessage at index {user_message_index} has neither text nor"
            " content"
        )

      async for _ in client.run_agent(
          RunAgentRequest(
              app_name=test_case.test_spec.agent,
              user_id=user_id,
              session_id=session.id,
              new_message=content,
              state_delta=user_message.state_delta,
          ),
          mode="record",
          test_case_dir=str(test_case_dir),
          user_message_index=user_message_index,
      ):
        pass

    # Retrieve the updated session
    updated_session = await client.get_session(
        app_name=test_case.test_spec.agent,
        user_id=user_id,
        session_id=session.id,
    )

    # Save session.yaml
    dump_pydantic_to_yaml(
        updated_session,
        generated_session_file,
        sort_keys=False,  # Output keys in the declaration order.
        exclude={
            "state": {"_adk_recordings_config": True},
            "events": {
                "__all__": {
                    "actions": {"state_delta": {"_adk_recordings_config": True}}
                }
            },
        },
    )

    return generated_session_file


async def run_conformance_record(paths: list[Path]) -> None:
  """Generate conformance tests from TestCaseInput files.

  Args:
    paths: list of directories containing test cases input files (spec.yaml).
  """
  click.echo("Generating ADK conformance tests...")

  # Look for spec.yaml files and load TestCase objects
  test_cases: dict[Path, TestCase] = {}

  for test_dir in paths:
    if not test_dir.exists():
      continue

    for spec_file in test_dir.rglob("spec.yaml"):
      try:
        test_case_dir = spec_file.parent
        category = test_case_dir.parent.name
        name = test_case_dir.name
        test_spec = load_test_case(test_case_dir)
        test_case = TestCase(
            category=category,
            name=name,
            dir=test_case_dir,
            test_spec=test_spec,
        )
        test_cases[test_case_dir] = test_case
        click.echo(f"Loaded test spec: {category}/{name}")
      except Exception as e:
        click.secho(f"Failed to load {spec_file}: {e}", fg="red", err=True)

  # Process all loaded test cases
  if test_cases:
    click.echo(f"\nProcessing {len(test_cases)} test cases...")

    for test_case in test_cases.values():
      try:
        await _create_conformance_test_files(test_case)
        click.secho(
            "Generated conformance test files for:"
            f" {test_case.category}/{test_case.name}",
            fg="green",
        )
      except Exception as e:
        click.secho(
            f"Failed to generate {test_case.category}/{test_case.name}: {e}",
            fg="red",
            err=True,
        )
  else:
    click.secho("No test specs found to process.", fg="yellow")

  click.secho("\nConformance test generation complete!", fg="blue")
