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

"""CLI implementation for ADK conformance testing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import textwrap
from typing import Optional

import click
from google.genai import types

from ..adk_web_server import RunAgentRequest
from ._generated_file_utils import load_recorded_session
from ._generated_file_utils import load_test_case
from ._replay_validators import compare_events
from ._replay_validators import compare_session
from .adk_web_server_client import AdkWebServerClient
from .test_case import TestCase
from .test_case import TestSpec


@dataclass
class _TestResult:
  """Result of running a single conformance test."""

  category: str
  name: str
  success: bool
  error_message: Optional[str] = None


@dataclass
class _ConformanceTestSummary:
  """Summary of all conformance test results."""

  total_tests: int
  passed_tests: int
  failed_tests: int
  results: list[_TestResult]

  @property
  def success_rate(self) -> float:
    """Calculate the success rate as a percentage."""
    if self.total_tests == 0:
      return 0.0
    return (self.passed_tests / self.total_tests) * 100


class ConformanceTestRunner:
  """Runs conformance tests in replay mode."""

  def __init__(
      self,
      test_paths: list[Path],
      client: AdkWebServerClient,
      mode: str = "replay",
      user_id: str = "adk_conformance_test_user",
  ):
    self.test_paths = test_paths
    self.mode = mode
    self.client = client
    self.user_id = user_id

  def _discover_test_cases(self) -> list[TestCase]:
    """Discover test cases from specified folder paths."""
    test_cases = []
    for test_path in self.test_paths:
      if not test_path.exists() or not test_path.is_dir():
        click.secho(f"Invalid path: {test_path}", fg="yellow", err=True)
        continue

      for spec_file in test_path.rglob("spec.yaml"):
        test_case_dir = spec_file.parent
        category = test_case_dir.parent.name
        name = test_case_dir.name

        # Skip if recordings missing in replay mode
        if (
            self.mode == "replay"
            and not (test_case_dir / "generated-recordings.yaml").exists()
        ):
          click.secho(
              f"Skipping {category}/{name}: no recordings",
              fg="yellow",
              err=True,
          )
          continue

        test_spec = load_test_case(test_case_dir)
        test_cases.append(
            TestCase(
                category=category,
                name=name,
                dir=test_case_dir,
                test_spec=test_spec,
            )
        )

    return sorted(test_cases, key=lambda tc: (tc.category, tc.name))

  async def _run_user_messages(
      self, session_id: str, test_case: TestCase
  ) -> None:
    """Run all user messages for a test case."""
    function_call_name_to_id_map = {}
    for user_message_index, user_message in enumerate(
        test_case.test_spec.user_messages
    ):
      # Create content from UserMessage object
      if user_message.content is not None:
        content = user_message.content

        # If the user provides a function response, it means this is for
        # long-running tool. Replace the function call ID with the actual
        # function call ID. This is needed because the function call ID is not
        # known when writing the test case.
        if (
            user_message.content.parts
            and user_message.content.parts[0].function_response
            and user_message.content.parts[0].function_response.name
        ):
          if (
              user_message.content.parts[0].function_response.name
              not in function_call_name_to_id_map
          ):
            raise ValueError(
                "Function response for"
                f" {user_message.content.parts[0].function_response.name} does"
                " not match any pending function call."
            )
          content.parts[0].function_response.id = function_call_name_to_id_map[
              user_message.content.parts[0].function_response.name
          ]
      elif user_message.text is not None:
        content = types.UserContent(parts=[types.Part(text=user_message.text)])
      else:
        raise ValueError(
            f"UserMessage at index {user_message_index} has neither text nor"
            " content"
        )

      request = RunAgentRequest(
          app_name=test_case.test_spec.agent,
          user_id=self.user_id,
          session_id=session_id,
          new_message=content,
          streaming=False,
          state_delta=user_message.state_delta,
      )

      # Run the agent but don't collect events here
      async for event in self.client.run_agent(
          request,
          mode="replay",
          test_case_dir=str(test_case.dir),
          user_message_index=user_message_index,
      ):
        if event.content and event.content.parts:
          for part in event.content.parts:
            if part.function_call:
              function_call_name_to_id_map[part.function_call.name] = (
                  part.function_call.id
              )

  async def _validate_test_results(
      self, session_id: str, test_case: TestCase
  ) -> _TestResult:
    """Validate test results by comparing with recorded data."""
    # Get final session and use its events for comparison
    final_session = await self.client.get_session(
        app_name=test_case.test_spec.agent,
        user_id=self.user_id,
        session_id=session_id,
    )
    if not final_session:
      return _TestResult(
          category=test_case.category,
          name=test_case.name,
          success=False,
          error_message="No final session available for comparison",
      )

    # Load recorded session data for comparison
    recorded_session = load_recorded_session(test_case.dir)
    if not recorded_session:
      return _TestResult(
          category=test_case.category,
          name=test_case.name,
          success=False,
          error_message="No recorded session found for replay comparison",
      )

    # Compare events and session
    events_result = compare_events(
        final_session.events, recorded_session.events
    )
    session_result = compare_session(final_session, recorded_session)

    # Determine overall success
    success = events_result.success and session_result.success
    error_messages = []
    if not events_result.success and events_result.error_message:
      error_messages.append(f"Event mismatch: {events_result.error_message}")
    if not session_result.success and session_result.error_message:
      error_messages.append(f"Session mismatch: {session_result.error_message}")

    return _TestResult(
        category=test_case.category,
        name=test_case.name,
        success=success,
        error_message="\n\n".join(error_messages) if error_messages else None,
    )

  async def _run_test_case_replay(self, test_case: TestCase) -> _TestResult:
    """Run a single test case in replay mode."""
    try:
      # Create session
      session = await self.client.create_session(
          app_name=test_case.test_spec.agent,
          user_id=self.user_id,
          state=test_case.test_spec.initial_state,
      )

      # Run each user message
      try:
        await self._run_user_messages(session.id, test_case)
      except Exception as e:
        return _TestResult(
            category=test_case.category,
            name=test_case.name,
            success=False,
            error_message=f"Replay verification failed: {e}",
        )

      # Validate results and return test result
      result = await self._validate_test_results(session.id, test_case)

      # Clean up session
      await self.client.delete_session(
          app_name=test_case.test_spec.agent,
          user_id=self.user_id,
          session_id=session.id,
      )

      return result

    except Exception as e:
      return _TestResult(
          category=test_case.category,
          name=test_case.name,
          success=False,
          error_message=f"Test setup failed: {e}",
      )

  async def run_all_tests(self) -> _ConformanceTestSummary:
    """Run all discovered test cases."""
    test_cases = self._discover_test_cases()
    if not test_cases:
      click.secho("No test cases found!", fg="yellow", err=True)
      return _ConformanceTestSummary(
          total_tests=0,
          passed_tests=0,
          failed_tests=0,
          results=[],
      )

    click.echo(f"""
Found {len(test_cases)} test cases to run in {self.mode} mode
""")

    results: list[_TestResult] = []
    for test_case in test_cases:
      click.echo(f"Running {test_case.category}/{test_case.name}...", nl=False)
      if self.mode == "replay":
        result = await self._run_test_case_replay(test_case)
      else:
        # TODO: Implement live mode
        result = _TestResult(
            category=test_case.category,
            name=test_case.name,
            success=False,
            error_message="Live mode not yet implemented",
        )
      results.append(result)
      _print_test_case_result(result)

    passed = sum(1 for r in results if r.success)
    return _ConformanceTestSummary(
        total_tests=len(results),
        passed_tests=passed,
        failed_tests=len(results) - passed,
        results=results,
    )


async def run_conformance_test(
    test_paths: list[Path],
    mode: str = "replay",
) -> None:
  """Run conformance tests."""
  _print_test_header(mode)

  async with AdkWebServerClient() as client:
    runner = ConformanceTestRunner(test_paths, client, mode)
    summary = await runner.run_all_tests()

  _print_test_summary(summary)


def _print_test_header(mode: str) -> None:
  """Print the conformance test header."""
  click.echo("=" * 50)
  click.echo(f"Running ADK conformance tests in {mode} mode...")
  click.echo("=" * 50)


def _print_test_case_result(result: _TestResult) -> None:
  """Print the result of a single test case."""
  if result.success:
    click.secho(" ✓ PASS", fg="green")
  else:
    click.secho(" ✗ FAIL", fg="red")
    if result.error_message:
      click.secho(f"Error: {result.error_message}", fg="red", err=True)


def _print_test_result_details(result: _TestResult) -> None:
  """Print detailed information about a failed test result."""
  click.secho(f"\n✗ {result.category}/{result.name}\n", fg="red")
  if result.error_message:
    indented_message = textwrap.indent(result.error_message, "  ")
    click.secho(indented_message, fg="red", err=True)


def _print_test_summary(summary: _ConformanceTestSummary) -> None:
  """Print the conformance test summary results."""
  # Print summary
  click.echo("\n" + "=" * 50)
  click.echo("CONFORMANCE TEST SUMMARY")
  click.echo("=" * 50)

  if summary.total_tests == 0:
    click.secho("No tests were run.", fg="yellow")
    return

  click.echo(f"Total tests: {summary.total_tests}")
  click.secho(f"Passed: {summary.passed_tests}", fg="green")

  if summary.failed_tests > 0:
    click.secho(f"Failed: {summary.failed_tests}", fg="red")
  else:
    click.echo(f"Failed: {summary.failed_tests}")

  click.echo(f"Success rate: {summary.success_rate:.1f}%")

  # List failed tests
  failed_tests = [r for r in summary.results if not r.success]
  if failed_tests:
    click.echo("\nFailed tests:")
    for result in failed_tests:
      _print_test_result_details(result)

  # Exit with error code if any tests failed
  if summary.failed_tests > 0:
    raise click.ClickException(f"{summary.failed_tests} test(s) failed")
  else:
    click.secho("\nAll tests passed! 🎉", fg="green")
