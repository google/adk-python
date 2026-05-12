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

"""Tests for generate_eval_cases CLI command."""

import json
import os
import pathlib

from click.testing import CliRunner
# We must mock or import the command safely for pytest
from google.adk.cli.cli_tools_click import cli_generate_eval_cases
import pytest


def test_cli_generate_eval_cases_integration(tmp_path):
  """E2E Test for the Vertex AI Scenario Generation Facade via the CLI."""
  # This requires identical project setup to Kokoro's e2e_test_gcp_ubuntu_docker
  if not os.environ.get("GOOGLE_CLOUD_PROJECT"):
    pytest.skip(
        "GOOGLE_CLOUD_PROJECT is not set. Skipping generation CLI integration"
        " test."
    )

  # 1. Provide a UserSimulationConfig proxy
  config_file = tmp_path / "user_sim_config.json"
  config_data = {
      "generation_instruction": (
          "Generate a test conversation scenario where the user asks a simple"
          " question."
      ),
      "count": 1,
      "model_name": "gemini-2.5-flash",
  }
  with open(config_file, "w") as f:
    json.dump(config_data, f)

  eval_set_id = "cli_gen_test_eval_set"

  # 2. Invoke the command via click's testing runner
  runner = CliRunner()
  result = runner.invoke(
      cli_generate_eval_cases,
      [
          str(
              pathlib.Path(__file__).parent
              / "fixture"
              / "home_automation_agent"
          ),
          eval_set_id,
          f"--user_simulation_config_file={config_file}",
          "--log_level=DEBUG",
      ],
  )

  # 3. Assert correct output
  assert (
      result.exit_code == 0
  ), f"Command failed: {result.exception}\nOutput: {result.output}"
  assert "Generating scenarios utilizing Vertex AI Eval SDK..." in result.output
  assert "added to eval set" in result.output
