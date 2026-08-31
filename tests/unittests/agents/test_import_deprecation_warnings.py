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

import os
import subprocess
import sys


def _subprocess_env() -> dict[str, str]:
  env = dict(os.environ)
  src_path = os.path.join(os.getcwd(), "src")
  pythonpath = env.get("PYTHONPATH", "")
  env["PYTHONPATH"] = (
      f"{src_path}{os.pathsep}{pythonpath}" if pythonpath else src_path
  )
  return env


def test_importing_runtime_agents_does_not_warn_about_agent_config():
  # Regression test for https://github.com/google/adk-python/issues/6968.
  # Run in a fresh subprocess with deprecations promoted to errors, so
  # module caching from other tests cannot hide the warning.
  result = subprocess.run(
      [
          sys.executable,
          "-W",
          "error::DeprecationWarning",
          "-c",
          (
              "from google.adk.agents import LlmAgent, SequentialAgent,"
              " LoopAgent, ParallelAgent"
          ),
      ],
      capture_output=True,
      text=True,
      env=_subprocess_env(),
  )
  assert result.returncode == 0, result.stderr


def test_directly_importing_llm_agent_config_still_warns():
  # Applications that actually use the deprecated Agent Config classes
  # should still be warned.
  result = subprocess.run(
      [
          sys.executable,
          "-W",
          "error::DeprecationWarning",
          "-c",
          "from google.adk.agents import LlmAgentConfig",
      ],
      capture_output=True,
      text=True,
      env=_subprocess_env(),
  )
  assert result.returncode != 0
  assert "DeprecationWarning" in result.stderr
