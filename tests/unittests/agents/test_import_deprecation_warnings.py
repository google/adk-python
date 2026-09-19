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

# Emitted when a subclass of the deprecated BaseAgentConfig is defined, i.e.
# when any of the *_agent_config modules is imported.
_BASE_AGENT_CONFIG_DEPRECATION = "BaseAgentConfig is deprecated"
# Emitted when the deprecated LlmAgentConfig is instantiated.
_LLM_AGENT_CONFIG_DEPRECATION = "LlmAgentConfig is deprecated"


def _subprocess_env() -> dict[str, str]:
  env = dict(os.environ)
  src_path = os.path.join(os.getcwd(), "src")
  pythonpath = env.get("PYTHONPATH", "")
  env["PYTHONPATH"] = (
      f"{src_path}{os.pathsep}{pythonpath}" if pythonpath else src_path
  )
  return env


def test_importing_runtime_agents_does_not_warn_about_agent_config():
  # Run in a fresh subprocess with every deprecation printed, so module
  # caching from other tests cannot hide the warning.
  result = subprocess.run(
      [
          sys.executable,
          "-W",
          "always::DeprecationWarning",
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
  assert _BASE_AGENT_CONFIG_DEPRECATION not in result.stderr
  # Guards against the assertion above passing because the import blew up.
  assert result.returncode == 0, result.stderr


def test_constructing_llm_agent_config_still_warns():
  # Applications that instantiate the deprecated Agent Config classes
  # should still be warned, even when runtime agents are imported first.
  result = subprocess.run(
      [
          sys.executable,
          "-W",
          "always::DeprecationWarning",
          "-c",
          (
              "from google.adk.agents import LlmAgent, LlmAgentConfig;"
              " LlmAgentConfig(name='test', instruction='test')"
          ),
      ],
      capture_output=True,
      text=True,
      env=_subprocess_env(),
  )
  assert _LLM_AGENT_CONFIG_DEPRECATION in result.stderr
  # Guards against the assertion above passing because construction blew up.
  assert result.returncode == 0, result.stderr
