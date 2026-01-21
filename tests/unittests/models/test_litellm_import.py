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

import importlib.util
import os
import subprocess
import sys

import pytest


def _subprocess_env() -> dict[str, str]:
  env = dict(os.environ)
  src_path = os.path.join(os.getcwd(), "src")
  pythonpath = env.get("PYTHONPATH", "")
  env["PYTHONPATH"] = (
      f"{src_path}{os.pathsep}{pythonpath}" if pythonpath else src_path
  )
  return env


def test_importing_models_does_not_import_litellm_or_set_mode():
  env = _subprocess_env()
  env.pop("LITELLM_MODE", None)

  result = subprocess.run(
      [
          sys.executable,
          "-c",
          (
              "import os, sys\n"
              "import google.adk.models\n"
              "print('litellm' in sys.modules)\n"
              "print(os.environ.get('LITELLM_MODE'))\n"
          ),
      ],
      check=True,
      capture_output=True,
      text=True,
      env=env,
  )
  stdout_lines = result.stdout.strip().splitlines()
  assert stdout_lines == ["False", "None"]


def test_ensure_litellm_imported_defaults_to_production():
  if importlib.util.find_spec("litellm") is None:
    pytest.skip("litellm is not installed")

  env = _subprocess_env()
  env.pop("LITELLM_MODE", None)

  result = subprocess.run(
      [
          sys.executable,
          "-c",
          (
              "import os\n"
              "from google.adk.models.lite_llm import"
              " _ensure_litellm_imported\n"
              "_ensure_litellm_imported()\n"
              "print(os.environ.get('LITELLM_MODE'))\n"
          ),
      ],
      check=True,
      capture_output=True,
      text=True,
      env=env,
  )
  assert result.stdout.strip() == "PRODUCTION"


def test_ensure_litellm_imported_does_not_override():
  if importlib.util.find_spec("litellm") is None:
    pytest.skip("litellm is not installed")

  env = _subprocess_env()
  env["LITELLM_MODE"] = "DEV"

  result = subprocess.run(
      [
          sys.executable,
          "-c",
          (
              "import os\n"
              "from google.adk.models.lite_llm import"
              " _ensure_litellm_imported\n"
              "_ensure_litellm_imported()\n"
              "print(os.environ.get('LITELLM_MODE'))\n"
          ),
      ],
      check=True,
      capture_output=True,
      text=True,
      env=env,
  )
  assert result.stdout.strip() == "DEV"
