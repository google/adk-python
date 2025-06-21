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

from unittest.mock import patch

from src.google.adk.models.anthropic_llm import Claude
from src.google.adk.models.google_llm import Gemini


def test_claude_custom_config():
  claude = Claude(project_id="test-project-claude", location="us-central1")

  assert claude.project_id == "test-project-claude"
  assert claude.location == "us-central1"


def test_gemini_custom_config():
  gemini = Gemini(project_id="test-project-gemini", location="europe-west1")

  assert gemini.project_id == "test-project-gemini"
  assert gemini.location == "europe-west1"


def test_claude_per_instance_configuration():
  claude1 = Claude(project_id="project-1", location="us-central1")
  claude2 = Claude(project_id="project-2", location="europe-west1")
  claude3 = Claude()

  assert claude1.project_id == "project-1"
  assert claude1.location == "us-central1"

  assert claude2.project_id == "project-2"
  assert claude2.location == "europe-west1"

  assert claude3.project_id is None
  assert claude3.location is None


def test_gemini_per_instance_configuration():
  gemini1 = Gemini(project_id="project-1", location="us-central1")
  gemini2 = Gemini(project_id="project-2", location="europe-west1")
  gemini3 = Gemini()

  assert gemini1.project_id == "project-1"
  assert gemini1.location == "us-central1"

  assert gemini2.project_id == "project-2"
  assert gemini2.location == "europe-west1"

  assert gemini3.project_id is None
  assert gemini3.location is None


def test_backward_compatibility():
  claude = Claude()
  gemini = Gemini()

  assert claude.project_id is None
  assert claude.location is None
  assert gemini.project_id is None
  assert gemini.location is None


@patch.dict(
    "os.environ",
    {
        "GOOGLE_CLOUD_PROJECT": "env-project",
        "GOOGLE_CLOUD_LOCATION": "env-location",
    },
)
def test_claude_fallback_to_env_vars():
  claude = Claude()

  cache_key = f"{claude.project_id or 'default'}:{claude.location or 'default'}"
  assert cache_key == "default:default"


def test_mixed_configuration():
  claude_custom = Claude(project_id="custom-project", location="us-west1")
  claude_default = Claude()

  key_custom = (
      f"{claude_custom.project_id or 'default'}:{claude_custom.location or 'default'}"
  )
  key_default = (
      f"{claude_default.project_id or 'default'}:{claude_default.location or 'default'}"
  )

  assert key_custom != key_default
  assert key_custom == "custom-project:us-west1"
  assert key_default == "default:default"
