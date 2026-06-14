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

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
ANSWERING_AGENT = (
    REPO_ROOT / "contributing/samples/adk_team/adk_answering_agent"
)


def test_answering_agents_use_configured_model():
  for relative_path in (
      "agent.py",
      "gemini_assistant/agent.py",
  ):
    source = (ANSWERING_AGENT / relative_path).read_text(encoding="utf-8")

    assert "from adk_answering_agent.settings import LLM_MODEL_NAME" in source
    assert "model=LLM_MODEL_NAME" in source
    assert 'model="gemini-3.5-flash"' not in source


def test_answering_workflow_sets_model_and_location_overrides():
  workflow = (
      REPO_ROOT / ".github/workflows/discussion_answering.yml"
  ).read_text(encoding="utf-8")

  assert (
      "LLM_MODEL_NAME: ${{ vars.ADK_ANSWERING_MODEL || 'gemini-2.5-flash' }}"
      in workflow
  )
  assert (
      "GOOGLE_CLOUD_LOCATION: ${{ vars.ADK_ANSWERING_LOCATION || "
      "secrets.GOOGLE_CLOUD_LOCATION }}"
  ) in workflow
