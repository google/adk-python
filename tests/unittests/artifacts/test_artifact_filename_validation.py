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

"""Tests for artifact filename validation."""

from google.adk.artifacts import artifact_util
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.errors.input_validation_error import InputValidationError
from google.genai import types
import pytest


INVALID_FILENAMES = ["", "   ", "..", "../evil", "images/../../evil", "/abs/path", "C:evil", "a\x00b"]


@pytest.mark.parametrize("filename", INVALID_FILENAMES)
def test_invalid_filenames_are_rejected(filename: str) -> None:
  with pytest.raises(InputValidationError):
    artifact_util.validate_artifact_filename(filename)


@pytest.mark.parametrize("filename", ["report.txt", "images/chart.png", "user:notes.md"])
def test_valid_filenames_pass(filename: str) -> None:
  artifact_util.validate_artifact_filename(filename)


@pytest.mark.asyncio
@pytest.mark.parametrize("filename", ["", "../evil", "/abs/path"])
async def test_in_memory_save_rejects_invalid_filename(filename: str) -> None:
  service = InMemoryArtifactService()

  with pytest.raises(InputValidationError):
    await service.save_artifact(
        app_name="app",
        user_id="user",
        session_id="session",
        filename=filename,
        artifact=types.Part(text="data"),
    )


@pytest.mark.asyncio
async def test_in_memory_save_keeps_nested_filenames() -> None:
  service = InMemoryArtifactService()

  version = await service.save_artifact(
      app_name="app",
      user_id="user",
      session_id="session",
      filename="images/chart.png",
      artifact=types.Part(text="data"),
  )

  assert version == 0
