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

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel
from pydantic import ConfigDict


class TestSpec(BaseModel):
  """Test specification for conformance test cases.

  This is the human-authored specification that defines what should be tested.
  Category and name are inferred from folder structure.
  """

  model_config = ConfigDict(
      extra="forbid",
  )

  description: str
  """Human-readable description of what this test validates."""

  agent: str
  """Name of the ADK agent to test against."""

  user_messages: list[str]
  """Sequence of user messages to send to the agent during test execution."""


@dataclass
class TestCase:
  """Represents a single conformance test case."""

  category: str
  name: str
  dir: Path
  test_spec: TestSpec
