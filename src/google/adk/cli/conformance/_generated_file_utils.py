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

"""Loading utilities for conformance testing."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import Optional

import click
import yaml

from ...sessions.session import Session
from .test_case import TestSpec


def load_test_case(test_case_dir: Path) -> TestSpec:
  """Load TestSpec from spec.yaml file."""
  spec_file = test_case_dir / "spec.yaml"
  with open(spec_file, "r", encoding="utf-8") as f:
    data: dict[str, Any] = yaml.safe_load(f)
  return TestSpec.model_validate(data)


def load_recorded_session(test_case_dir: Path) -> Optional[Session]:
  """Load recorded session data from generated-session.yaml file."""
  session_file = test_case_dir / "generated-session.yaml"
  if not session_file.exists():
    return None

  with open(session_file, "r", encoding="utf-8") as f:
    session_data = yaml.safe_load(f)
    if not session_data:
      return None

  try:
    return Session.model_validate(session_data)
  except Exception as e:
    click.secho(
        f"Warning: Failed to parse session data: {e}", fg="yellow", err=True
    )
    return None
