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

"""Resolution of the ADK telemetry schema version from the environment."""

from __future__ import annotations

from typing import Optional

from google.adk.telemetry._schema_version import ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN
from google.adk.telemetry._schema_version import resolve_schema_version
import pytest


@pytest.mark.parametrize(
    'opt_in,expected',
    [
        (None, 2),
        ('1', 1),
        (' 1\n', 1),
        ('otel_semconv_1_36', 1),
        ('2', 2),
        ('otel_semconv_1_44', 2),
        ('', 2),
        ('bogus', 2),
    ],
    ids=[
        'unset',
        '1',
        '1-padded',
        'semconv-1.36',
        '2',
        'semconv-1.44',
        'empty',
        'unrecognized',
    ],
)
def test_resolve_schema_version(
    monkeypatch: pytest.MonkeyPatch, opt_in: Optional[str], expected: int
):
  if opt_in is None:
    monkeypatch.delenv(ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN, raising=False)
  else:
    monkeypatch.setenv(ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN, opt_in)

  assert resolve_schema_version() == expected
