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

from __future__ import annotations

import re

from google.adk.evaluation.local_eval_service import _get_session_id

# Vertex AI Agent Engine Sessions only accept custom session IDs that match
# `[a-z0-9-]`, with a letter or digit as the first and last character.
# See https://github.com/google/adk-python/issues/6683
_AGENT_ENGINE_SESSION_ID_PATTERN = re.compile(r'^[a-z0-9-]+$')


def test_eval_session_id_matches_agent_engine_constraints():
  # Regression test for https://github.com/google/adk-python/issues/6683
  # The generated eval session ID is passed straight to the configured
  # SessionService (e.g. VertexAiSessionService), which forwards it to remote
  # backends such as Agent Engine as a custom session ID. IDs containing
  # underscores or uppercase letters are rejected by the remote API, so the
  # generated ID must match `[a-z0-9-]` and start/end with an alphanumeric
  # character.
  session_id = _get_session_id()
  assert _AGENT_ENGINE_SESSION_ID_PATTERN.fullmatch(session_id), (
      f'Generated eval session id {session_id!r} must match'
      f' {_AGENT_ENGINE_SESSION_ID_PATTERN.pattern}.'
  )
  assert session_id[0].isalnum()
  assert session_id[-1].isalnum()
