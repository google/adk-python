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

from google.adk.events.event_actions import EventActions
from google.adk.tools.exit_loop_tool import exit_current_loop
from google.adk.tools.exit_loop_tool import exit_loop


class _FakeToolContext:

  def __init__(self):
    self.actions = EventActions()


def test_exit_loop_is_root_scoped():
  ctx = _FakeToolContext()
  exit_loop(ctx)
  assert ctx.actions.escalate is True
  assert ctx.actions.skip_summarization is True
  assert ctx.actions.escalation_context is None


def test_exit_current_loop_is_parent_scoped():
  ctx = _FakeToolContext()
  exit_current_loop(ctx)
  assert ctx.actions.escalate is True
  assert ctx.actions.skip_summarization is True
  assert ctx.actions.escalation_context is not None
  assert ctx.actions.escalation_context.type == 'parent'
