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

from dataclasses import FrozenInstanceError

from google.adk.agents import CallbackHook
from google.adk.agents import CallbackInvocationInfo
from google.adk.agents.callback_context import CallbackHook as ContextCallbackHook
from google.adk.agents.callback_context import CallbackInvocationInfo as ContextCallbackInvocationInfo
import pytest


def test_callback_metadata_is_exported_from_public_agent_modules():
  """Callback metadata has stable public import paths."""
  assert ContextCallbackHook is CallbackHook
  assert ContextCallbackInvocationInfo is CallbackInvocationInfo


def test_callback_hook_values_are_stable():
  """Callback hook values match the documented lifecycle names."""
  assert [hook.value for hook in CallbackHook] == [
      'before_agent',
      'after_agent',
      'before_model',
      'after_model',
      'before_tool',
      'after_tool',
      'on_model_error',
      'on_tool_error',
      'on_agent_error',
  ]


def test_callback_invocation_info_is_immutable():
  """Callback invocation metadata cannot be changed by consumers."""
  callback_info = CallbackInvocationInfo(hook=CallbackHook.BEFORE_MODEL)

  with pytest.raises(FrozenInstanceError):
    callback_info.hook = CallbackHook.AFTER_MODEL
