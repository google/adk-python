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

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import enum
from typing import Protocol


class CallbackType(str, enum.Enum):
  """Lifecycle callback types that receive callback-scoped context metadata."""

  BEFORE_AGENT = 'before_agent'
  AFTER_AGENT = 'after_agent'
  BEFORE_MODEL = 'before_model'
  AFTER_MODEL = 'after_model'
  BEFORE_TOOL = 'before_tool'
  AFTER_TOOL = 'after_tool'
  ON_MODEL_ERROR = 'on_model_error'
  ON_TOOL_ERROR = 'on_tool_error'
  ON_AGENT_ERROR = 'on_agent_error'


@dataclass(frozen=True)
class CallbackInfo:
  """Describes the lifecycle callback currently being invoked.

  Attributes:
    type: The type of the active lifecycle callback.
  """

  type: CallbackType


class _CallbackContext(Protocol):
  _callback_info: CallbackInfo | None


@contextmanager
def _callback_scope(
    callback_type: CallbackType, callback_context: _CallbackContext
) -> Iterator[None]:
  """Sets callback metadata on the context for the duration of a callback."""
  prev = callback_context._callback_info
  callback_context._callback_info = CallbackInfo(type=callback_type)
  try:
    yield
  finally:
    callback_context._callback_info = prev
