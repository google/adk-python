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

"""Reading a model's finish reason without trusting the proto3 zero value."""

from __future__ import annotations

from typing import TypeGuard

from google.genai import types


def is_reported_finish_reason(
    finish_reason: types.FinishReason | None,
) -> TypeGuard[types.FinishReason]:
  """Whether the model reported why generation stopped.

  The zero value is truthy, so an unset field otherwise reads as a reason of
  its own, and a turn that ended normally is published as a failed one.
  """
  return (
      finish_reason is not None
      and finish_reason is not types.FinishReason.FINISH_REASON_UNSPECIFIED
  )
