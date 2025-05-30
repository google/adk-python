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

from typing import Optional

from pydantic import BaseModel

from .eval_case import EvalCase


class EvalSet(BaseModel):
  """A set of eval cases."""

  eval_set_id: str
  """Unique identifier for the eval set."""

  name: Optional[str] = None
  """Name of the dataset."""

  description: Optional[str] = None
  """Description of the dataset."""

  eval_cases: list[EvalCase]
  """List of eval cases in the dataset. Each case represents a single
  interaction to be evaluated."""

  creation_timestamp: float = 0.0
  """The time at which this eval set was created."""
