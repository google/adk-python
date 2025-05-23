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

from abc import ABC
from abc import abstractmethod

from .eval_case import EvalCase
from .eval_set import EvalSet


class EvalSetsManager(ABC):
  """An interface to manage an Eval Sets."""

  @abstractmethod
  def get_eval_set(self, app_name: str, eval_set_id: str) -> EvalSet:
    """Returns an EvalSet identified by an app_name and eval_set_id."""
    raise NotImplementedError()

  @abstractmethod
  def create_eval_set(self, app_name: str, eval_set_id: str):
    """Creates an empty EvalSet given the app_name and eval_set_id."""
    raise NotImplementedError()

  @abstractmethod
  def list_eval_sets(self, app_name: str) -> list[str]:
    """Returns a list of EvalSets that belong to the given app_name."""
    raise NotImplementedError()

  @abstractmethod
  def add_eval_case(self, app_name: str, eval_set_id: str, eval_case: EvalCase):
    """Adds the given EvalCase to an existing EvalSet identified by app_name and eval_set_id."""
    raise NotImplementedError()
