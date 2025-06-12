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

from abc import ABC
from abc import abstractmethod
from typing import Optional

from ..errors.not_found_error import NotFoundError
from .eval_case import EvalCase
from .eval_set import EvalSet


class EvalSetsManager(ABC):
  """An interface to manage an Eval Sets."""

  @abstractmethod
  def get_eval_set(self, app_name: str, eval_set_id: str) -> Optional[EvalSet]:
    """Returns an EvalSet identified by an app_name and eval_set_id."""

  @abstractmethod
  def create_eval_set(self, app_name: str, eval_set_id: str):
    """Creates an empty EvalSet given the app_name and eval_set_id."""

  @abstractmethod
  def list_eval_sets(self, app_name: str) -> list[str]:
    """Returns a list of EvalSets that belong to the given app_name."""

  @abstractmethod
  def get_eval_case(
      self, app_name: str, eval_set_id: str, eval_case_id: str
  ) -> Optional[EvalCase]:
    """Returns an EvalCase if found, otherwise None."""

  @abstractmethod
  def add_eval_case(self, app_name: str, eval_set_id: str, eval_case: EvalCase):
    """Adds the given EvalCase to an existing EvalSet identified by app_name and eval_set_id.

    Raises:
      NotFoundError: If the eval set is not found.
    """

  @abstractmethod
  def update_eval_case(
      self, app_name: str, eval_set_id: str, updated_eval_case: EvalCase
  ):
    """Updates an existing EvalCase give the app_name and eval_set_id.

    Raises:
      NotFoundError: If the eval set or the eval case is not found.
    """

  @abstractmethod
  def delete_eval_case(
      self, app_name: str, eval_set_id: str, eval_case_id: str
  ):
    """Deletes the given EvalCase identified by app_name, eval_set_id and eval_case_id.

    Raises:
      NotFoundError: If the eval set or the eval case to delete is not found.
    """
