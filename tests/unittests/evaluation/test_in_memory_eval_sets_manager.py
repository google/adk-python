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

import time

from google.adk.errors.not_found_error import NotFoundError
from google.adk.evaluation.eval_case import EvalCase
from google.adk.evaluation.in_memory_eval_sets_manager import InMemoryEvalSetsManager
import pytest


@pytest.fixture
def app_name():
  return "test_app"


@pytest.fixture
def manager():
  return InMemoryEvalSetsManager()


@pytest.fixture
def eval_set_id():
  return "test_eval_set"


@pytest.fixture
def eval_case_id():
  return "test_eval_case"


def test_create_eval_set(manager, app_name, eval_set_id):
  manager.create_eval_set(app_name, eval_set_id)
  eval_set = manager.get_eval_set(app_name, eval_set_id)
  assert eval_set is not None
  assert eval_set.eval_set_id == eval_set_id
  assert eval_set.eval_cases == []


def test_create_eval_set_already_exists(manager, app_name, eval_set_id):
  manager.create_eval_set(app_name, eval_set_id)
  with pytest.raises(ValueError):
    manager.create_eval_set(app_name, eval_set_id)


def test_get_eval_set(manager, app_name, eval_set_id):
  manager.create_eval_set(app_name, eval_set_id)
  eval_set = manager.get_eval_set(app_name, eval_set_id)
  assert eval_set is not None
  assert eval_set.eval_set_id == eval_set_id


def test_get_eval_set_not_found(manager, app_name):
  eval_set = manager.get_eval_set(app_name, "nonexistent_set")
  assert eval_set is None


def test_get_eval_set_wrong_app(manager, app_name, eval_set_id):
  manager.create_eval_set(app_name, eval_set_id)
  eval_set = manager.get_eval_set("wrong_app", eval_set_id)
  assert eval_set is None


def test_list_eval_sets(manager, app_name):
  manager.create_eval_set(app_name, "set1")
  manager.create_eval_set(app_name, "set2")
  eval_sets = manager.list_eval_sets(app_name)
  assert len(eval_sets) == 2
  assert "set1" in eval_sets
  assert "set2" in eval_sets


def test_list_eval_sets_wrong_app(manager, app_name):
  manager.create_eval_set(app_name, "set1")
  eval_sets = manager.list_eval_sets("wrong_app")
  assert len(eval_sets) == 0


def test_add_eval_case(manager, app_name, eval_set_id, eval_case_id):
  manager.create_eval_set(app_name, eval_set_id)
  eval_case = EvalCase(eval_id=eval_case_id, conversation=[])
  manager.add_eval_case(app_name, eval_set_id, eval_case)

  retrieved_case = manager.get_eval_case(app_name, eval_set_id, eval_case_id)
  assert retrieved_case is not None
  assert retrieved_case.eval_id == eval_case_id

  eval_set = manager.get_eval_set(app_name, eval_set_id)
  assert len(eval_set.eval_cases) == 1
  assert eval_set.eval_cases[0].eval_id == eval_case_id


def test_add_eval_case_set_not_found(
    manager, app_name, eval_set_id, eval_case_id
):
  eval_case = EvalCase(eval_id=eval_case_id, conversation=[])
  with pytest.raises(NotFoundError):
    manager.add_eval_case(app_name, eval_set_id, eval_case)


def test_add_eval_case_already_exists(
    manager, app_name, eval_set_id, eval_case_id
):
  manager.create_eval_set(app_name, eval_set_id)
  eval_case = EvalCase(eval_id=eval_case_id, conversation=[])
  manager.add_eval_case(app_name, eval_set_id, eval_case)
  with pytest.raises(ValueError):
    manager.add_eval_case(app_name, eval_set_id, eval_case)


def test_get_eval_case(manager, app_name, eval_set_id, eval_case_id):
  manager.create_eval_set(app_name, eval_set_id)
  eval_case = EvalCase(eval_id=eval_case_id, conversation=[])
  manager.add_eval_case(app_name, eval_set_id, eval_case)
  retrieved_case = manager.get_eval_case(app_name, eval_set_id, eval_case_id)
  assert retrieved_case is not None
  assert retrieved_case.eval_id == eval_case_id


def test_get_eval_case_not_found(manager, app_name, eval_set_id):
  manager.create_eval_set(app_name, eval_set_id)
  retrieved_case = manager.get_eval_case(
      app_name, eval_set_id, "nonexistent_case"
  )
  assert retrieved_case is None


def test_get_eval_case_set_not_found(manager, app_name, eval_case_id):
  retrieved_case = manager.get_eval_case(
      app_name, "nonexistent_set", eval_case_id
  )
  assert retrieved_case is None


def test_update_eval_case(manager, app_name, eval_set_id, eval_case_id):
  manager.create_eval_set(app_name, eval_set_id)
  eval_case = EvalCase(eval_id=eval_case_id, conversation=[])
  manager.add_eval_case(app_name, eval_set_id, eval_case)

  updated_eval_case = EvalCase(
      eval_id=eval_case_id, conversation=[], creation_timestamp=time.time()
  )
  manager.update_eval_case(app_name, eval_set_id, updated_eval_case)

  retrieved_case = manager.get_eval_case(app_name, eval_set_id, eval_case_id)
  assert retrieved_case is not None
  assert retrieved_case.creation_timestamp != 0.0
  assert (
      retrieved_case.creation_timestamp == updated_eval_case.creation_timestamp
  )

  eval_set = manager.get_eval_set(app_name, eval_set_id)
  assert len(eval_set.eval_cases) == 1
  assert (
      eval_set.eval_cases[0].creation_timestamp
      == updated_eval_case.creation_timestamp
  )


def test_update_eval_case_not_found(
    manager, app_name, eval_set_id, eval_case_id
):
  manager.create_eval_set(app_name, eval_set_id)
  updated_eval_case = EvalCase(eval_id=eval_case_id, conversation=[])
  with pytest.raises(NotFoundError):
    manager.update_eval_case(app_name, eval_set_id, updated_eval_case)


def test_delete_eval_case(manager, app_name, eval_set_id, eval_case_id):
  manager.create_eval_set(app_name, eval_set_id)
  eval_case = EvalCase(eval_id=eval_case_id, conversation=[])
  manager.add_eval_case(app_name, eval_set_id, eval_case)

  manager.delete_eval_case(app_name, eval_set_id, eval_case_id)

  retrieved_case = manager.get_eval_case(app_name, eval_set_id, eval_case_id)
  assert retrieved_case is None

  eval_set = manager.get_eval_set(app_name, eval_set_id)
  assert len(eval_set.eval_cases) == 0


def test_delete_eval_case_not_found(
    manager, app_name, eval_set_id, eval_case_id
):
  manager.create_eval_set(app_name, eval_set_id)
  with pytest.raises(NotFoundError):
    manager.delete_eval_case(app_name, eval_set_id, eval_case_id)
