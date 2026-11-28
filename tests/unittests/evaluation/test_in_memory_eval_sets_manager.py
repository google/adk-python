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

import time

from google.adk.errors.not_found_error import NotFoundError
from google.adk.evaluation.eval_case import EvalCase
from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.in_memory_eval_sets_manager import InMemoryEvalSetsManager
from google.genai import types
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
  eval_set = manager.create_eval_set(app_name, eval_set_id)
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


def test_create_eval_set_returns_independent_data(
    manager, app_name, eval_set_id
):
  """Editing the created set cannot add cases outside the manager's API."""
  created = manager.create_eval_set(app_name, eval_set_id)

  created.eval_cases.append(EvalCase(eval_id="untracked", conversation=[]))

  assert manager.get_eval_set(app_name, eval_set_id).eval_cases == []
  assert manager.get_eval_case(app_name, eval_set_id, "untracked") is None


@pytest.mark.parametrize("operation", ["add", "update"])
def test_stored_eval_case_is_independent_of_input(
    manager, app_name, eval_set_id, eval_case_id, operation
):
  """Editing a submitted case cannot change stored IDs or nested conversation data."""
  manager.create_eval_set(app_name, eval_set_id)
  case = EvalCase(
      eval_id=eval_case_id,
      conversation=[
          Invocation(
              user_content=types.Content(
                  role="user", parts=[types.Part(text="original prompt")]
              )
          )
      ],
  )
  if operation == "add":
    manager.add_eval_case(app_name, eval_set_id, case)
  else:
    manager.add_eval_case(
        app_name, eval_set_id, EvalCase(eval_id=eval_case_id, conversation=[])
    )
    manager.update_eval_case(app_name, eval_set_id, case)

  case.eval_id = "changed_id"
  case.conversation[0].user_content.parts[0].text = "changed prompt"

  stored = manager.get_eval_case(app_name, eval_set_id, eval_case_id)
  assert stored.eval_id == eval_case_id
  assert stored.conversation[0].user_content.parts[0].text == "original prompt"
  assert manager.get_eval_set(app_name, eval_set_id).eval_cases == [stored]


@pytest.mark.parametrize("read_set", [False, True])
def test_retrieved_eval_case_is_independent_of_storage(
    manager, app_name, eval_set_id, eval_case_id, read_set
):
  """Editing either read result cannot invalidate a stored case's index."""
  manager.create_eval_set(app_name, eval_set_id)
  manager.add_eval_case(
      app_name,
      eval_set_id,
      EvalCase(
          eval_id=eval_case_id,
          conversation=[],
          final_session_state={"items": [1]},
      ),
  )
  case = (
      manager.get_eval_set(app_name, eval_set_id).eval_cases[0]
      if read_set
      else manager.get_eval_case(app_name, eval_set_id, eval_case_id)
  )

  case.eval_id = "changed_id"
  case.final_session_state["items"].append(2)

  stored = manager.get_eval_case(app_name, eval_set_id, eval_case_id)
  assert stored.eval_id == eval_case_id
  assert stored.final_session_state == {"items": [1]}
  assert manager.get_eval_set(app_name, eval_set_id).eval_cases == [stored]


def test_clearing_retrieved_cases_does_not_delete_stored_cases(
    manager, app_name, eval_set_id, eval_case_id
):
  """Clearing a returned list leaves set reads and case reads consistent."""
  manager.create_eval_set(app_name, eval_set_id)
  manager.add_eval_case(
      app_name, eval_set_id, EvalCase(eval_id=eval_case_id, conversation=[])
  )
  retrieved = manager.get_eval_set(app_name, eval_set_id)

  retrieved.eval_cases.clear()

  stored = manager.get_eval_case(app_name, eval_set_id, eval_case_id)
  assert manager.get_eval_set(app_name, eval_set_id).eval_cases == [stored]
