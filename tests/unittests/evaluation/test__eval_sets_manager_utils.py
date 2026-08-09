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

from google.adk.errors.not_found_error import NotFoundError
from google.adk.evaluation._eval_sets_manager_utils import add_eval_case_to_eval_set
from google.adk.evaluation._eval_sets_manager_utils import delete_eval_case_from_eval_set
from google.adk.evaluation._eval_sets_manager_utils import get_eval_case_from_eval_set
from google.adk.evaluation._eval_sets_manager_utils import get_eval_set_from_app_and_id
from google.adk.evaluation._eval_sets_manager_utils import update_eval_case_in_eval_set
from google.adk.evaluation.eval_case import EvalCase
from google.adk.evaluation.eval_set import EvalSet
from google.adk.evaluation.in_memory_eval_sets_manager import InMemoryEvalSetsManager
import pytest


def _eval_case(eval_id: str, creation_timestamp: float = 0.0) -> EvalCase:
  """Builds a minimal valid EvalCase.

  `creation_timestamp` is only used as a marker so that two cases sharing an
  eval id can still be told apart.
  """
  return EvalCase(
      eval_id=eval_id,
      conversation=[],
      creation_timestamp=creation_timestamp,
  )


def _eval_set(
    eval_cases: list[EvalCase], eval_set_id: str = "set_1"
) -> EvalSet:
  return EvalSet(eval_set_id=eval_set_id, eval_cases=eval_cases)


def _eval_ids(eval_set: EvalSet) -> list[str]:
  return [eval_case.eval_id for eval_case in eval_set.eval_cases]


class TestGetEvalSetFromAppAndId:

  def test_returns_the_eval_set_held_by_the_manager(self):
    manager = InMemoryEvalSetsManager()
    created = manager.create_eval_set("my_app", "set_1")

    assert get_eval_set_from_app_and_id(manager, "my_app", "set_1") is created

  def test_unknown_eval_set_id_raises_not_found_naming_the_id(self):
    manager = InMemoryEvalSetsManager()
    manager.create_eval_set("my_app", "set_1")

    with pytest.raises(NotFoundError, match="Eval set `set_2` not found."):
      get_eval_set_from_app_and_id(manager, "my_app", "set_2")

  def test_eval_set_belonging_to_another_app_is_not_found(self):
    # The lookup is scoped by app name, so an id known under one app must not
    # resolve under a different one.
    manager = InMemoryEvalSetsManager()
    manager.create_eval_set("app_a", "set_1")

    with pytest.raises(NotFoundError, match="Eval set `set_1` not found."):
      get_eval_set_from_app_and_id(manager, "app_b", "set_1")


class TestGetEvalCaseFromEvalSet:

  def test_returns_the_stored_case_object_for_a_known_id(self):
    first = _eval_case("a")
    second = _eval_case("b")
    eval_set = _eval_set([first, second])

    # The caller gets the object that lives in the eval set, not a copy, so
    # that mutating it updates the eval set.
    assert get_eval_case_from_eval_set(eval_set, "b") is second

  def test_returns_none_for_an_unknown_id(self):
    eval_set = _eval_set([_eval_case("a")])

    assert get_eval_case_from_eval_set(eval_set, "b") is None

  def test_returns_none_for_an_empty_eval_set(self):
    assert get_eval_case_from_eval_set(_eval_set([]), "a") is None


class TestAddEvalCaseToEvalSet:

  def test_appends_the_case_and_returns_the_same_eval_set(self):
    eval_set = _eval_set([_eval_case("a")])
    added = _eval_case("b")

    returned = add_eval_case_to_eval_set(eval_set, added)

    # The eval set is mutated in place and handed back.
    assert returned is eval_set
    assert _eval_ids(eval_set) == ["a", "b"]
    assert eval_set.eval_cases[1] is added

  def test_adding_to_an_empty_eval_set_yields_a_single_case(self):
    eval_set = _eval_set([])

    add_eval_case_to_eval_set(eval_set, _eval_case("a"))

    assert _eval_ids(eval_set) == ["a"]

  def test_duplicate_eval_id_raises_value_error_naming_case_and_set(self):
    eval_set = _eval_set([_eval_case("a")], eval_set_id="set_1")

    with pytest.raises(
        ValueError,
        match="Eval id `a` already exists in `set_1` eval set.",
    ):
      add_eval_case_to_eval_set(eval_set, _eval_case("a", 7.0))

  def test_duplicate_eval_id_leaves_the_eval_set_untouched(self):
    eval_set = _eval_set([_eval_case("a", 1.0)])

    with pytest.raises(ValueError):
      add_eval_case_to_eval_set(eval_set, _eval_case("a", 7.0))

    assert _eval_ids(eval_set) == ["a"]
    assert eval_set.eval_cases[0].creation_timestamp == 1.0


class TestUpdateEvalCaseInEvalSet:

  def test_replaces_the_case_carrying_the_same_eval_id(self):
    eval_set = _eval_set([_eval_case("a", 1.0), _eval_case("b", 2.0)])

    returned = update_eval_case_in_eval_set(eval_set, _eval_case("a", 99.0))

    assert returned is eval_set
    # "a" is replaced, "b" is untouched, and no case is added or lost.
    assert sorted(_eval_ids(eval_set)) == ["a", "b"]
    assert get_eval_case_from_eval_set(eval_set, "a").creation_timestamp == 99.0
    assert get_eval_case_from_eval_set(eval_set, "b").creation_timestamp == 2.0

  def test_unknown_eval_id_raises_not_found_naming_case_and_set(self):
    eval_set = _eval_set([_eval_case("a")], eval_set_id="set_1")

    with pytest.raises(
        NotFoundError,
        match="Eval case `zz` not found in eval set `set_1`.",
    ):
      update_eval_case_in_eval_set(eval_set, _eval_case("zz"))

  def test_unknown_eval_id_leaves_the_eval_set_untouched(self):
    eval_set = _eval_set([_eval_case("a", 1.0)])

    with pytest.raises(NotFoundError):
      update_eval_case_in_eval_set(eval_set, _eval_case("zz", 7.0))

    assert _eval_ids(eval_set) == ["a"]
    assert eval_set.eval_cases[0].creation_timestamp == 1.0


class TestDeleteEvalCaseFromEvalSet:

  def test_removes_only_the_named_case_and_keeps_the_others_in_order(self):
    eval_set = _eval_set([_eval_case("a"), _eval_case("b"), _eval_case("c")])

    returned = delete_eval_case_from_eval_set(eval_set, "b")

    assert returned is eval_set
    assert _eval_ids(eval_set) == ["a", "c"]

  def test_deleting_the_only_case_empties_the_eval_set(self):
    eval_set = _eval_set([_eval_case("a")])

    delete_eval_case_from_eval_set(eval_set, "a")

    assert eval_set.eval_cases == []

  def test_unknown_eval_id_raises_not_found_naming_case_and_set(self):
    eval_set = _eval_set([_eval_case("a")], eval_set_id="set_1")

    with pytest.raises(
        NotFoundError,
        match="Eval case `zz` not found in eval set `set_1`.",
    ):
      delete_eval_case_from_eval_set(eval_set, "zz")

  def test_unknown_eval_id_leaves_the_eval_set_untouched(self):
    eval_set = _eval_set([_eval_case("a"), _eval_case("b")])

    with pytest.raises(NotFoundError):
      delete_eval_case_from_eval_set(eval_set, "zz")

    assert _eval_ids(eval_set) == ["a", "b"]

  def test_deleting_an_id_frees_it_up_to_be_added_again(self):
    # Deletion must clear the id entirely, otherwise the duplicate-id guard in
    # add_eval_case_to_eval_set would refuse the re-add.
    eval_set = _eval_set([_eval_case("a", 1.0)])

    delete_eval_case_from_eval_set(eval_set, "a")
    add_eval_case_to_eval_set(eval_set, _eval_case("a", 7.0))

    assert _eval_ids(eval_set) == ["a"]
    assert eval_set.eval_cases[0].creation_timestamp == 7.0
