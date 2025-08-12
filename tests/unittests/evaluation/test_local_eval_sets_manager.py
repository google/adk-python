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

import json
import os
import uuid

from google.adk.errors.not_found_error import NotFoundError
from google.adk.evaluation.eval_case import EvalCase
from google.adk.evaluation.eval_case import IntermediateData
from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.eval_set import EvalSet
from google.adk.evaluation.local_eval_sets_manager import _EVAL_SET_FILE_EXTENSION
from google.adk.evaluation.local_eval_sets_manager import convert_eval_set_to_pydanctic_schema
from google.adk.evaluation.local_eval_sets_manager import load_eval_set_from_file
from google.adk.evaluation.local_eval_sets_manager import LocalEvalSetsManager
from google.genai import types as genai_types
from pydantic import ValidationError
import pytest


class TestConvertEvalSetToPydancticSchema:
  """Tests convert_eval_set_to_pydanctic_schema method."""

  def test_convert_eval_set_to_pydanctic_schema_complete(self):
    eval_set_id = "test_eval_set"
    eval_set_in_json_format = [{
        "name": "roll_17_sided_dice_twice",
        "data": [
            {
                "query": "What can you do?",
                "expected_tool_use": [],
                "expected_intermediate_agent_responses": [],
                "reference": (
                    "I can roll dice of different sizes and check if a number"
                    " is prime. I can also use multiple tools in parallel.\n"
                ),
            },
            {
                "query": "Roll a 17 sided dice twice for me",
                "expected_tool_use": [
                    {"tool_name": "roll_die", "tool_input": {"sides": 17}},
                    {"tool_name": "roll_die", "tool_input": {"sides": 17}},
                ],
                "expected_intermediate_agent_responses": [
                    {"author": "agent1", "text": "thought1"}
                ],
                "reference": (
                    "I have rolled a 17 sided die twice. The first roll was 13"
                    " and the second roll was 4.\n"
                ),
            },
        ],
        "initial_session": {
            "state": {},
            "app_name": "hello_world",
            "user_id": "user",
        },
    }]

    eval_set = convert_eval_set_to_pydanctic_schema(
        eval_set_id, eval_set_in_json_format
    )

    assert eval_set.eval_set_id == eval_set_id
    assert len(eval_set.eval_cases) == 1
    assert eval_set.eval_cases[0].eval_id == "roll_17_sided_dice_twice"
    assert len(eval_set.eval_cases[0].conversation) == 2
    assert eval_set.eval_cases[0].session_input.app_name == "hello_world"
    assert (
        len(eval_set.eval_cases[0].conversation[1].intermediate_data.tool_uses)
        == 2
    )
    assert (
        len(
            eval_set.eval_cases[0]
            .conversation[1]
            .intermediate_data.intermediate_responses
        )
        == 1
    )

  def test_convert_eval_set_to_pydanctic_schema_minimal(self):
    eval_set_id = "test_eval_set"
    eval_set_in_json_format = [{
        "name": "minimal_case",
        "data": [{"query": "Hello", "reference": "World"}],
    }]

    eval_set = convert_eval_set_to_pydanctic_schema(
        eval_set_id, eval_set_in_json_format
    )

    assert eval_set.eval_set_id == eval_set_id
    assert len(eval_set.eval_cases) == 1
    assert eval_set.eval_cases[0].eval_id == "minimal_case"
    assert len(eval_set.eval_cases[0].conversation) == 1
    assert (
        eval_set.eval_cases[0].conversation[0].user_content.parts[0].text
        == "Hello"
    )
    assert (
        eval_set.eval_cases[0].conversation[0].final_response.parts[0].text
        == "World"
    )

  def test_convert_eval_set_to_pydanctic_schema_empty_tool_use_and_intermediate_responses(
      self,
  ):
    eval_set_id = "test_eval_set"
    eval_set_in_json_format = [{
        "name": "empty_lists",
        "data": [{
            "query": "Test",
            "reference": "Test Ref",
            "expected_tool_use": [],
            "expected_intermediate_agent_responses": [],
        }],
    }]

    eval_set = convert_eval_set_to_pydanctic_schema(
        eval_set_id, eval_set_in_json_format
    )

    assert eval_set.eval_set_id == eval_set_id
    assert len(eval_set.eval_cases) == 1
    assert (
        len(eval_set.eval_cases[0].conversation[0].intermediate_data.tool_uses)
        == 0
    )
    assert (
        len(
            eval_set.eval_cases[0]
            .conversation[0]
            .intermediate_data.intermediate_responses
        )
        == 0
    )

  def test_convert_eval_set_to_pydanctic_schema_empty_initial_session(self):
    eval_set_id = "test_eval_set"
    eval_set_in_json_format = [{
        "name": "empty_session",
        "data": [{"query": "Test", "reference": "Test Ref"}],
        "initial_session": {},
    }]

    eval_set = convert_eval_set_to_pydanctic_schema(
        eval_set_id, eval_set_in_json_format
    )

    assert eval_set.eval_set_id == eval_set_id
    assert eval_set.eval_cases[0].session_input is None

  def test_convert_eval_set_to_pydanctic_schema_invalid_data(self):
    # This test implicitly checks for potential validation errors during Pydantic
    # object creation
    eval_set_id = "test_eval_set"
    eval_set_in_json_format = [{
        "name": 123,  # Invalid name type
        "data": [{
            "query": 456,  # Invalid query type
            "reference": 789,  # Invalid reference type
            "expected_tool_use": [{
                "tool_name": 123,
                "tool_input": 456,
            }],  # Invalid tool name and input
            "expected_intermediate_agent_responses": [
                {"author": 123, "text": 456}  # Invalid author and text
            ],
        }],
        "initial_session": {
            "state": "invalid",  # Invalid state type
            "app_name": 123,  # Invalid app_name type
            "user_id": 456,  # Invalid user_id type
        },
    }]

    with pytest.raises(ValidationError):
      convert_eval_set_to_pydanctic_schema(eval_set_id, eval_set_in_json_format)


class TestLoadEvalSetFromFile:
  """Tests for load_eval_set_from_file method."""

  def test_load_eval_set_from_file_new_format(self, tmp_path):
    # Create a dummy file with EvalSet in the new Pydantic JSON format
    eval_set = EvalSet(
        eval_set_id="new_format_eval_set",
        eval_cases=[
            EvalCase(
                eval_id="new_format_case",
                conversation=[
                    Invocation(
                        invocation_id=str(uuid.uuid4()),
                        user_content=genai_types.Content(
                            parts=[genai_types.Part(text="New Format Query")]
                        ),
                        final_response=genai_types.Content(
                            parts=[
                                genai_types.Part(text="New Format Reference")
                            ]
                        ),
                    )
                ],
            )
        ],
    )
    file_path = tmp_path / "new_format.json"
    with open(file_path, "w", encoding="utf-8") as f:
      f.write(eval_set.model_dump_json())

    loaded_eval_set = load_eval_set_from_file(
        str(file_path), "new_format_eval_set"
    )

    assert loaded_eval_set == eval_set

  def test_load_eval_set_from_file_old_format(self, tmp_path, mocker):
    mocked_time = 12345678
    mocked_invocation_id = "15061953"
    mocker.patch("time.time", return_value=mocked_time)
    mocker.patch("uuid.uuid4", return_value=mocked_invocation_id)

    # Create a dummy file with EvalSet in the old JSON format
    old_format_json = [{
        "name": "old_format_case",
        "data": [
            {"query": "Old Format Query", "reference": "Old Format Reference"}
        ],
    }]
    file_path = tmp_path / "old_format.json"
    with open(file_path, "w", encoding="utf-8") as f:
      json.dump(old_format_json, f)

    loaded_eval_set = load_eval_set_from_file(
        str(file_path), "old_format_eval_set"
    )

    expected_eval_set = EvalSet(
        eval_set_id="old_format_eval_set",
        name="old_format_eval_set",
        creation_timestamp=mocked_time,
        eval_cases=[
            EvalCase(
                eval_id="old_format_case",
                creation_timestamp=mocked_time,
                conversation=[
                    Invocation(
                        invocation_id=mocked_invocation_id,
                        user_content=genai_types.Content(
                            parts=[genai_types.Part(text="Old Format Query")],
                            role="user",
                        ),
                        final_response=genai_types.Content(
                            parts=[
                                genai_types.Part(text="Old Format Reference")
                            ],
                            role="model",
                        ),
                        intermediate_data=IntermediateData(
                            tool_uses=[],
                            intermediate_responses=[],
                        ),
                        creation_timestamp=mocked_time,
                    )
                ],
            )
        ],
    )

    assert loaded_eval_set == expected_eval_set

  def test_load_eval_set_from_file_nonexistent_file(self):
    with pytest.raises(FileNotFoundError):
      load_eval_set_from_file("nonexistent_file.json", "test_eval_set")

  def test_load_eval_set_from_file_invalid_json(self, tmp_path):
    # Create a dummy file with invalid JSON
    file_path = tmp_path / "invalid.json"
    with open(file_path, "w", encoding="utf-8") as f:
      f.write("invalid json")

    with pytest.raises(json.JSONDecodeError):
      load_eval_set_from_file(str(file_path), "test_eval_set")

  def test_load_eval_set_from_file_invalid_data(self, tmp_path, mocker):
    # Create a dummy file with invalid data that fails both Pydantic validation
    # and the old format conversion.  We mock the
    # convert_eval_set_to_pydanctic_schema function to raise a ValueError
    # so that we can assert that the exception is raised.
    file_path = tmp_path / "invalid_data.json"
    with open(file_path, "w", encoding="utf-8") as f:
      f.write('{"invalid": "data"}')

    mocker.patch(
        "google.adk.evaluation.local_eval_sets_manager.convert_eval_set_to_pydanctic_schema",
        side_effect=ValueError(),
    )

    with pytest.raises(ValueError):
      load_eval_set_from_file(str(file_path), "test_eval_set")


class TestLocalEvalSetsManager:
  """Tests for LocalEvalSetsManager."""

  @pytest.fixture
  def local_eval_sets_manager(tmp_path):
    agents_dir = str(tmp_path)
    return LocalEvalSetsManager(agents_dir=agents_dir)

  def test_local_eval_sets_manager_get_eval_set_success(
      self, local_eval_sets_manager, mocker
  ):
    app_name = "test_app"
    eval_set_id = "test_eval_set"
    mock_eval_set = EvalSet(eval_set_id=eval_set_id, eval_cases=[])
    mocker.patch(
        "google.adk.evaluation.local_eval_sets_manager.load_eval_set_from_file",
        return_value=mock_eval_set,
    )
    mocker.patch("os.path.exists", return_value=True)

    eval_set = local_eval_sets_manager.get_eval_set(app_name, eval_set_id)

    assert eval_set == mock_eval_set

  def test_local_eval_sets_manager_get_eval_set_not_found(
      self, local_eval_sets_manager, mocker
  ):
    app_name = "test_app"
    eval_set_id = "test_eval_set"
    mocker.patch(
        "google.adk.evaluation.local_eval_sets_manager.load_eval_set_from_file",
        side_effect=FileNotFoundError,
    )

    eval_set = local_eval_sets_manager.get_eval_set(app_name, eval_set_id)

    assert eval_set is None

  def test_local_eval_sets_manager_create_eval_set_success(
      self, local_eval_sets_manager, mocker
  ):
    mocked_time = 12345678
    mocker.patch("time.time", return_value=mocked_time)
    app_name = "test_app"
    eval_set_id = "test_eval_set"
    mocker.patch("os.path.exists", return_value=False)
    mock_write_eval_set_to_path = mocker.patch(
        "google.adk.evaluation.local_eval_sets_manager.LocalEvalSetsManager._write_eval_set_to_path"
    )
    eval_set_file_path = os.path.join(
        local_eval_sets_manager._agents_dir,
        app_name,
        eval_set_id + _EVAL_SET_FILE_EXTENSION,
    )

    local_eval_sets_manager.create_eval_set(app_name, eval_set_id)
    mock_write_eval_set_to_path.assert_called_once_with(
        eval_set_file_path,
        EvalSet(
            eval_set_id=eval_set_id,
            name=eval_set_id,
            eval_cases=[],
            creation_timestamp=mocked_time,
        ),
    )

  def test_local_eval_sets_manager_create_eval_set_invalid_id(
      self, local_eval_sets_manager
  ):
    app_name = "test_app"
    eval_set_id = "invalid-id"

    with pytest.raises(ValueError, match="Invalid Eval Set Id"):
      local_eval_sets_manager.create_eval_set(app_name, eval_set_id)

  def test_local_eval_sets_manager_list_eval_sets_success(
      self, local_eval_sets_manager, mocker
  ):
    app_name = "test_app"
    mock_listdir_return = [
        "eval_set_1.evalset.json",
        "eval_set_2.evalset.json",
        "not_an_eval_set.txt",
    ]
    mocker.patch("os.listdir", return_value=mock_listdir_return)
    mocker.patch("os.path.join", return_value="dummy_path")
    mocker.patch("os.path.basename", side_effect=lambda x: x)

    eval_sets = local_eval_sets_manager.list_eval_sets(app_name)

    assert eval_sets == ["eval_set_1", "eval_set_2"]

  def test_local_eval_sets_manager_list_eval_sets_not_found(
      self, local_eval_sets_manager, mocker
  ):
    app_name = "test_app"
    mocker.patch("os.listdir", side_effect=FileNotFoundError)

    with pytest.raises(NotFoundError):
      local_eval_sets_manager.list_eval_sets(app_name)

  def test_local_eval_sets_manager_add_eval_case_success(
      self, local_eval_sets_manager, mocker
  ):
    app_name = "test_app"
    eval_set_id = "test_eval_set"
    eval_case_id = "test_eval_case"
    mock_eval_case = EvalCase(eval_id=eval_case_id, conversation=[])
    mock_eval_set = EvalSet(eval_set_id=eval_set_id, eval_cases=[])

    mocker.patch(
        "google.adk.evaluation.local_eval_sets_manager.LocalEvalSetsManager.get_eval_set",
        return_value=mock_eval_set,
    )
    mock_write_eval_set_to_path = mocker.patch(
        "google.adk.evaluation.local_eval_sets_manager.LocalEvalSetsManager._write_eval_set_to_path"
    )

    local_eval_sets_manager.add_eval_case(app_name, eval_set_id, mock_eval_case)

    assert len(mock_eval_set.eval_cases) == 1
    assert mock_eval_set.eval_cases[0] == mock_eval_case
    expected_eval_set_file_path = os.path.join(
        local_eval_sets_manager._agents_dir,
        app_name,
        eval_set_id + _EVAL_SET_FILE_EXTENSION,
    )
    mock_eval_set.eval_cases.append(mock_eval_case)
    mock_write_eval_set_to_path.assert_called_once_with(
        expected_eval_set_file_path, mock_eval_set
    )

  def test_local_eval_sets_manager_add_eval_case_eval_set_not_found(
      self, local_eval_sets_manager, mocker
  ):
    app_name = "test_app"
    eval_set_id = "test_eval_set"
    eval_case_id = "test_eval_case"
    mock_eval_case = EvalCase(eval_id=eval_case_id, conversation=[])

    mocker.patch(
        "google.adk.evaluation.local_eval_sets_manager.LocalEvalSetsManager.get_eval_set",
        return_value=None,
    )

    with pytest.raises(
        NotFoundError, match="Eval set `test_eval_set` not found."
    ):
      local_eval_sets_manager.add_eval_case(
          app_name, eval_set_id, mock_eval_case
      )

  def test_local_eval_sets_manager_add_eval_case_eval_case_id_exists(
      self, local_eval_sets_manager, mocker
  ):
    app_name = "test_app"
    eval_set_id = "test_eval_set"
    eval_case_id = "test_eval_case"
    mock_eval_case = EvalCase(eval_id=eval_case_id, conversation=[])
    mock_eval_set = EvalSet(
        eval_set_id=eval_set_id, eval_cases=[mock_eval_case]
    )

    mocker.patch(
        "google.adk.evaluation.local_eval_sets_manager.LocalEvalSetsManager.get_eval_set",
        return_value=mock_eval_set,
    )

    with pytest.raises(
        ValueError,
        match=(
            f"Eval id `{eval_case_id}` already exists in `{eval_set_id}` eval"
            " set."
        ),
    ):
      local_eval_sets_manager.add_eval_case(
          app_name, eval_set_id, mock_eval_case
      )

  def test_local_eval_sets_manager_get_eval_case_success(
      self, local_eval_sets_manager, mocker
  ):
    app_name = "test_app"
    eval_set_id = "test_eval_set"
    eval_case_id = "test_eval_case"
    mock_eval_case = EvalCase(eval_id=eval_case_id, conversation=[])
    mock_eval_set = EvalSet(
        eval_set_id=eval_set_id, eval_cases=[mock_eval_case]
    )

    mocker.patch(
        "google.adk.evaluation.local_eval_sets_manager.LocalEvalSetsManager.get_eval_set",
        return_value=mock_eval_set,
    )

    eval_case = local_eval_sets_manager.get_eval_case(
        app_name, eval_set_id, eval_case_id
    )

    assert eval_case == mock_eval_case

  def test_local_eval_sets_manager_get_eval_case_eval_set_not_found(
      self, local_eval_sets_manager, mocker
  ):
    app_name = "test_app"
    eval_set_id = "test_eval_set"
    eval_case_id = "test_eval_case"

    mocker.patch(
        "google.adk.evaluation.local_eval_sets_manager.LocalEvalSetsManager.get_eval_set",
        return_value=None,
    )

    eval_case = local_eval_sets_manager.get_eval_case(
        app_name, eval_set_id, eval_case_id
    )

    assert eval_case is None

  def test_local_eval_sets_manager_get_eval_case_eval_case_not_found(
      self, local_eval_sets_manager, mocker
  ):
    app_name = "test_app"
    eval_set_id = "test_eval_set"
    eval_case_id = "test_eval_case"
    mock_eval_set = EvalSet(eval_set_id=eval_set_id, eval_cases=[])

    mocker.patch(
        "google.adk.evaluation.local_eval_sets_manager.LocalEvalSetsManager.get_eval_set",
        return_value=mock_eval_set,
    )

    eval_case = local_eval_sets_manager.get_eval_case(
        app_name, eval_set_id, eval_case_id
    )

    assert eval_case is None

  def test_local_eval_sets_manager_update_eval_case_success(
      self, local_eval_sets_manager, mocker
  ):
    app_name = "test_app"
    eval_set_id = "test_eval_set"
    eval_case_id = "test_eval_case"
    mock_eval_case = EvalCase(
        eval_id=eval_case_id, conversation=[], creation_timestamp=456
    )
    updated_eval_case = EvalCase(
        eval_id=eval_case_id, conversation=[], creation_timestamp=123
    )
    mock_eval_set = EvalSet(
        eval_set_id=eval_set_id, eval_cases=[mock_eval_case]
    )

    mocker.patch(
        "google.adk.evaluation.local_eval_sets_manager.LocalEvalSetsManager.get_eval_set",
        return_value=mock_eval_set,
    )
    mocker.patch(
        "google.adk.evaluation.local_eval_sets_manager.LocalEvalSetsManager.get_eval_case",
        return_value=mock_eval_case,
    )
    mock_write_eval_set_to_path = mocker.patch(
        "google.adk.evaluation.local_eval_sets_manager.LocalEvalSetsManager._write_eval_set_to_path"
    )

    local_eval_sets_manager.update_eval_case(
        app_name, eval_set_id, updated_eval_case
    )

    assert len(mock_eval_set.eval_cases) == 1
    assert mock_eval_set.eval_cases[0] == updated_eval_case
    expected_eval_set_file_path = os.path.join(
        local_eval_sets_manager._agents_dir,
        app_name,
        eval_set_id + _EVAL_SET_FILE_EXTENSION,
    )
    mock_write_eval_set_to_path.assert_called_once_with(
        expected_eval_set_file_path,
        EvalSet(eval_set_id=eval_set_id, eval_cases=[updated_eval_case]),
    )

  def test_local_eval_sets_manager_update_eval_case_eval_set_not_found(
      self, local_eval_sets_manager, mocker
  ):
    app_name = "test_app"
    eval_set_id = "test_eval_set"
    eval_case_id = "test_eval_case"
    updated_eval_case = EvalCase(eval_id=eval_case_id, conversation=[])

    mocker.patch(
        "google.adk.evaluation.local_eval_sets_manager.LocalEvalSetsManager.get_eval_case",
        return_value=None,
    )

    with pytest.raises(
        NotFoundError,
        match=f"Eval set `{eval_set_id}` not found.",
    ):
      local_eval_sets_manager.update_eval_case(
          app_name, eval_set_id, updated_eval_case
      )

  def test_local_eval_sets_manager_update_eval_case_eval_case_not_found(
      self, local_eval_sets_manager, mocker
  ):
    app_name = "test_app"
    eval_set_id = "test_eval_set"
    eval_case_id = "test_eval_case"
    updated_eval_case = EvalCase(eval_id=eval_case_id, conversation=[])
    mock_eval_set = EvalSet(eval_set_id=eval_set_id, eval_cases=[])
    mocker.patch(
        "google.adk.evaluation.local_eval_sets_manager.LocalEvalSetsManager.get_eval_set",
        return_value=mock_eval_set,
    )
    mocker.patch(
        "google.adk.evaluation.local_eval_sets_manager.LocalEvalSetsManager.get_eval_case",
        return_value=None,
    )
    with pytest.raises(
        NotFoundError,
        match=(
            f"Eval case `{eval_case_id}` not found in eval set `{eval_set_id}`."
        ),
    ):
      local_eval_sets_manager.update_eval_case(
          app_name, eval_set_id, updated_eval_case
      )

  def test_local_eval_sets_manager_delete_eval_case_success(
      self, local_eval_sets_manager, mocker
  ):
    app_name = "test_app"
    eval_set_id = "test_eval_set"
    eval_case_id = "test_eval_case"
    mock_eval_case = EvalCase(eval_id=eval_case_id, conversation=[])
    mock_eval_set = EvalSet(
        eval_set_id=eval_set_id, eval_cases=[mock_eval_case]
    )

    mocker.patch(
        "google.adk.evaluation.local_eval_sets_manager.LocalEvalSetsManager.get_eval_set",
        return_value=mock_eval_set,
    )
    mocker.patch(
        "google.adk.evaluation.local_eval_sets_manager.LocalEvalSetsManager.get_eval_case",
        return_value=mock_eval_case,
    )
    mock_write_eval_set_to_path = mocker.patch(
        "google.adk.evaluation.local_eval_sets_manager.LocalEvalSetsManager._write_eval_set_to_path"
    )

    local_eval_sets_manager.delete_eval_case(
        app_name, eval_set_id, eval_case_id
    )

    assert len(mock_eval_set.eval_cases) == 0
    expected_eval_set_file_path = os.path.join(
        local_eval_sets_manager._agents_dir,
        app_name,
        eval_set_id + _EVAL_SET_FILE_EXTENSION,
    )
    mock_write_eval_set_to_path.assert_called_once_with(
        expected_eval_set_file_path,
        EvalSet(eval_set_id=eval_set_id, eval_cases=[]),
    )

  def test_local_eval_sets_manager_delete_eval_case_eval_set_not_found(
      self, local_eval_sets_manager, mocker
  ):
    app_name = "test_app"
    eval_set_id = "test_eval_set"
    eval_case_id = "test_eval_case"

    mocker.patch(
        "google.adk.evaluation.local_eval_sets_manager.LocalEvalSetsManager.get_eval_case",
        return_value=None,
    )
    mock_write_eval_set_to_path = mocker.patch(
        "google.adk.evaluation.local_eval_sets_manager.LocalEvalSetsManager._write_eval_set_to_path"
    )

    with pytest.raises(
        NotFoundError,
        match=f"Eval set `{eval_set_id}` not found.",
    ):
      local_eval_sets_manager.delete_eval_case(
          app_name, eval_set_id, eval_case_id
      )

    mock_write_eval_set_to_path.assert_not_called()

  def test_local_eval_sets_manager_delete_eval_case_eval_case_not_found(
      self, local_eval_sets_manager, mocker
  ):
    app_name = "test_app"
    eval_set_id = "test_eval_set"
    eval_case_id = "test_eval_case"
    mock_eval_set = EvalSet(eval_set_id=eval_set_id, eval_cases=[])
    mocker.patch(
        "google.adk.evaluation.local_eval_sets_manager.LocalEvalSetsManager.get_eval_set",
        return_value=mock_eval_set,
    )
    mocker.patch(
        "google.adk.evaluation.local_eval_sets_manager.LocalEvalSetsManager.get_eval_case",
        return_value=None,
    )
    with pytest.raises(
        NotFoundError,
        match=(
            f"Eval case `{eval_case_id}` not found in eval set `{eval_set_id}`."
        ),
    ):
      local_eval_sets_manager.delete_eval_case(
          app_name, eval_set_id, eval_case_id
      )
