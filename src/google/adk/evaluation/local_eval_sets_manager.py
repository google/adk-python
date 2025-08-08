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
import logging
import os
import re
import time
from typing import Any
from typing import Optional
import uuid

from google.genai import types as genai_types
from pydantic import ValidationError
from typing_extensions import override

from ..errors.not_found_error import NotFoundError
from ._eval_sets_manager_utils import add_eval_case_to_eval_set
from ._eval_sets_manager_utils import delete_eval_case_from_eval_set
from ._eval_sets_manager_utils import get_eval_case_from_eval_set
from ._eval_sets_manager_utils import get_eval_set_from_app_and_id
from ._eval_sets_manager_utils import update_eval_case_in_eval_set
from .eval_case import EvalCase
from .eval_case import IntermediateData
from .eval_case import Invocation
from .eval_case import SessionInput
from .eval_set import EvalSet
from .eval_sets_manager import EvalSetsManager

logger = logging.getLogger("google_adk." + __name__)

_EVAL_SET_FILE_EXTENSION = ".evalset.json"


def _convert_invocation_to_pydantic_schema(
    invocation_in_json_format: dict[str, Any],
) -> Invocation:
  """Converts an invocation from old json format to new Pydantic Schema."""
  query = invocation_in_json_format["query"]
  reference = invocation_in_json_format.get("reference", "")
  expected_tool_use = []
  expected_intermediate_agent_responses = []

  for old_tool_use in invocation_in_json_format.get("expected_tool_use", []):
    expected_tool_use.append(
        genai_types.FunctionCall(
            name=old_tool_use["tool_name"], args=old_tool_use["tool_input"]
        )
    )

  for old_intermediate_response in invocation_in_json_format.get(
      "expected_intermediate_agent_responses", []
  ):
    expected_intermediate_agent_responses.append((
        old_intermediate_response["author"],
        [genai_types.Part.from_text(text=old_intermediate_response["text"])],
    ))

  return Invocation(
      invocation_id=str(uuid.uuid4()),
      user_content=genai_types.Content(
          parts=[genai_types.Part.from_text(text=query)], role="user"
      ),
      final_response=genai_types.Content(
          parts=[genai_types.Part.from_text(text=reference)], role="model"
      ),
      intermediate_data=IntermediateData(
          tool_uses=expected_tool_use,
          intermediate_responses=expected_intermediate_agent_responses,
      ),
      creation_timestamp=time.time(),
  )


def convert_eval_set_to_pydanctic_schema(
    eval_set_id: str,
    eval_set_in_json_format: list[dict[str, Any]],
) -> EvalSet:
  r"""Returns an pydantic EvalSet generated from the json representation.

    Args:
      eval_set_id: Eval set id.
      eval_set_in_json_format: Eval set specified in JSON format.

    Here is a sample eval set in JSON format:
  [
    {
      "name": "roll_17_sided_dice_twice",
      "data": [
        {
          "query": "What can you do?",
          "expected_tool_use": [],
          "expected_intermediate_agent_responses": [],
          "reference": "I can roll dice of different sizes and check if a number
            is prime. I can also use multiple tools in parallel.\n"
        },
        {
          "query": "Roll a 17 sided dice twice for me",
          "expected_tool_use": [
            {
              "tool_name": "roll_die",
              "tool_input": {
                "sides": 17
              }
            },
            {
              "tool_name": "roll_die",
              "tool_input": {
                "sides": 17
              }
            }
          ],
          "expected_intermediate_agent_responses": [],
          "reference": "I have rolled a 17 sided die twice. The first roll was
            13 and the second roll was 4.\n"
        }
      ],
      "initial_session": {
        "state": {},
        "app_name": "hello_world",
        "user_id": "user"
      }
    }
  ]
  """
  eval_cases = []
  for old_eval_case in eval_set_in_json_format:
    new_invocations = []

    for old_invocation in old_eval_case["data"]:
      new_invocations.append(
          _convert_invocation_to_pydantic_schema(old_invocation)
      )

    session_input = None
    if (
        "initial_session" in old_eval_case
        and len(old_eval_case["initial_session"]) > 0
    ):
      session_input = SessionInput(
          app_name=old_eval_case["initial_session"].get("app_name", ""),
          user_id=old_eval_case["initial_session"].get("user_id", ""),
          state=old_eval_case["initial_session"].get("state", {}),
      )

    new_eval_case = EvalCase(
        eval_id=old_eval_case["name"],
        conversation=new_invocations,
        session_input=session_input,
        creation_timestamp=time.time(),
    )
    eval_cases.append(new_eval_case)

  return EvalSet(
      eval_set_id=eval_set_id,
      name=eval_set_id,
      creation_timestamp=time.time(),
      eval_cases=eval_cases,
  )


def load_eval_set_from_file(
    eval_set_file_path: str, eval_set_id: str
) -> EvalSet:
  """Returns an EvalSet that is read from the given file."""
  with open(eval_set_file_path, "r", encoding="utf-8") as f:
    content = f.read()
    try:
      return EvalSet.model_validate_json(content)
    except ValidationError:
      # We assume that the eval data was specified in the old format and try
      # to convert it to the new format.
      return convert_eval_set_to_pydanctic_schema(
          eval_set_id, json.loads(content)
      )


class LocalEvalSetsManager(EvalSetsManager):
  """An EvalSets manager that stores eval sets locally on disk."""

  def __init__(self, agents_dir: str):
    self._agents_dir = agents_dir

  @override
  def get_eval_set(self, app_name: str, eval_set_id: str) -> Optional[EvalSet]:
    """Returns an EvalSet identified by an app_name and eval_set_id."""
    # Load the eval set file data
    try:
      eval_set_file_path = self._get_eval_set_file_path(app_name, eval_set_id)
      return load_eval_set_from_file(eval_set_file_path, eval_set_id)
    except FileNotFoundError:
      return None

  @override
  def create_eval_set(self, app_name: str, eval_set_id: str):
    """Creates an empty EvalSet given the app_name and eval_set_id."""
    self._validate_id(id_name="Eval Set Id", id_value=eval_set_id)

    # Define the file path
    new_eval_set_path = self._get_eval_set_file_path(app_name, eval_set_id)

    logger.info("Creating eval set file `%s`", new_eval_set_path)

    if not os.path.exists(new_eval_set_path):
      # Write the JSON string to the file
      logger.info("Eval set file doesn't exist, we will create a new one.")
      new_eval_set = EvalSet(
          eval_set_id=eval_set_id,
          name=eval_set_id,
          eval_cases=[],
          creation_timestamp=time.time(),
      )
      self._write_eval_set_to_path(new_eval_set_path, new_eval_set)

  @override
  def list_eval_sets(self, app_name: str) -> list[str]:
    """Returns a list of EvalSets that belong to the given app_name.

    Args:
      app_name: The app name to list the eval sets for.

    Returns:
      A list of EvalSet ids.

    Raises:
      NotFoundError: If the eval directory for the app is not found.
    """
    eval_set_file_path = os.path.join(self._agents_dir, app_name)
    eval_sets = []
    try:
      for file in os.listdir(eval_set_file_path):
        if file.endswith(_EVAL_SET_FILE_EXTENSION):
          eval_sets.append(
              os.path.basename(file).removesuffix(_EVAL_SET_FILE_EXTENSION)
          )
      return sorted(eval_sets)
    except FileNotFoundError as e:
      raise NotFoundError(
          f"Eval directory for app `{app_name}` not found."
      ) from e

  @override
  def get_eval_case(
      self, app_name: str, eval_set_id: str, eval_case_id: str
  ) -> Optional[EvalCase]:
    """Returns an EvalCase if found, otherwise None."""
    eval_set = self.get_eval_set(app_name, eval_set_id)
    if not eval_set:
      return None
    return get_eval_case_from_eval_set(eval_set, eval_case_id)

  @override
  def add_eval_case(self, app_name: str, eval_set_id: str, eval_case: EvalCase):
    """Adds the given EvalCase to an existing EvalSet identified by app_name and eval_set_id.

    Raises:
      NotFoundError: If the eval set is not found.
    """
    eval_set = get_eval_set_from_app_and_id(self, app_name, eval_set_id)
    updated_eval_set = add_eval_case_to_eval_set(eval_set, eval_case)

    self._save_eval_set(app_name, eval_set_id, updated_eval_set)

  @override
  def update_eval_case(
      self, app_name: str, eval_set_id: str, updated_eval_case: EvalCase
  ):
    """Updates an existing EvalCase give the app_name and eval_set_id.

    Raises:
      NotFoundError: If the eval set or the eval case is not found.
    """
    eval_set = get_eval_set_from_app_and_id(self, app_name, eval_set_id)
    updated_eval_set = update_eval_case_in_eval_set(eval_set, updated_eval_case)
    self._save_eval_set(app_name, eval_set_id, updated_eval_set)

  @override
  def delete_eval_case(
      self, app_name: str, eval_set_id: str, eval_case_id: str
  ):
    """Deletes the given EvalCase identified by app_name, eval_set_id and eval_case_id.

    Raises:
      NotFoundError: If the eval set or the eval case to delete is not found.
    """
    eval_set = get_eval_set_from_app_and_id(self, app_name, eval_set_id)
    updated_eval_set = delete_eval_case_from_eval_set(eval_set, eval_case_id)
    self._save_eval_set(app_name, eval_set_id, updated_eval_set)

  def _get_eval_set_file_path(self, app_name: str, eval_set_id: str) -> str:
    return os.path.join(
        self._agents_dir,
        app_name,
        eval_set_id + _EVAL_SET_FILE_EXTENSION,
    )

  def _validate_id(self, id_name: str, id_value: str):
    pattern = r"^[a-zA-Z0-9_]+$"
    if not bool(re.fullmatch(pattern, id_value)):
      raise ValueError(
          f"Invalid {id_name}. {id_name} should have the `{pattern}` format",
      )

  def _write_eval_set_to_path(self, eval_set_path: str, eval_set: EvalSet):
    with open(eval_set_path, "w", encoding="utf-8") as f:
      f.write(eval_set.model_dump_json(indent=2))

  def _save_eval_set(self, app_name: str, eval_set_id: str, eval_set: EvalSet):
    eval_set_file_path = self._get_eval_set_file_path(app_name, eval_set_id)
    self._write_eval_set_to_path(eval_set_file_path, eval_set)
