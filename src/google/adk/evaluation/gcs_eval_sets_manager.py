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

import logging
import re
import time
from typing import Optional

from google.cloud import exceptions as cloud_exceptions
from google.cloud import storage
from typing_extensions import override

from ._eval_sets_manager_utils import add_eval_case_to_eval_set
from ._eval_sets_manager_utils import delete_eval_case_from_eval_set
from ._eval_sets_manager_utils import get_eval_case_from_eval_set
from ._eval_sets_manager_utils import get_eval_set_from_app_and_id
from ._eval_sets_manager_utils import update_eval_case_in_eval_set
from .eval_case import EvalCase
from .eval_set import EvalSet
from .eval_sets_manager import EvalSetsManager

logger = logging.getLogger("google_adk." + __name__)

_EVAL_SETS_DIR = "evals/eval_sets"
_EVAL_SET_FILE_EXTENSION = ".evalset.json"


class GcsEvalSetsManager(EvalSetsManager):
  """An EvalSetsManager that stores eval sets in a GCS bucket."""

  def __init__(self, bucket_name: str, **kwargs):
    """Initializes the GcsEvalSetsManager.

    Args:
      bucket_name: The name of the bucket to use.
      **kwargs: Keyword arguments to pass to the Google Cloud Storage client.
    """
    self.bucket_name = bucket_name
    self.storage_client = storage.Client(**kwargs)
    self.bucket = self.storage_client.bucket(self.bucket_name)
    # Check if the bucket exists.
    if not self.bucket.exists():
      raise ValueError(
          f"Bucket `{self.bucket_name}` does not exist. Please create it "
          "before using the GcsEvalSetsManager."
      )

  def _get_eval_sets_dir(self, app_name: str) -> str:
    return f"{app_name}/{_EVAL_SETS_DIR}"

  def _get_eval_set_blob_name(self, app_name: str, eval_set_id: str) -> str:
    eval_sets_dir = self._get_eval_sets_dir(app_name)
    return f"{eval_sets_dir}/{eval_set_id}{_EVAL_SET_FILE_EXTENSION}"

  def _validate_id(self, id_name: str, id_value: str):
    pattern = r"^[a-zA-Z0-9_]+$"
    if not bool(re.fullmatch(pattern, id_value)):
      raise ValueError(
          f"Invalid {id_name}. {id_name} should have the `{pattern}` format",
      )

  def _write_eval_set_to_blob(self, blob_name: str, eval_set: EvalSet):
    """Writes an EvalSet to GCS."""
    blob = self.bucket.blob(blob_name)
    blob.upload_from_string(
        eval_set.model_dump_json(indent=2),
        content_type="application/json",
    )

  def _save_eval_set(self, app_name: str, eval_set_id: str, eval_set: EvalSet):
    eval_set_blob_name = self._get_eval_set_blob_name(app_name, eval_set_id)
    self._write_eval_set_to_blob(eval_set_blob_name, eval_set)

  @override
  def get_eval_set(self, app_name: str, eval_set_id: str) -> Optional[EvalSet]:
    """Returns an EvalSet identified by an app_name and eval_set_id."""
    eval_set_blob_name = self._get_eval_set_blob_name(app_name, eval_set_id)
    blob = self.bucket.blob(eval_set_blob_name)
    if not blob.exists():
      return None
    eval_set_data = blob.download_as_text()
    return EvalSet.model_validate_json(eval_set_data)

  @override
  def create_eval_set(self, app_name: str, eval_set_id: str):
    """Creates an empty EvalSet and saves it to GCS."""
    self._validate_id(id_name="Eval Set Id", id_value=eval_set_id)
    new_eval_set_blob_name = self._get_eval_set_blob_name(app_name, eval_set_id)
    if self.bucket.blob(new_eval_set_blob_name).exists():
      raise ValueError(
          f"Eval set `{eval_set_id}` already exists for app `{app_name}`."
      )
    logger.info("Creating eval set blob: `%s`", new_eval_set_blob_name)
    new_eval_set = EvalSet(
        eval_set_id=eval_set_id,
        name=eval_set_id,
        eval_cases=[],
        creation_timestamp=time.time(),
    )
    self._write_eval_set_to_blob(new_eval_set_blob_name, new_eval_set)

  @override
  def list_eval_sets(self, app_name: str) -> list[str]:
    """Returns a list of EvalSet ids that belong to the given app_name."""
    eval_sets_dir = self._get_eval_sets_dir(app_name)
    eval_sets = []
    try:
      for blob in self.bucket.list_blobs(prefix=eval_sets_dir):
        if not blob.name.endswith(_EVAL_SET_FILE_EXTENSION):
          continue
        eval_set_id = blob.name.split("/")[-1].removesuffix(
            _EVAL_SET_FILE_EXTENSION
        )
        eval_sets.append(eval_set_id)
      return sorted(eval_sets)
    except cloud_exceptions.NotFound as e:
      raise ValueError(
          f"App `{app_name}` not found in GCS bucket `{self.bucket_name}`."
      ) from e

  @override
  def get_eval_case(
      self, app_name: str, eval_set_id: str, eval_case_id: str
  ) -> Optional[EvalCase]:
    """Returns an EvalCase identified by an app_name, eval_set_id and eval_case_id."""
    eval_set = self.get_eval_set(app_name, eval_set_id)
    if not eval_set:
      return None
    return get_eval_case_from_eval_set(eval_set, eval_case_id)

  @override
  def add_eval_case(self, app_name: str, eval_set_id: str, eval_case: EvalCase):
    """Adds the given EvalCase to an existing EvalSet.

    Args:
      app_name: The name of the app.
      eval_set_id: The id of the eval set containing the eval case to update.
      eval_case: The EvalCase to add.

    Raises:
      NotFoundError: If the eval set is not found.
      ValueError: If the eval case already exists in the eval set.
    """
    eval_set = get_eval_set_from_app_and_id(self, app_name, eval_set_id)
    updated_eval_set = add_eval_case_to_eval_set(eval_set, eval_case)
    self._save_eval_set(app_name, eval_set_id, updated_eval_set)

  @override
  def update_eval_case(
      self, app_name: str, eval_set_id: str, updated_eval_case: EvalCase
  ):
    """Updates an existing EvalCase.

    Args:
      app_name: The name of the app.
      eval_set_id: The id of the eval set containing the eval case to update.
      updated_eval_case: The updated EvalCase. Overwrites the existing EvalCase
        using the eval_id field.

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
    """Deletes the EvalCase with the given eval_case_id from the given EvalSet.

    Args:
      app_name: The name of the app.
      eval_set_id: The id of the eval set containing the eval case to delete.
      eval_case_id: The id of the eval case to delete.

    Raises:
      NotFoundError: If the eval set or the eval case to delete is not found.
    """
    eval_set = get_eval_set_from_app_and_id(self, app_name, eval_set_id)
    updated_eval_set = delete_eval_case_from_eval_set(eval_set, eval_case_id)
    self._save_eval_set(app_name, eval_set_id, updated_eval_set)
