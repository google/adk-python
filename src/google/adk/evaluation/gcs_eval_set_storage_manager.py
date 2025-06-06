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

from google.cloud import exceptions as cloud_exceptions
from google.cloud import storage
from typing_extensions import override

from .eval_set import EvalSet
from .eval_sets_manager import EvalSetStorageManager

logger = logging.getLogger("google_adk." + __name__)

_EVAL_SETS_DIR = "evals/eval_sets"
_EVAL_SET_FILE_EXTENSION = ".evalset.json"


class GcsEvalSetStorageManager(EvalSetStorageManager):
  """This class is currently under development and should be used with caution.

  Its API may change without notice.

  An EvalSetStorageManager that stores eval sets in a GCS bucket.
  """

  def __init__(self, bucket_name: str, **kwargs):
    """Initializes the GcsEvalSetStorageManager.

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
          f"Bucket `{self.bucket_name}` does not exist. Please create it before"
          " using the GcsEvalSetStorageManager."
      )

  def _get_eval_sets_dir(self, app_name: str) -> str:
    return f"{app_name}/{_EVAL_SETS_DIR}"

  def _validate_id(self, id_name: str, id_value: str):
    pattern = r"^[a-zA-Z0-9_]+$"
    if not bool(re.fullmatch(pattern, id_value)):
      raise ValueError(
          f"Invalid {id_name}. {id_name} should have the `{pattern}` format",
      )

  @override
  def get_eval_set_path(self, app_name: str, eval_set_id: str) -> str:
    """Gets the path to the EvalSet identified by app_name and eval_set_id."""
    eval_sets_dir = self._get_eval_sets_dir(app_name)
    return f"{eval_sets_dir}/{eval_set_id}{_EVAL_SET_FILE_EXTENSION}"

  @override
  def list_eval_sets(self, app_name: str) -> list[str]:
    """Gets the EvalSet id from the given path."""
    eval_sets_dir = self._get_eval_sets_dir(app_name)
    eval_sets = []
    try:
      for blob in self.bucket.list_blobs(prefix=eval_sets_dir):
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
  def save_eval_set(self, path: str, eval_set: EvalSet):
    """Writes the EvalSet to the given path."""
    blob = self.bucket.blob(path)
    blob.upload_from_string(
        eval_set.model_dump_json(indent=2),
        content_type="application/json",
    )

  @override
  def load_eval_set(self, path: str) -> EvalSet | None:
    """Loads the EvalSet from the given path."""
    try:
      blob = self.bucket.blob(path)
      eval_set_data = blob.download_as_text()
      return EvalSet.model_validate_json(eval_set_data)
    except cloud_exceptions.NotFound:
      return None
