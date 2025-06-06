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
import os
from typing import Optional

from google.cloud import exceptions as cloud_exceptions
from google.cloud import storage
from typing_extensions import override

from .eval_result import EvalSetResult
from .eval_set_results_manager import EvalSetResultsStorageManager

logger = logging.getLogger("google_adk." + __name__)

_EVAL_HISTORY_DIR = "evals/eval_history"
_EVAL_SET_RESULT_FILE_EXTENSION = ".evalset_result.json"


class GcsEvalSetResultsStorageManager(EvalSetResultsStorageManager):
  """An EvalSetResultsStorageManager that stores eval set results in a GCS bucket."""

  def __init__(self, bucket_name: str, **kwargs):
    """Initializes the GcsEvalSetResultsStorageManager.

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
          " using the GcsEvalSetResultsStorageManager."
      )

  def _get_eval_history_dir(self, app_name: str) -> str:
    return f"{app_name}/{_EVAL_HISTORY_DIR}"

  @override
  def get_eval_set_result_path(
      self, app_name: str, eval_set_result_id: str
  ) -> str:
    """Gets the path to the EvalSetResult identified by app_name and eval_set_result_id."""
    eval_history_dir = self._get_eval_history_dir(app_name)
    return f"{eval_history_dir}/{eval_set_result_id}{_EVAL_SET_RESULT_FILE_EXTENSION}"

  @override
  def list_eval_set_results(self, app_name: str) -> list[str]:
    """Gets the EvalSetResult id from the given path."""
    eval_history_dir = self._get_eval_history_dir(app_name)
    eval_set_results = []
    try:
      for blob in self.bucket.list_blobs(prefix=eval_history_dir):
        eval_set_result_id = blob.name.split("/")[-1].removesuffix(
            _EVAL_SET_RESULT_FILE_EXTENSION
        )
        eval_set_results.append(eval_set_result_id)
      return sorted(eval_set_results)
    except cloud_exceptions.NotFound as e:
      raise ValueError(
          f"App `{app_name}` not found in GCS bucket `{self.bucket_name}`."
      ) from e

  @override
  def save_eval_set_result(self, path: str, eval_set_result: EvalSetResult):
    """Writes the EvalSetResult to the given path."""
    logger.info("Saving EvalSetResult to gs://%s/%s", self.bucket_name, path)
    blob = self.bucket.blob(path)
    blob.upload_from_string(
        eval_set_result.model_dump_json(indent=2),
        content_type="application/json",
    )

  @override
  def load_eval_set_result(self, path: str) -> Optional[EvalSetResult]:
    """Loads the EvalSetResult from the given path."""
    try:
      blob = self.bucket.blob(path)
      eval_set_result_data = blob.download_as_text()
      return EvalSetResult.model_validate_json(eval_set_result_data)
    except cloud_exceptions.NotFound:
      return None
