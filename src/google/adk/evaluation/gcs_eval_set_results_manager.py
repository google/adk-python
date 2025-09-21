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

from google.cloud import exceptions as cloud_exceptions
from google.cloud import storage
from typing_extensions import override

from ..errors.not_found_error import NotFoundError
from ._eval_set_results_manager_utils import create_eval_set_result
from .eval_result import EvalCaseResult
from .eval_result import EvalSetResult
from .eval_set_results_manager import EvalSetResultsManager

logger = logging.getLogger("google_adk." + __name__)

_EVAL_HISTORY_DIR = "evals/eval_history"
_EVAL_SET_RESULT_FILE_EXTENSION = ".evalset_result.json"


class GcsEvalSetResultsManager(EvalSetResultsManager):
  """An EvalSetResultsManager that stores eval results in a GCS bucket."""

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
          f"Bucket `{self.bucket_name}` does not exist. Please create it before"
          " using the GcsEvalSetsManager."
      )

  def _get_eval_history_dir(self, app_name: str) -> str:
    return f"{app_name}/{_EVAL_HISTORY_DIR}"

  def _get_eval_set_result_blob_name(
      self, app_name: str, eval_set_result_id: str
  ) -> str:
    eval_history_dir = self._get_eval_history_dir(app_name)
    return f"{eval_history_dir}/{eval_set_result_id}{_EVAL_SET_RESULT_FILE_EXTENSION}"

  def _write_eval_set_result(
      self, blob_name: str, eval_set_result: EvalSetResult
  ):
    """Writes an EvalSetResult to GCS."""
    blob = self.bucket.blob(blob_name)
    blob.upload_from_string(
        eval_set_result.model_dump_json(indent=2),
        content_type="application/json",
    )

  @override
  def save_eval_set_result(
      self,
      app_name: str,
      eval_set_id: str,
      eval_case_results: list[EvalCaseResult],
  ) -> None:
    """Creates and saves a new EvalSetResult given eval_case_results."""
    eval_set_result = create_eval_set_result(
        app_name, eval_set_id, eval_case_results
    )

    eval_set_result_blob_name = self._get_eval_set_result_blob_name(
        app_name, eval_set_result.eval_set_result_id
    )
    logger.info("Writing eval result to blob: %s", eval_set_result_blob_name)
    self._write_eval_set_result(eval_set_result_blob_name, eval_set_result)

  @override
  def get_eval_set_result(
      self, app_name: str, eval_set_result_id: str
  ) -> EvalSetResult:
    """Returns an EvalSetResult from app_name and eval_set_result_id."""
    eval_set_result_blob_name = self._get_eval_set_result_blob_name(
        app_name, eval_set_result_id
    )
    blob = self.bucket.blob(eval_set_result_blob_name)
    if not blob.exists():
      raise NotFoundError(f"Eval set result `{eval_set_result_id}` not found.")
    eval_set_result_data = blob.download_as_text()
    return EvalSetResult.model_validate_json(eval_set_result_data)

  @override
  def list_eval_set_results(self, app_name: str) -> list[str]:
    """Returns the eval result ids that belong to the given app_name."""
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
