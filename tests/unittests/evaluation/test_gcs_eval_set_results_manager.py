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

from google.adk.errors.not_found_error import NotFoundError
from google.adk.evaluation._eval_set_results_manager_utils import _sanitize_eval_set_result_name
from google.adk.evaluation._eval_set_results_manager_utils import create_eval_set_result
from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.eval_metrics import EvalMetricResult
from google.adk.evaluation.eval_metrics import EvalMetricResultPerInvocation
from google.adk.evaluation.eval_result import EvalCaseResult
from google.adk.evaluation.evaluator import EvalStatus
from google.adk.evaluation.gcs_eval_set_results_manager import GcsEvalSetResultsManager
from google.genai import types as genai_types
import pytest

from .mock_gcs_utils import MockBucket
from .mock_gcs_utils import MockClient


def _get_test_eval_case_results():
  # Create mock Invocation objects
  actual_invocation_1 = Invocation(
      invocation_id="actual_1",
      user_content=genai_types.Content(
          parts=[genai_types.Part(text="input_1")]
      ),
  )
  expected_invocation_1 = Invocation(
      invocation_id="expected_1",
      user_content=genai_types.Content(
          parts=[genai_types.Part(text="expected_input_1")]
      ),
  )
  actual_invocation_2 = Invocation(
      invocation_id="actual_2",
      user_content=genai_types.Content(
          parts=[genai_types.Part(text="input_2")]
      ),
  )
  expected_invocation_2 = Invocation(
      invocation_id="expected_2",
      user_content=genai_types.Content(
          parts=[genai_types.Part(text="expected_input_2")]
      ),
  )

  eval_metric_result_1 = EvalMetricResult(
      metric_name="metric",
      threshold=0.8,
      score=1.0,
      eval_status=EvalStatus.PASSED,
  )
  eval_metric_result_2 = EvalMetricResult(
      metric_name="metric",
      threshold=0.8,
      score=0.5,
      eval_status=EvalStatus.FAILED,
  )
  eval_metric_result_per_invocation_1 = EvalMetricResultPerInvocation(
      actual_invocation=actual_invocation_1,
      expected_invocation=expected_invocation_1,
      eval_metric_results=[eval_metric_result_1],
  )
  eval_metric_result_per_invocation_2 = EvalMetricResultPerInvocation(
      actual_invocation=actual_invocation_2,
      expected_invocation=expected_invocation_2,
      eval_metric_results=[eval_metric_result_2],
  )
  return [
      EvalCaseResult(
          eval_set_id="eval_set",
          eval_id="eval_case_1",
          final_eval_status=EvalStatus.PASSED,
          overall_eval_metric_results=[eval_metric_result_1],
          eval_metric_result_per_invocation=[
              eval_metric_result_per_invocation_1
          ],
          session_id="session_1",
      ),
      EvalCaseResult(
          eval_set_id="eval_set",
          eval_id="eval_case_2",
          final_eval_status=EvalStatus.FAILED,
          overall_eval_metric_results=[eval_metric_result_2],
          eval_metric_result_per_invocation=[
              eval_metric_result_per_invocation_2
          ],
          session_id="session_2",
      ),
  ]


class TestGcsEvalSetResultsManager:

  @pytest.fixture
  def gcs_eval_set_results_manager(self, mocker):
    mock_storage_client = MockClient()
    bucket_name = "test_bucket"
    mock_bucket = MockBucket(bucket_name)
    mocker.patch.object(mock_storage_client, "bucket", return_value=mock_bucket)
    mocker.patch(
        "google.cloud.storage.Client", return_value=mock_storage_client
    )
    return GcsEvalSetResultsManager(bucket_name=bucket_name)

  def test_save_eval_set_result(self, gcs_eval_set_results_manager, mocker):
    mocker.patch("time.time", return_value=12345678)
    app_name = "test_app"
    eval_set_id = "test_eval_set"
    eval_case_results = _get_test_eval_case_results()
    eval_set_result = create_eval_set_result(
        app_name, eval_set_id, eval_case_results
    )
    blob_name = gcs_eval_set_results_manager._get_eval_set_result_blob_name(
        app_name, eval_set_result.eval_set_result_id
    )
    mock_write_eval_set_result = mocker.patch.object(
        gcs_eval_set_results_manager,
        "_write_eval_set_result",
    )
    gcs_eval_set_results_manager.save_eval_set_result(
        app_name, eval_set_id, eval_case_results
    )
    mock_write_eval_set_result.assert_called_once_with(
        blob_name,
        eval_set_result,
    )

  def test_get_eval_set_result_not_found(
      self, gcs_eval_set_results_manager, mocker
  ):
    mocker.patch("time.time", return_value=12345678)
    app_name = "test_app"
    with pytest.raises(NotFoundError) as e:
      gcs_eval_set_results_manager.get_eval_set_result(
          app_name, "non_existent_id"
      )

  def test_get_eval_set_result(self, gcs_eval_set_results_manager, mocker):
    mocker.patch("time.time", return_value=12345678)
    app_name = "test_app"
    eval_set_id = "test_eval_set"
    eval_case_results = _get_test_eval_case_results()
    gcs_eval_set_results_manager.save_eval_set_result(
        app_name, eval_set_id, eval_case_results
    )
    eval_set_result = create_eval_set_result(
        app_name, eval_set_id, eval_case_results
    )
    retrieved_eval_set_result = (
        gcs_eval_set_results_manager.get_eval_set_result(
            app_name, eval_set_result.eval_set_result_id
        )
    )
    assert retrieved_eval_set_result == eval_set_result

  def test_list_eval_set_results(self, gcs_eval_set_results_manager, mocker):
    mocker.patch("time.time", return_value=123)
    app_name = "test_app"
    eval_set_ids = ["test_eval_set_1", "test_eval_set_2", "test_eval_set_3"]
    for eval_set_id in eval_set_ids:
      eval_case_results = _get_test_eval_case_results()
      gcs_eval_set_results_manager.save_eval_set_result(
          app_name, eval_set_id, eval_case_results
      )
    retrieved_eval_set_result_ids = (
        gcs_eval_set_results_manager.list_eval_set_results(app_name)
    )
    assert retrieved_eval_set_result_ids == [
        "test_app_test_eval_set_1_123",
        "test_app_test_eval_set_2_123",
        "test_app_test_eval_set_3_123",
    ]

  def test_list_eval_set_results_empty(self, gcs_eval_set_results_manager):
    app_name = "test_app"
    retrieved_eval_set_result_ids = (
        gcs_eval_set_results_manager.list_eval_set_results(app_name)
    )
    assert retrieved_eval_set_result_ids == []
