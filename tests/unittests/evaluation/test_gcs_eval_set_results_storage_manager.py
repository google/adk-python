from unittest import mock

from google.adk.evaluation.eval_result import EvalSetResult
from google.adk.evaluation.gcs_eval_set_results_storage_manager import GcsEvalSetResultsStorageManager
from google.cloud import exceptions as cloud_exceptions
import pytest

from .test_gcs_eval_set_storage_manager import MockBlob
from .test_gcs_eval_set_storage_manager import MockBucket
from .test_gcs_eval_set_storage_manager import MockClient


def mock_gcs_eval_set_results_storage_manager():
  with mock.patch("google.cloud.storage.Client", return_value=MockClient()):
    storage_manager = GcsEvalSetResultsStorageManager(bucket_name="test_bucket")
    storage_manager.bucket = storage_manager.storage_client.bucket(
        "test_bucket"
    )
    return storage_manager


def test_get_eval_set_result_path():
  manager = mock_gcs_eval_set_results_storage_manager()
  path = manager.get_eval_set_result_path("test_app", "test_eval_set_result")
  assert (
      path
      == "test_app/evals/eval_history/test_eval_set_result.evalset_result.json"
  )


def test_list_eval_set_results():
  mock_blob1 = MockBlob(
      name="test_app/evals/eval_history/eval_result_1.evalset_result.json"
  )
  mock_blob2 = MockBlob(
      name="test_app/evals/eval_history/eval_result_2.evalset_result.json"
  )
  mock_bucket = MockBucket("test_bucket")
  mock_bucket.blobs = {
      "test_app/evals/eval_history/eval_result_1.evalset_result.json": (
          mock_blob1
      ),
      "test_app/evals/eval_history/eval_result_2.evalset_result.json": (
          mock_blob2
      ),
  }

  manager = mock_gcs_eval_set_results_storage_manager()
  manager.bucket = mock_bucket
  eval_results = manager.list_eval_set_results("test_app")
  assert eval_results == ["eval_result_1", "eval_result_2"]


def test_list_eval_set_results_app_not_found(mocker):
  manager = mock_gcs_eval_set_results_storage_manager()
  mock_bucket = MockBucket("test_bucket")
  manager.bucket = mock_bucket
  mocker.patch.object(
      manager.bucket,
      "list_blobs",
      side_effect=cloud_exceptions.NotFound("Not found"),
  )
  with pytest.raises(
      ValueError, match="App `test_app` not found in GCS bucket `test_bucket`"
  ):
    manager.list_eval_set_results("test_app")


def test_save_eval_set_result():
  manager = mock_gcs_eval_set_results_storage_manager()
  mock_bucket = MockBucket("test_bucket")
  mock_blob = MockBlob(
      name=(
          "test_app/evals/eval_history/test_eval_set_result.evalset_result.json"
      )
  )
  mock_bucket.blobs = {
      "test_app/evals/eval_history/test_eval_set_result.evalset_result.json": (
          mock_blob
      )
  }
  manager.bucket = mock_bucket
  eval_set_result = EvalSetResult(
      eval_set_result_id="test_eval_set_result", eval_set_id="test_eval_set"
  )
  manager.save_eval_set_result(
      "test_app/evals/eval_history/test_eval_set_result.evalset_result.json",
      eval_set_result,
  )
  assert mock_blob.content.decode() == eval_set_result.model_dump_json(indent=2)


def test_load_eval_set_result():
  manager = mock_gcs_eval_set_results_storage_manager()
  mock_bucket = MockBucket("test_bucket")
  manager.bucket = mock_bucket
  eval_set_result_data = (
      '{"eval_set_result_id": "test_eval_set_result", "eval_set_id":'
      ' "test_eval_set", "eval_case_results": []}'
  )
  mock_blob = MockBlob(
      name=(
          "test_app/evals/eval_history/test_eval_set_result.evalset_result.json"
      )
  )
  mock_blob.content = eval_set_result_data.encode()
  mock_bucket.blobs = {
      "test_app/evals/eval_history/test_eval_set_result.evalset_result.json": (
          mock_blob
      )
  }

  loaded_eval_set_result = manager.load_eval_set_result(
      "test_app/evals/eval_history/test_eval_set_result.evalset_result.json"
  )
  assert loaded_eval_set_result == EvalSetResult.model_validate_json(
      eval_set_result_data
  )


def test_load_eval_set_result_not_found():
  manager = mock_gcs_eval_set_results_storage_manager()
  mock_bucket = MockBucket("test_bucket")
  manager.bucket = mock_bucket
  mock_blob = MockBlob(name="nonexistent-blob")
  mock_bucket.blobs = {"nonexistent-blob": mock_blob}

  with mock.patch.object(
      manager.bucket, "blob", side_effect=cloud_exceptions.NotFound("Not found")
  ):
    loaded_eval_set_result = manager.load_eval_set_result(
        "test_app/evals/eval_history/test_eval_set_result.evalset_result.json"
    )
    assert loaded_eval_set_result is None
