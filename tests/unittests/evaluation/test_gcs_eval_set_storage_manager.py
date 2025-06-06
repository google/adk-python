from typing import Optional
from typing import Union
from unittest import mock

from google.adk.evaluation.eval_set import EvalSet
from google.adk.evaluation.gcs_eval_set_storage_manager import GcsEvalSetStorageManager
from google.cloud import exceptions as cloud_exceptions
import pytest


class MockBlob:
  """Mocks a GCS Blob object.

  This class provides mock implementations for a few common GCS Blob methods,
  allowing the user to test code that interacts with GCS without actually
  connecting to a real bucket.
  """

  def __init__(self, name: str) -> None:
    """Initializes a MockBlob.

    Args:
        name: The name of the blob.
    """
    self.name = name
    self.content: Optional[bytes] = None
    self.content_type: Optional[str] = None

  def upload_from_string(
      self, data: Union[str, bytes], content_type: Optional[str] = None
  ) -> None:
    """Mocks uploading data to the blob (from a string or bytes).

    Args:
        data: The data to upload (string or bytes).
        content_type:  The content type of the data (optional).
    """
    if isinstance(data, str):
      self.content = data.encode("utf-8")
    elif isinstance(data, bytes):
      self.content = data
    else:
      raise TypeError("data must be str or bytes")

    if content_type:
      self.content_type = content_type

  def download_as_text(self) -> str:
    """Mocks downloading the blob's content as text.

    Returns:
        str: The content of the blob as text.

    Raises:
        Exception: If the blob doesn't exist (hasn't been uploaded to).
    """
    if self.content is None:
      return b""
    return self.content

  def delete(self) -> None:
    """Mocks deleting a blob."""
    self.content = None
    self.content_type = None

  def exists(self) -> bool:
    """Mocks checking if the blob exists."""
    return True


class MockBucket:
  """Mocks a GCS Bucket object."""

  def __init__(self, name: str) -> None:
    """Initializes a MockBucket.

    Args:
        name: The name of the bucket.
    """
    self.name = name
    self.blobs: dict[str, MockBlob] = {}

  def blob(self, blob_name: str) -> MockBlob:
    """Mocks getting a Blob object (doesn't create it in storage).

    Args:
        blob_name: The name of the blob.

    Returns:
        A MockBlob instance.
    """
    if blob_name not in self.blobs:
      self.blobs[blob_name] = MockBlob(blob_name)
    return self.blobs[blob_name]

  def list_blobs(self, prefix: Optional[str] = None) -> list[MockBlob]:
    """Mocks listing blobs in a bucket, optionally with a prefix."""
    if prefix:
      return [
          blob for name, blob in self.blobs.items() if name.startswith(prefix)
      ]
    return list(self.blobs.values())

  def exists(self) -> bool:
    """Mocks checking if the bucket exists."""
    return True


class MockClient:
  """Mocks the GCS Client."""

  def __init__(self) -> None:
    """Initializes MockClient."""
    self.buckets: dict[str, MockBucket] = {}

  def bucket(self, bucket_name: str) -> MockBucket:
    """Mocks getting a Bucket object."""
    if bucket_name not in self.buckets:
      self.buckets[bucket_name] = MockBucket(bucket_name)
    return self.buckets[bucket_name]


def mock_gcs_eval_set_storage_manager():
  with mock.patch("google.cloud.storage.Client", return_value=MockClient()):
    storage_manager = GcsEvalSetStorageManager(bucket_name="test_bucket")
    storage_manager.bucket = storage_manager.storage_client.bucket(
        "test_bucket"
    )
    return storage_manager


def test_get_eval_set_path():
  manager = mock_gcs_eval_set_storage_manager()
  path = manager.get_eval_set_path("test_app", "test_eval_set")
  assert path == "test_app/evals/eval_sets/test_eval_set.evalset.json"


def test_list_eval_sets():
  mock_blob1 = MockBlob(name="test_app/evals/eval_sets/eval_set_1.evalset.json")
  mock_blob2 = MockBlob(name="test_app/evals/eval_sets/eval_set_2.evalset.json")
  mock_bucket = MockBucket("test_bucket")
  mock_bucket.blobs = {
      "test_app/evals/eval_sets/eval_set_1.evalset.json": mock_blob1,
      "test_app/evals/eval_sets/eval_set_2.evalset.json": mock_blob2,
  }

  manager = mock_gcs_eval_set_storage_manager()
  manager.bucket = mock_bucket
  eval_sets = manager.list_eval_sets("test_app")
  assert eval_sets == ["eval_set_1", "eval_set_2"]


def test_list_eval_sets_app_not_found(mocker):
  manager = mock_gcs_eval_set_storage_manager()
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
    manager.list_eval_sets("test_app")


def test_save_eval_set():
  manager = mock_gcs_eval_set_storage_manager()
  mock_bucket = MockBucket("test_bucket")
  mock_blob = MockBlob(
      name="test_app/evals/eval_sets/test_eval_set.evalset.json"
  )
  mock_bucket.blobs = {
      "test_app/evals/eval_sets/test_eval_set.evalset.json": mock_blob
  }
  manager.bucket = mock_bucket
  eval_set = EvalSet(eval_set_id="test_eval_set", eval_cases=[])
  manager.save_eval_set(
      "test_app/evals/eval_sets/test_eval_set.evalset.json", eval_set
  )
  assert mock_blob.content.decode() == eval_set.model_dump_json(indent=2)


def test_load_eval_set():
  manager = mock_gcs_eval_set_storage_manager()
  mock_bucket = MockBucket("test_bucket")
  manager.bucket = mock_bucket
  eval_set_data = '{"eval_set_id": "test_eval_set", "eval_cases": []}'
  mock_blob = MockBlob(
      name="test_app/evals/eval_sets/test_eval_set.evalset.json"
  )
  mock_blob.content = eval_set_data.encode()
  mock_bucket.blobs = {
      "test_app/evals/eval_sets/test_eval_set.evalset.json": mock_blob
  }

  loaded_eval_set = manager.load_eval_set(
      "test_app/evals/eval_sets/test_eval_set.evalset.json"
  )
  assert loaded_eval_set == EvalSet.model_validate_json(eval_set_data)


def test_load_eval_set_not_found():
  manager = mock_gcs_eval_set_storage_manager()
  mock_bucket = MockBucket("test_bucket")
  manager.bucket = mock_bucket
  mock_blob = MockBlob(name="nonexistent-blob")
  mock_bucket.blobs = {"nonexistent-blob": mock_blob}

  with mock.patch.object(
      manager.bucket, "blob", side_effect=cloud_exceptions.NotFound("Not found")
  ):
    loaded_eval_set = manager.load_eval_set(
        "test_app/evals/eval_sets/test_eval_set.evalset.json"
    )
    assert loaded_eval_set is None


def test_validate_id():
  manager = mock_gcs_eval_set_storage_manager()
  manager._validate_id("eval_set_id", "valid_id123")
  with pytest.raises(ValueError, match="Invalid eval_set_id."):
    manager._validate_id("eval_set_id", "invalid id")
  with pytest.raises(ValueError, match="Invalid eval_set_id."):
    manager._validate_id("eval_set_id", "invalid-id")
