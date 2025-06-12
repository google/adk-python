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

from typing import Optional
from typing import Union

from google.adk.errors.not_found_error import NotFoundError
from google.adk.evaluation.eval_case import EvalCase
from google.adk.evaluation.eval_set import EvalSet
from google.adk.evaluation.gcs_eval_sets_manager import _EVAL_SET_FILE_EXTENSION
from google.adk.evaluation.gcs_eval_sets_manager import GcsEvalSetsManager
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
    self._exists: bool = False

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
    self._exists = True

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
    self._exists = False

  def exists(self) -> bool:
    """Mocks checking if the blob exists."""
    return self._exists


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


class TestGcsEvalSetsManager:
  """Tests for GcsEvalSetsManager."""

  @pytest.fixture
  def gcs_eval_sets_manager(self, mocker):
    mock_storage_client = MockClient()
    bucket_name = "test_bucket"
    mock_bucket = MockBucket(bucket_name)
    mocker.patch.object(mock_storage_client, "bucket", return_value=mock_bucket)
    mocker.patch(
        "google.cloud.storage.Client", return_value=mock_storage_client
    )
    return GcsEvalSetsManager(bucket_name=bucket_name)

  def test_gcs_eval_sets_manager_get_eval_set_success(
      self, gcs_eval_sets_manager
  ):
    app_name = "test_app"
    eval_set_id = "test_eval_set"
    mock_eval_set = EvalSet(eval_set_id=eval_set_id, eval_cases=[])
    mock_bucket = gcs_eval_sets_manager.bucket
    mock_blob = mock_bucket.blob(
        f"{app_name}/evals/eval_sets/{eval_set_id}{_EVAL_SET_FILE_EXTENSION}"
    )
    mock_blob.upload_from_string(mock_eval_set.model_dump_json())

    eval_set = gcs_eval_sets_manager.get_eval_set(app_name, eval_set_id)

    assert eval_set == mock_eval_set

  def test_gcs_eval_sets_manager_get_eval_set_not_found(
      self, gcs_eval_sets_manager
  ):
    app_name = "test_app"
    eval_set_id = "test_eval_set_not_exist"
    eval_set = gcs_eval_sets_manager.get_eval_set(app_name, eval_set_id)

    assert eval_set is None

  def test_gcs_eval_sets_manager_create_eval_set_success(
      self, gcs_eval_sets_manager, mocker
  ):
    mocked_time = 12345678
    mocker.patch("time.time", return_value=mocked_time)
    app_name = "test_app"
    eval_set_id = "test_eval_set"
    mock_write_eval_set_to_blob = mocker.patch.object(
        gcs_eval_sets_manager,
        "_write_eval_set_to_blob",
    )
    eval_set_blob_name = gcs_eval_sets_manager._get_eval_set_blob_name(
        app_name, eval_set_id
    )

    gcs_eval_sets_manager.create_eval_set(app_name, eval_set_id)

    mock_write_eval_set_to_blob.assert_called_once_with(
        eval_set_blob_name,
        EvalSet(
            eval_set_id=eval_set_id,
            name=eval_set_id,
            eval_cases=[],
            creation_timestamp=mocked_time,
        ),
    )

  def test_gcs_eval_sets_manager_create_eval_set_invalid_id(
      self, gcs_eval_sets_manager
  ):
    app_name = "test_app"
    eval_set_id = "invalid-id"

    with pytest.raises(ValueError, match="Invalid Eval Set Id"):
      gcs_eval_sets_manager.create_eval_set(app_name, eval_set_id)

  def test_gcs_eval_sets_manager_list_eval_sets_success(
      self, gcs_eval_sets_manager
  ):
    app_name = "test_app"
    mock_blob_1 = MockBlob(
        f"test_app/evals/eval_sets/eval_set_1{_EVAL_SET_FILE_EXTENSION}"
    )
    mock_blob_2 = MockBlob(
        f"test_app/evals/eval_sets/eval_set_2{_EVAL_SET_FILE_EXTENSION}"
    )
    mock_blob_3 = MockBlob("test_app/evals/eval_sets/not_an_eval_set.txt")
    mock_bucket = gcs_eval_sets_manager.bucket
    mock_bucket.blobs = {
        mock_blob_1.name: mock_blob_1,
        mock_blob_2.name: mock_blob_2,
        mock_blob_3.name: mock_blob_3,
    }

    eval_sets = gcs_eval_sets_manager.list_eval_sets(app_name)

    assert eval_sets == ["eval_set_1", "eval_set_2"]

  def test_gcs_eval_sets_manager_add_eval_case_success(
      self, gcs_eval_sets_manager, mocker
  ):
    app_name = "test_app"
    eval_set_id = "test_eval_set"
    eval_case_id = "test_eval_case"
    mock_eval_case = EvalCase(eval_id=eval_case_id, conversation=[])
    mock_eval_set = EvalSet(eval_set_id=eval_set_id, eval_cases=[])
    mocker.patch.object(
        gcs_eval_sets_manager, "get_eval_set", return_value=mock_eval_set
    )
    mock_write_eval_set_to_blob = mocker.patch.object(
        gcs_eval_sets_manager, "_write_eval_set_to_blob"
    )
    eval_set_blob_name = gcs_eval_sets_manager._get_eval_set_blob_name(
        app_name, eval_set_id
    )

    gcs_eval_sets_manager.add_eval_case(app_name, eval_set_id, mock_eval_case)

    assert len(mock_eval_set.eval_cases) == 1
    assert mock_eval_set.eval_cases[0] == mock_eval_case
    mock_write_eval_set_to_blob.assert_called_once_with(
        eval_set_blob_name, mock_eval_set
    )

  def test_gcs_eval_sets_manager_add_eval_case_eval_set_not_found(
      self, gcs_eval_sets_manager, mocker
  ):
    app_name = "test_app"
    eval_set_id = "test_eval_set"
    eval_case_id = "test_eval_case"
    mock_eval_case = EvalCase(eval_id=eval_case_id, conversation=[])
    mocker.patch.object(
        gcs_eval_sets_manager, "get_eval_set", return_value=None
    )

    with pytest.raises(
        NotFoundError, match="Eval set `test_eval_set` not found."
    ):
      gcs_eval_sets_manager.add_eval_case(app_name, eval_set_id, mock_eval_case)

  def test_gcs_eval_sets_manager_add_eval_case_eval_case_id_exists(
      self, gcs_eval_sets_manager, mocker
  ):
    app_name = "test_app"
    eval_set_id = "test_eval_set"
    eval_case_id = "test_eval_case"
    mock_eval_case = EvalCase(eval_id=eval_case_id, conversation=[])
    mock_eval_set = EvalSet(
        eval_set_id=eval_set_id, eval_cases=[mock_eval_case]
    )
    mocker.patch.object(
        gcs_eval_sets_manager, "get_eval_set", return_value=mock_eval_set
    )

    with pytest.raises(
        ValueError,
        match=(
            f"Eval id `{eval_case_id}` already exists in `{eval_set_id}` eval"
            " set."
        ),
    ):
      gcs_eval_sets_manager.add_eval_case(app_name, eval_set_id, mock_eval_case)

  def test_gcs_eval_sets_manager_get_eval_case_success(
      self, gcs_eval_sets_manager, mocker
  ):
    app_name = "test_app"
    eval_set_id = "test_eval_set"
    eval_case_id = "test_eval_case"
    mock_eval_case = EvalCase(eval_id=eval_case_id, conversation=[])
    mock_eval_set = EvalSet(
        eval_set_id=eval_set_id, eval_cases=[mock_eval_case]
    )
    mocker.patch.object(
        gcs_eval_sets_manager, "get_eval_set", return_value=mock_eval_set
    )

    eval_case = gcs_eval_sets_manager.get_eval_case(
        app_name, eval_set_id, eval_case_id
    )

    assert eval_case == mock_eval_case

  def test_gcs_eval_sets_manager_get_eval_case_eval_set_not_found(
      self, gcs_eval_sets_manager, mocker
  ):
    app_name = "test_app"
    eval_set_id = "test_eval_set"
    eval_case_id = "test_eval_case"
    mocker.patch.object(
        gcs_eval_sets_manager, "get_eval_set", return_value=None
    )

    eval_case = gcs_eval_sets_manager.get_eval_case(
        app_name, eval_set_id, eval_case_id
    )

    assert eval_case is None

  def test_gcs_eval_sets_manager_get_eval_case_eval_case_not_found(
      self, gcs_eval_sets_manager, mocker
  ):
    app_name = "test_app"
    eval_set_id = "test_eval_set"
    eval_case_id = "test_eval_case"
    mock_eval_set = EvalSet(eval_set_id=eval_set_id, eval_cases=[])
    mocker.patch.object(
        gcs_eval_sets_manager, "get_eval_set", return_value=mock_eval_set
    )

    eval_case = gcs_eval_sets_manager.get_eval_case(
        app_name, eval_set_id, eval_case_id
    )

    assert eval_case is None

  def test_gcs_eval_sets_manager_update_eval_case_success(
      self, gcs_eval_sets_manager, mocker
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
    mocker.patch.object(
        gcs_eval_sets_manager, "get_eval_set", return_value=mock_eval_set
    )
    mocker.patch.object(
        gcs_eval_sets_manager, "get_eval_case", return_value=mock_eval_case
    )
    mock_write_eval_set_to_blob = mocker.patch.object(
        gcs_eval_sets_manager, "_write_eval_set_to_blob"
    )
    eval_set_blob_name = gcs_eval_sets_manager._get_eval_set_blob_name(
        app_name, eval_set_id
    )

    gcs_eval_sets_manager.update_eval_case(
        app_name, eval_set_id, updated_eval_case
    )

    assert len(mock_eval_set.eval_cases) == 1
    assert mock_eval_set.eval_cases[0] == updated_eval_case
    mock_write_eval_set_to_blob.assert_called_once_with(
        eval_set_blob_name,
        EvalSet(eval_set_id=eval_set_id, eval_cases=[updated_eval_case]),
    )

  def test_gcs_eval_sets_manager_update_eval_case_eval_set_not_found(
      self, gcs_eval_sets_manager, mocker
  ):
    app_name = "test_app"
    eval_set_id = "test_eval_set"
    eval_case_id = "test_eval_case"
    updated_eval_case = EvalCase(eval_id=eval_case_id, conversation=[])
    mocker.patch.object(
        gcs_eval_sets_manager, "get_eval_case", return_value=None
    )

    with pytest.raises(
        NotFoundError,
        match=f"Eval set `{eval_set_id}` not found.",
    ):
      gcs_eval_sets_manager.update_eval_case(
          app_name, eval_set_id, updated_eval_case
      )

  def test_gcs_eval_sets_manager_update_eval_case_eval_case_not_found(
      self, gcs_eval_sets_manager, mocker
  ):
    app_name = "test_app"
    eval_set_id = "test_eval_set"
    eval_case_id = "test_eval_case"
    mock_eval_set = EvalSet(eval_set_id=eval_set_id, eval_cases=[])
    mocker.patch.object(
        gcs_eval_sets_manager, "get_eval_set", return_value=mock_eval_set
    )
    updated_eval_case = EvalCase(eval_id=eval_case_id, conversation=[])

    with pytest.raises(
        NotFoundError,
        match=(
            f"Eval case `{eval_case_id}` not found in eval set `{eval_set_id}`."
        ),
    ):
      gcs_eval_sets_manager.update_eval_case(
          app_name, eval_set_id, updated_eval_case
      )

  def test_gcs_eval_sets_manager_delete_eval_case_success(
      self, gcs_eval_sets_manager, mocker
  ):
    app_name = "test_app"
    eval_set_id = "test_eval_set"
    eval_case_id = "test_eval_case"
    mock_eval_case = EvalCase(eval_id=eval_case_id, conversation=[])
    mock_eval_set = EvalSet(
        eval_set_id=eval_set_id, eval_cases=[mock_eval_case]
    )
    mock_bucket = gcs_eval_sets_manager.bucket
    mock_blob = mock_bucket.blob(
        f"{app_name}/evals/eval_sets/{eval_set_id}{_EVAL_SET_FILE_EXTENSION}"
    )
    mock_blob.upload_from_string(mock_eval_set.model_dump_json())
    mocker.patch.object(
        gcs_eval_sets_manager, "get_eval_set", return_value=mock_eval_set
    )
    mocker.patch.object(
        gcs_eval_sets_manager, "get_eval_case", return_value=mock_eval_case
    )
    mock_write_eval_set_to_blob = mocker.patch.object(
        gcs_eval_sets_manager, "_write_eval_set_to_blob"
    )
    eval_set_blob_name = gcs_eval_sets_manager._get_eval_set_blob_name(
        app_name, eval_set_id
    )

    gcs_eval_sets_manager.delete_eval_case(app_name, eval_set_id, eval_case_id)

    assert len(mock_eval_set.eval_cases) == 0
    mock_write_eval_set_to_blob.assert_called_once_with(
        eval_set_blob_name,
        EvalSet(eval_set_id=eval_set_id, eval_cases=[]),
    )

  def test_gcs_eval_sets_manager_delete_eval_case_eval_set_not_found(
      self, gcs_eval_sets_manager, mocker
  ):
    app_name = "test_app"
    eval_set_id = "test_eval_set"
    eval_case_id = "test_eval_case"

    mock_write_eval_set_to_blob = mocker.patch.object(
        gcs_eval_sets_manager, "_write_eval_set_to_blob"
    )

    with pytest.raises(
        NotFoundError,
        match=f"Eval set `{eval_set_id}` not found.",
    ):
      gcs_eval_sets_manager.delete_eval_case(
          app_name, eval_set_id, eval_case_id
      )
    mock_write_eval_set_to_blob.assert_not_called()

  def test_gcs_eval_sets_manager_delete_eval_case_eval_case_not_found(
      self, gcs_eval_sets_manager, mocker
  ):
    app_name = "test_app"
    eval_set_id = "test_eval_set"
    eval_case_id = "test_eval_case"
    mock_eval_set = EvalSet(eval_set_id=eval_set_id, eval_cases=[])
    mocker.patch.object(
        gcs_eval_sets_manager, "get_eval_set", return_value=mock_eval_set
    )
    mocker.patch.object(
        gcs_eval_sets_manager, "get_eval_case", return_value=None
    )
    mock_write_eval_set_to_blob = mocker.patch.object(
        gcs_eval_sets_manager, "_write_eval_set_to_blob"
    )

    with pytest.raises(
        NotFoundError,
        match=(
            f"Eval case `{eval_case_id}` not found in eval set `{eval_set_id}`."
        ),
    ):
      gcs_eval_sets_manager.delete_eval_case(
          app_name, eval_set_id, eval_case_id
      )
    mock_write_eval_set_to_blob.assert_not_called()
