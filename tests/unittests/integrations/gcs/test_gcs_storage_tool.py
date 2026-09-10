# Copyright 2026 Google LLC
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

from unittest import mock
import warnings

from google.adk.integrations.gcs import client
from google.adk.integrations.gcs import storage_tool
from google.adk.integrations.gcs.settings import GCSToolSettings
from google.auth.credentials import Credentials
import pytest


def test_resolve_local_path_returns_path_under_root(tmp_path):
  """Test _resolve_local_path accepts a relative path inside the root."""
  resolved = storage_tool._resolve_local_path(
      "path/to/file.txt", GCSToolSettings(local_file_root=str(tmp_path))
  )
  assert resolved == str(tmp_path.resolve() / "path" / "to" / "file.txt")


def test_resolve_local_path_without_local_file_root():
  """Test _resolve_local_path refuses any path when no root is configured."""
  with pytest.raises(ValueError, match="Local file access is disabled"):
    storage_tool._resolve_local_path("file.txt", GCSToolSettings())

  with pytest.raises(ValueError, match="Local file access is disabled"):
    storage_tool._resolve_local_path("file.txt", None)


@pytest.mark.parametrize(
    "path, expected_error",
    [
        pytest.param("/etc/passwd", "must be relative", id="absolute"),
        pytest.param(
            "../outside.txt", "escapes the configured root", id="dotdot"
        ),
        pytest.param(
            "link/escaped.txt", "escapes the configured root", id="symlink"
        ),
        pytest.param("", "must name a file", id="empty"),
        pytest.param(".", "must name a file", id="dot"),
        pytest.param("sub/..", "must name a file", id="back-to-root"),
    ],
)
def test_resolve_local_path_rejects(tmp_path, path, expected_error):
  """Test _resolve_local_path rejects paths that do not name a file in the root."""
  root = tmp_path / "root"
  root.mkdir()
  outside = tmp_path / "outside"
  outside.mkdir()
  (root / "link").symlink_to(outside, target_is_directory=True)

  with pytest.raises(ValueError, match=expected_error):
    storage_tool._resolve_local_path(
        path, GCSToolSettings(local_file_root=str(root))
    )


def test_list_objects():
  """Test list_objects function."""
  with mock.patch.object(
      client, "get_gcs_client", autospec=True
  ) as mock_get_client:
    mock_client = mock.MagicMock()
    mock_get_client.return_value = mock_client
    mock_bucket = mock.MagicMock()
    mock_client.get_bucket.return_value = mock_bucket
    mock_blob = mock.MagicMock()
    mock_blob.name = "test-object"
    mock_bucket.list_blobs.return_value = [mock_blob]

    creds = mock.create_autospec(Credentials, instance=True)
    result = storage_tool.list_objects(
        bucket_name="test-bucket", credentials=creds
    )
    assert result == {
        "status": "SUCCESS",
        "results": ["test-object"],
    }


def test_list_objects_pagination():
  """Test list_objects function with pagination."""
  with mock.patch.object(
      client, "get_gcs_client", autospec=True
  ) as mock_get_client:
    mock_client = mock.MagicMock()
    mock_get_client.return_value = mock_client
    mock_bucket = mock.MagicMock()
    mock_client.get_bucket.return_value = mock_bucket
    mock_blob = mock.MagicMock()
    mock_blob.name = "test-object"
    mock_blobs = mock.MagicMock()
    mock_blobs.pages = iter([[mock_blob]])
    mock_blobs.next_page_token = "next-page-token"
    mock_bucket.list_blobs.return_value = mock_blobs

    creds = mock.create_autospec(Credentials, instance=True)
    result = storage_tool.list_objects(
        bucket_name="test-bucket",
        credentials=creds,
        page_size=1,
        page_token="token",
    )
    assert result == {
        "status": "SUCCESS",
        "results": ["test-object"],
        "next_page_token": "next-page-token",
    }
    mock_bucket.list_blobs.assert_called_once_with(
        max_results=1, page_token="token"
    )


def test_get_object_metadata():
  """Test get_object_metadata function."""
  with mock.patch.object(
      client, "get_gcs_client", autospec=True
  ) as mock_get_client:
    mock_client = mock.MagicMock()
    mock_get_client.return_value = mock_client
    mock_bucket = mock.MagicMock()
    mock_client.get_bucket.return_value = mock_bucket
    mock_blob = mock.MagicMock()
    mock_bucket.get_blob.return_value = mock_blob
    setattr(
        mock_blob,
        "_properties",
        {
            "kind": "storage#object",
            "id": "test-bucket/test-object/1",
            "name": "test-object",
            "bucket": "test-bucket",
            "size": "1024",
            "contentType": "text/plain",
            "timeCreated": "2024-01-01",
            "updated": "2024-01-02",
            "md5Hash": "hash",
            "metadata": {"key": "value"},
        },
    )

    creds = mock.create_autospec(Credentials, instance=True)
    result = storage_tool.get_object_metadata(
        bucket_name="test-bucket",
        object_name="test-object",
        credentials=creds,
        generation=1,
    )
    expected_result = getattr(mock_blob, "_properties", {}).copy()
    assert result == {"status": "SUCCESS", "results": expected_result}
    mock_bucket.get_blob.assert_called_once_with("test-object", generation=1)


def test_get_object_metadata_not_found():
  """Test get_object_metadata function when object is not found."""
  with mock.patch.object(
      client, "get_gcs_client", autospec=True
  ) as mock_get_client:
    mock_client = mock.MagicMock()
    mock_get_client.return_value = mock_client
    mock_bucket = mock.MagicMock()
    mock_client.get_bucket.return_value = mock_bucket
    mock_bucket.get_blob.return_value = None

    creds = mock.create_autospec(Credentials, instance=True)
    result = storage_tool.get_object_metadata(
        bucket_name="test-bucket",
        object_name="non-existent",
        credentials=creds,
    )
    assert result["status"] == "ERROR"
    assert "not found" in result["error_details"]
    mock_bucket.get_blob.assert_called_once_with("non-existent")


def test_create_object():
  """Test create_object function."""
  with mock.patch.object(
      client, "get_gcs_client", autospec=True
  ) as mock_get_client:
    mock_client = mock.MagicMock()
    mock_get_client.return_value = mock_client
    mock_bucket = mock.MagicMock()
    mock_client.get_bucket.return_value = mock_bucket
    mock_blob = mock.MagicMock()
    mock_bucket.blob.return_value = mock_blob

    creds = mock.create_autospec(Credentials, instance=True)
    result = storage_tool.create_object(
        bucket_name="test-bucket",
        object_name="test-object",
        data="data",
        credentials=creds,
    )
    assert result["status"] == "SUCCESS"
    mock_blob.upload_from_string.assert_called_once_with("data")


def test_create_object_from_file(tmp_path):
  """Test create_object function using source_file_path."""
  with mock.patch.object(
      client, "get_gcs_client", autospec=True
  ) as mock_get_client:
    mock_client = mock.MagicMock()
    mock_get_client.return_value = mock_client
    mock_bucket = mock.MagicMock()
    mock_client.get_bucket.return_value = mock_bucket
    mock_blob = mock.MagicMock()
    mock_bucket.blob.return_value = mock_blob

    creds = mock.create_autospec(Credentials, instance=True)
    result = storage_tool.create_object(
        bucket_name="test-bucket",
        object_name="test-object",
        source_file_path="path/to/file.txt",
        credentials=creds,
        settings=GCSToolSettings(local_file_root=str(tmp_path)),
    )
    assert result["status"] == "SUCCESS"
    mock_blob.upload_from_filename.assert_called_once_with(
        str(tmp_path.resolve() / "path" / "to" / "file.txt")
    )


def test_create_object_from_file_without_local_file_root():
  """Test create_object refuses a local file when no root is configured."""
  with mock.patch.object(
      client, "get_gcs_client", autospec=True
  ) as mock_get_client:
    mock_client = mock.MagicMock()
    mock_get_client.return_value = mock_client
    mock_bucket = mock.MagicMock()
    mock_client.get_bucket.return_value = mock_bucket
    mock_blob = mock.MagicMock()
    mock_bucket.blob.return_value = mock_blob

    creds = mock.create_autospec(Credentials, instance=True)
    result = storage_tool.create_object(
        bucket_name="test-bucket",
        object_name="test-object",
        source_file_path="secrets.json",
        credentials=creds,
    )
    assert result["status"] == "ERROR"
    assert "Local file access is disabled" in result["error_details"]
    mock_blob.upload_from_filename.assert_not_called()


def test_create_object_from_file_rejects_absolute_path(tmp_path):
  """Test create_object refuses an absolute source_file_path."""
  with mock.patch.object(
      client, "get_gcs_client", autospec=True
  ) as mock_get_client:
    mock_client = mock.MagicMock()
    mock_get_client.return_value = mock_client
    mock_bucket = mock.MagicMock()
    mock_client.get_bucket.return_value = mock_bucket
    mock_blob = mock.MagicMock()
    mock_bucket.blob.return_value = mock_blob

    creds = mock.create_autospec(Credentials, instance=True)
    result = storage_tool.create_object(
        bucket_name="test-bucket",
        object_name="test-object",
        source_file_path=str(tmp_path / "secrets.json"),
        credentials=creds,
        settings=GCSToolSettings(local_file_root=str(tmp_path)),
    )
    assert result["status"] == "ERROR"
    assert "must be relative" in result["error_details"]
    mock_blob.upload_from_filename.assert_not_called()


def test_create_object_no_data():
  """Test create_object function when neither data nor source_file_path is provided."""
  with mock.patch.object(
      client, "get_gcs_client", autospec=True
  ) as mock_get_client:
    mock_client = mock.MagicMock()
    mock_get_client.return_value = mock_client
    mock_bucket = mock.MagicMock()
    mock_client.get_bucket.return_value = mock_bucket
    mock_blob = mock.MagicMock()
    mock_bucket.blob.return_value = mock_blob

    creds = mock.create_autospec(Credentials, instance=True)
    result = storage_tool.create_object(
        bucket_name="test-bucket",
        object_name="test-object",
        credentials=creds,
    )
    assert result["status"] == "ERROR"
    assert "must be provided" in result["error_details"]


def test_get_object_data():
  """Test get_object_data function."""
  with mock.patch.object(
      client, "get_gcs_client", autospec=True
  ) as mock_get_client:
    mock_client = mock.MagicMock()
    mock_get_client.return_value = mock_client
    mock_bucket = mock.MagicMock()
    mock_client.get_bucket.return_value = mock_bucket
    mock_blob = mock.MagicMock()
    mock_bucket.get_blob.return_value = mock_blob
    mock_blob.download_as_bytes.return_value = b"content"

    creds = mock.create_autospec(Credentials, instance=True)
    result = storage_tool.get_object_data(
        bucket_name="test-bucket",
        object_name="test-object",
        credentials=creds,
        generation=1,
    )
    assert result == {
        "status": "SUCCESS",
        "results": "content",
        "encoding": "text",
    }
    mock_bucket.get_blob.assert_called_once_with("test-object", generation=1)


def test_get_object_data_no_generation():
  """Test get_object_data function without generation parameter."""
  with mock.patch.object(
      client, "get_gcs_client", autospec=True
  ) as mock_get_client:
    mock_client = mock.MagicMock()
    mock_get_client.return_value = mock_client
    mock_bucket = mock.MagicMock()
    mock_client.get_bucket.return_value = mock_bucket
    mock_blob = mock.MagicMock()
    mock_bucket.get_blob.return_value = mock_blob
    mock_blob.download_as_bytes.return_value = b"\xff\xff"

    creds = mock.create_autospec(Credentials, instance=True)
    result = storage_tool.get_object_data(
        bucket_name="test-bucket",
        object_name="test-object",
        credentials=creds,
    )
    assert result == {
        "status": "SUCCESS",
        "results": "//8=",
        "encoding": "base64",
    }
    mock_bucket.get_blob.assert_called_once_with("test-object")


def test_get_object_data_to_file(tmp_path):
  """Test get_object_data function downloading directly to destination_file_path."""
  with mock.patch.object(
      client, "get_gcs_client", autospec=True
  ) as mock_get_client:
    mock_client = mock.MagicMock()
    mock_get_client.return_value = mock_client
    mock_bucket = mock.MagicMock()
    mock_client.get_bucket.return_value = mock_bucket
    mock_blob = mock.MagicMock()
    mock_bucket.get_blob.return_value = mock_blob

    creds = mock.create_autospec(Credentials, instance=True)
    result = storage_tool.get_object_data(
        bucket_name="test-bucket",
        object_name="test-object",
        destination_file_path="path/to/download.txt",
        credentials=creds,
        settings=GCSToolSettings(local_file_root=str(tmp_path)),
    )
    assert result["status"] == "SUCCESS"
    mock_blob.download_to_filename.assert_called_once_with(
        str(tmp_path.resolve() / "path" / "to" / "download.txt")
    )


def test_get_object_data_to_file_creates_parent_directories(tmp_path):
  """Test get_object_data creates missing parent directories under the root."""
  with mock.patch.object(
      client, "get_gcs_client", autospec=True
  ) as mock_get_client:
    mock_client = mock.MagicMock()
    mock_get_client.return_value = mock_client
    mock_bucket = mock.MagicMock()
    mock_client.get_bucket.return_value = mock_bucket
    mock_blob = mock.MagicMock()
    mock_bucket.get_blob.return_value = mock_blob

    creds = mock.create_autospec(Credentials, instance=True)
    result = storage_tool.get_object_data(
        bucket_name="test-bucket",
        object_name="test-object",
        destination_file_path="downloads/report.pdf",
        credentials=creds,
        settings=GCSToolSettings(local_file_root=str(tmp_path)),
    )
    assert result["status"] == "SUCCESS"
    assert (tmp_path / "downloads").is_dir()
    mock_blob.download_to_filename.assert_called_once_with(
        str(tmp_path.resolve() / "downloads" / "report.pdf")
    )


def test_get_object_data_to_file_without_local_file_root():
  """Test get_object_data refuses a download when no root is configured."""
  with mock.patch.object(
      client, "get_gcs_client", autospec=True
  ) as mock_get_client:
    mock_client = mock.MagicMock()
    mock_get_client.return_value = mock_client
    mock_bucket = mock.MagicMock()
    mock_client.get_bucket.return_value = mock_bucket
    mock_blob = mock.MagicMock()
    mock_bucket.get_blob.return_value = mock_blob

    creds = mock.create_autospec(Credentials, instance=True)
    result = storage_tool.get_object_data(
        bucket_name="test-bucket",
        object_name="test-object",
        destination_file_path="agent/main.py",
        credentials=creds,
    )
    assert result["status"] == "ERROR"
    assert "Local file access is disabled" in result["error_details"]
    mock_blob.download_to_filename.assert_not_called()


def test_get_object_data_to_file_rejects_path_escaping_root(tmp_path):
  """Test get_object_data refuses a destination outside the configured root."""
  root = tmp_path / "root"
  root.mkdir()
  with mock.patch.object(
      client, "get_gcs_client", autospec=True
  ) as mock_get_client:
    mock_client = mock.MagicMock()
    mock_get_client.return_value = mock_client
    mock_bucket = mock.MagicMock()
    mock_client.get_bucket.return_value = mock_bucket
    mock_blob = mock.MagicMock()
    mock_bucket.get_blob.return_value = mock_blob

    creds = mock.create_autospec(Credentials, instance=True)
    result = storage_tool.get_object_data(
        bucket_name="test-bucket",
        object_name="test-object",
        destination_file_path="../outside.txt",
        credentials=creds,
        settings=GCSToolSettings(local_file_root=str(root)),
    )
    assert result["status"] == "ERROR"
    assert "escapes the configured root" in result["error_details"]
    mock_blob.download_to_filename.assert_not_called()


def test_get_object_data_to_file_rejects_symlink_escaping_root(tmp_path):
  """Test get_object_data resolves symlinks before confining the destination."""
  root = tmp_path / "root"
  root.mkdir()
  outside = tmp_path / "outside"
  outside.mkdir()
  (root / "link").symlink_to(outside, target_is_directory=True)
  with mock.patch.object(
      client, "get_gcs_client", autospec=True
  ) as mock_get_client:
    mock_client = mock.MagicMock()
    mock_get_client.return_value = mock_client
    mock_bucket = mock.MagicMock()
    mock_client.get_bucket.return_value = mock_bucket
    mock_blob = mock.MagicMock()
    mock_bucket.get_blob.return_value = mock_blob

    creds = mock.create_autospec(Credentials, instance=True)
    result = storage_tool.get_object_data(
        bucket_name="test-bucket",
        object_name="test-object",
        destination_file_path="link/escaped.txt",
        credentials=creds,
        settings=GCSToolSettings(local_file_root=str(root)),
    )
    assert result["status"] == "ERROR"
    assert "escapes the configured root" in result["error_details"]
    mock_blob.download_to_filename.assert_not_called()


def test_delete_objects():
  """Test delete_objects function."""
  with mock.patch.object(
      client, "get_gcs_client", autospec=True
  ) as mock_get_client:
    mock_client = mock.MagicMock()
    mock_get_client.return_value = mock_client
    mock_bucket = mock.MagicMock()
    mock_client.get_bucket.return_value = mock_bucket

    creds = mock.create_autospec(Credentials, instance=True)
    result = storage_tool.delete_objects(
        bucket_name="test-bucket",
        object_names=["test-object"],
        credentials=creds,
    )
    assert result["status"] == "SUCCESS"
    mock_bucket.delete_blobs.assert_called_once_with(blobs=["test-object"])


def test_get_bucket_deprecated():
  """Test get_bucket function in storage_tool is deprecated but works."""
  with mock.patch(
      "google.adk.integrations.gcs.admin_tool.get_bucket", autospec=True
  ) as mock_admin_get_bucket:
    mock_admin_get_bucket.return_value = {"status": "SUCCESS", "results": {}}
    creds = mock.create_autospec(Credentials, instance=True)

    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      result = storage_tool.get_bucket(
          bucket_name="test-bucket", credentials=creds
      )

      assert len(w) == 1
      assert issubclass(w[0].category, DeprecationWarning)
      assert "deprecated" in str(w[0].message)

    mock_admin_get_bucket.assert_called_once_with(
        bucket_name="test-bucket", credentials=creds
    )
    assert result == {"status": "SUCCESS", "results": {}}
