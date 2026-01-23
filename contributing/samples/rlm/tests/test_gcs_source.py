"""
Unit tests for GCSFileSource with mocked GCS client.

These tests do not require actual GCS access - all GCS operations are mocked.
"""

from datetime import datetime
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

# Skip all tests if google-cloud-storage is not installed
pytest.importorskip("google.cloud.storage")


@pytest.fixture
def mock_storage():
  """Mock google.cloud.storage module."""
  with patch("adk_rlm.files.sources.gcs.storage") as mock_storage:
    mock_client = MagicMock()
    mock_storage.Client.return_value = mock_client
    mock_storage.Client.from_service_account_json.return_value = mock_client
    yield mock_storage, mock_client


@pytest.fixture
def gcs_source(mock_storage):
  """Create GCSFileSource with mocked client."""
  from adk_rlm.files.sources.gcs import GCSFileSource

  _, mock_client = mock_storage
  source = GCSFileSource(bucket="test-bucket")
  source.client = mock_client
  return source


class TestGCSFileSourceInit:
  """Test GCSFileSource initialization."""

  def test_init_with_default_credentials(self, mock_storage):
    """Test initialization with Application Default Credentials."""
    from adk_rlm.files.sources.gcs import GCSFileSource

    mock_storage_mod, _ = mock_storage
    source = GCSFileSource(bucket="test-bucket")

    assert source.default_bucket == "test-bucket"
    assert source.source_type == "gcs"
    mock_storage_mod.Client.assert_called_once()

  def test_init_with_service_account(self, mock_storage):
    """Test initialization with service account JSON."""
    from adk_rlm.files.sources.gcs import GCSFileSource

    mock_storage_mod, _ = mock_storage
    GCSFileSource(
        bucket="test-bucket",
        credentials_path="/path/to/sa.json",
        project="my-project",
    )

    mock_storage_mod.Client.from_service_account_json.assert_called_once_with(
        "/path/to/sa.json", project="my-project"
    )

  def test_init_with_custom_settings(self, mock_storage):
    """Test initialization with custom timeout and retry config."""
    from adk_rlm.files.sources.gcs import GCSFileSource
    from adk_rlm.files.sources.gcs import RetryConfig

    retry_config = RetryConfig(max_attempts=5, initial_delay=1.0)
    source = GCSFileSource(
        bucket="test-bucket",
        timeout=120.0,
        retry_config=retry_config,
        max_concurrent=5,
        large_file_threshold=50_000_000,
    )

    assert source.timeout == 120.0
    assert source.retry_config.max_attempts == 5
    assert source.max_concurrent == 5
    assert source.large_file_threshold == 50_000_000


class TestPathParsing:
  """Test GCS path parsing."""

  def test_parse_path_with_gs_prefix(self, gcs_source):
    """Test parsing gs:// URIs."""
    bucket, blob = gcs_source._parse_path("gs://my-bucket/path/to/file.pdf")
    assert bucket == "my-bucket"
    assert blob == "path/to/file.pdf"

  def test_parse_path_with_gs_prefix_root(self, gcs_source):
    """Test parsing gs:// URI with file at root."""
    bucket, blob = gcs_source._parse_path("gs://my-bucket/file.pdf")
    assert bucket == "my-bucket"
    assert blob == "file.pdf"

  def test_parse_path_without_prefix(self, gcs_source):
    """Test parsing paths without gs:// uses default bucket."""
    bucket, blob = gcs_source._parse_path("path/to/file.pdf")
    assert bucket == "test-bucket"
    assert blob == "path/to/file.pdf"

  def test_parse_path_no_bucket_raises(self, mock_storage):
    """Test that missing bucket raises ValueError."""
    from adk_rlm.files.sources.gcs import GCSFileSource

    source = GCSFileSource()  # No default bucket
    with pytest.raises(ValueError, match="No bucket specified"):
      source._parse_path("path/to/file.pdf")

  def test_parse_path_empty_blob_name(self, gcs_source):
    """Test parsing gs:// with only bucket."""
    bucket, blob = gcs_source._parse_path("gs://my-bucket")
    assert bucket == "my-bucket"
    assert blob == ""


class TestResolve:
  """Test path resolution including glob patterns."""

  def test_resolve_single_file_exists(self, gcs_source):
    """Test resolving a single file that exists."""
    mock_bucket = MagicMock()
    mock_blob = MagicMock()
    mock_blob.exists.return_value = True
    mock_bucket.blob.return_value = mock_blob
    gcs_source.client.bucket.return_value = mock_bucket

    result = gcs_source.resolve("gs://test-bucket/file.pdf")

    assert result == ["gs://test-bucket/file.pdf"]
    mock_blob.exists.assert_called_once()

  def test_resolve_single_file_not_exists(self, gcs_source):
    """Test resolving a file that doesn't exist."""
    mock_bucket = MagicMock()
    mock_blob = MagicMock()
    mock_blob.exists.return_value = False
    mock_bucket.blob.return_value = mock_blob
    gcs_source.client.bucket.return_value = mock_bucket

    result = gcs_source.resolve("gs://test-bucket/missing.pdf")

    assert result == []

  def test_resolve_glob_pattern(self, gcs_source):
    """Test resolving glob patterns."""
    mock_bucket = MagicMock()
    mock_blob1 = MagicMock()
    mock_blob1.name = "data/file1.pdf"
    mock_blob2 = MagicMock()
    mock_blob2.name = "data/file2.pdf"
    mock_blob3 = MagicMock()
    mock_blob3.name = "data/file.txt"

    mock_bucket.list_blobs.return_value = [mock_blob1, mock_blob2, mock_blob3]
    gcs_source.client.bucket.return_value = mock_bucket

    result = gcs_source.resolve("gs://test-bucket/data/*.pdf")

    assert result == [
        "gs://test-bucket/data/file1.pdf",
        "gs://test-bucket/data/file2.pdf",
    ]
    mock_bucket.list_blobs.assert_called_once()

  def test_resolve_recursive_glob(self, gcs_source):
    """Test resolving ** recursive patterns."""
    mock_bucket = MagicMock()
    mock_blob1 = MagicMock()
    mock_blob1.name = "data/2024/report.pdf"
    mock_blob2 = MagicMock()
    mock_blob2.name = "data/2023/report.pdf"
    mock_blob3 = MagicMock()
    mock_blob3.name = "data/readme.txt"

    mock_bucket.list_blobs.return_value = [mock_blob1, mock_blob2, mock_blob3]
    gcs_source.client.bucket.return_value = mock_bucket

    result = gcs_source.resolve("gs://test-bucket/data/**/*.pdf")

    assert len(result) == 2
    assert "gs://test-bucket/data/2024/report.pdf" in result
    assert "gs://test-bucket/data/2023/report.pdf" in result

  def test_resolve_glob_no_matches(self, gcs_source):
    """Test glob pattern with no matches."""
    mock_bucket = MagicMock()
    mock_bucket.list_blobs.return_value = []
    gcs_source.client.bucket.return_value = mock_bucket

    result = gcs_source.resolve("gs://test-bucket/data/*.pdf")

    assert result == []


class TestGetMetadata:
  """Test metadata fetching."""

  def test_get_metadata(self, gcs_source):
    """Test fetching metadata without downloading content."""
    mock_bucket = MagicMock()
    mock_blob = MagicMock()
    mock_blob.size = 1024
    mock_blob.content_type = "application/pdf"
    mock_blob.updated = datetime(2024, 1, 15, 10, 30, 0)
    mock_blob.content_encoding = None
    mock_blob.storage_class = "STANDARD"
    mock_blob.generation = 12345
    mock_blob.metageneration = 1
    mock_blob.etag = "abc123"
    mock_blob.md5_hash = "xyz789"
    mock_blob.crc32c = "crc123"
    mock_bucket.blob.return_value = mock_blob
    gcs_source.client.bucket.return_value = mock_bucket

    metadata = gcs_source.get_metadata("gs://test-bucket/report.pdf")

    assert metadata.name == "report.pdf"
    assert metadata.size_bytes == 1024
    assert metadata.mime_type == "application/pdf"
    assert metadata.source_type == "gcs"
    assert metadata.extra["bucket"] == "test-bucket"
    assert metadata.extra["storage_class"] == "STANDARD"
    mock_blob.reload.assert_called_once()

  def test_get_metadata_guesses_mime_type(self, gcs_source):
    """Test that MIME type is guessed when not provided by GCS."""
    mock_bucket = MagicMock()
    mock_blob = MagicMock()
    mock_blob.size = 500
    mock_blob.content_type = None  # GCS didn't provide MIME type
    mock_blob.updated = None
    mock_blob.content_encoding = None
    mock_blob.storage_class = "STANDARD"
    mock_blob.generation = 1
    mock_blob.metageneration = 1
    mock_blob.etag = None
    mock_blob.md5_hash = None
    mock_blob.crc32c = None
    mock_bucket.blob.return_value = mock_blob
    gcs_source.client.bucket.return_value = mock_bucket

    metadata = gcs_source.get_metadata("gs://test-bucket/data.json")

    assert metadata.mime_type == "application/json"


class TestLoad:
  """Test file loading."""

  def test_load_small_file(self, gcs_source):
    """Test loading a small file directly into memory."""
    mock_bucket = MagicMock()
    mock_blob = MagicMock()
    mock_blob.size = 1024
    mock_blob.content_type = "text/plain"
    mock_blob.updated = datetime.now()
    mock_blob.content_encoding = None
    mock_blob.storage_class = "STANDARD"
    mock_blob.generation = 1
    mock_blob.metageneration = 1
    mock_blob.etag = "abc"
    mock_blob.md5_hash = None
    mock_blob.crc32c = None
    mock_blob.download_as_bytes.return_value = b"Hello, world!"
    mock_bucket.blob.return_value = mock_blob
    gcs_source.client.bucket.return_value = mock_bucket

    loaded = gcs_source.load("gs://test-bucket/file.txt")

    assert loaded.content == b"Hello, world!"
    assert loaded.metadata.name == "file.txt"
    assert loaded.metadata.source_type == "gcs"

  def test_load_uses_chunked_for_large_files(self, gcs_source):
    """Test that large files use chunked loading strategy."""
    gcs_source.large_file_threshold = 1000  # Low threshold for testing

    mock_bucket = MagicMock()
    mock_blob = MagicMock()
    mock_blob.size = 2000  # Above threshold
    mock_blob.content_type = "application/octet-stream"
    mock_blob.updated = None
    mock_blob.content_encoding = None
    mock_blob.storage_class = "STANDARD"
    mock_blob.generation = 1
    mock_blob.metageneration = 1
    mock_blob.etag = None
    mock_blob.md5_hash = None
    mock_blob.crc32c = None

    # Mock the download_to_file to write to temp file
    def download_side_effect(file_obj, timeout=None):
      file_obj.write(b"large file content")

    mock_blob.download_to_file.side_effect = download_side_effect
    mock_bucket.blob.return_value = mock_blob
    gcs_source.client.bucket.return_value = mock_bucket

    loaded = gcs_source.load("gs://test-bucket/large.bin")

    assert loaded.content == b"large file content"
    mock_blob.download_to_file.assert_called_once()


class TestLoadErrors:
  """Test error handling during load."""

  def test_load_not_found_raises(self, gcs_source):
    """Test loading a nonexistent file raises FileNotFoundError."""
    from google.cloud.exceptions import NotFound

    mock_bucket = MagicMock()
    mock_blob = MagicMock()
    mock_blob.reload.side_effect = NotFound("Not found")
    mock_bucket.blob.return_value = mock_blob
    gcs_source.client.bucket.return_value = mock_bucket

    with pytest.raises(FileNotFoundError):
      gcs_source.load("gs://test-bucket/missing.txt")

  def test_load_permission_denied_raises(self, gcs_source):
    """Test loading without permission raises PermissionError."""
    from google.cloud.exceptions import Forbidden

    mock_bucket = MagicMock()
    mock_blob = MagicMock()
    mock_blob.reload.side_effect = Forbidden("Access denied")
    mock_bucket.blob.return_value = mock_blob
    gcs_source.client.bucket.return_value = mock_bucket

    with pytest.raises(PermissionError, match="Access denied"):
      gcs_source.load("gs://test-bucket/secret.txt")


class TestRetryLogic:
  """Test retry behavior for transient errors."""

  def test_retry_on_transient_error(self, mock_storage):
    """Test that transient errors trigger retries."""
    from adk_rlm.files.sources.gcs import GCSFileSource
    from adk_rlm.files.sources.gcs import RetryConfig

    _, mock_client = mock_storage
    source = GCSFileSource(
        bucket="test-bucket",
        retry_config=RetryConfig(max_attempts=3, initial_delay=0.01),
    )
    source.client = mock_client

    mock_bucket = MagicMock()
    mock_blob = MagicMock()
    mock_blob.size = 100
    mock_blob.content_type = "text/plain"
    mock_blob.updated = None
    mock_blob.content_encoding = None
    mock_blob.storage_class = "STANDARD"
    mock_blob.generation = 1
    mock_blob.metageneration = 1
    mock_blob.etag = None
    mock_blob.md5_hash = None
    mock_blob.crc32c = None

    # Fail twice, then succeed for all subsequent calls
    call_count = 0

    def reload_side_effect(*args, **kwargs):
      nonlocal call_count
      call_count += 1
      if call_count < 3:
        raise Exception("ServiceUnavailable")
      return mock_blob

    mock_blob.reload.side_effect = reload_side_effect
    mock_blob.download_as_bytes.return_value = b"data"
    mock_bucket.blob.return_value = mock_blob
    mock_client.bucket.return_value = mock_bucket

    loaded = source.load("gs://test-bucket/file.txt")

    assert loaded.content == b"data"
    # load() calls get_metadata() first (which retries reload 3 times),
    # then _load_direct() which does a best-effort reload after download
    assert call_count >= 3

  def test_max_retries_exceeded(self, mock_storage):
    """Test that exceeding max retries raises error."""
    from adk_rlm.files.sources.gcs import GCSFileSource
    from adk_rlm.files.sources.gcs import RetryConfig

    _, mock_client = mock_storage
    source = GCSFileSource(
        bucket="test-bucket",
        retry_config=RetryConfig(max_attempts=2, initial_delay=0.01),
    )
    source.client = mock_client

    mock_bucket = MagicMock()
    mock_blob = MagicMock()
    mock_blob.reload.side_effect = Exception("ServiceUnavailable")
    mock_bucket.blob.return_value = mock_blob
    mock_client.bucket.return_value = mock_bucket

    with pytest.raises(RuntimeError, match="failed after 2 attempts"):
      source.load("gs://test-bucket/file.txt")

  def test_non_retryable_error_not_retried(self, mock_storage):
    """Test that non-retryable errors are not retried."""
    from adk_rlm.files.sources.gcs import GCSFileSource
    from adk_rlm.files.sources.gcs import RetryConfig

    _, mock_client = mock_storage
    source = GCSFileSource(
        bucket="test-bucket",
        retry_config=RetryConfig(max_attempts=3, initial_delay=0.01),
    )
    source.client = mock_client

    mock_bucket = MagicMock()
    mock_blob = MagicMock()

    call_count = 0

    def reload_side_effect(*args, **kwargs):
      nonlocal call_count
      call_count += 1
      raise ValueError("Bad request - not retryable")

    mock_blob.reload.side_effect = reload_side_effect
    mock_bucket.blob.return_value = mock_blob
    mock_client.bucket.return_value = mock_bucket

    with pytest.raises(ValueError, match="Bad request"):
      source.load("gs://test-bucket/file.txt")

    # Should only try once (no retries)
    assert call_count == 1


class TestLoadMany:
  """Test parallel loading functionality."""

  def test_load_many_empty_list(self, gcs_source):
    """Test loading empty list."""
    results = list(gcs_source.load_many([]))
    assert results == []

  def test_load_many_single_file(self, gcs_source):
    """Test loading single file doesn't use parallelism."""
    mock_bucket = MagicMock()
    mock_blob = MagicMock()
    mock_blob.size = 100
    mock_blob.content_type = "text/plain"
    mock_blob.updated = None
    mock_blob.content_encoding = None
    mock_blob.storage_class = "STANDARD"
    mock_blob.generation = 1
    mock_blob.metageneration = 1
    mock_blob.etag = None
    mock_blob.md5_hash = None
    mock_blob.crc32c = None
    mock_blob.download_as_bytes.return_value = b"content"
    mock_bucket.blob.return_value = mock_blob
    gcs_source.client.bucket.return_value = mock_bucket

    results = list(gcs_source.load_many(["gs://test-bucket/file.txt"]))

    assert len(results) == 1
    assert results[0].content == b"content"

  def test_load_many_parallel(self, gcs_source):
    """Test loading multiple files in parallel."""

    def make_blob(name, content):
      blob = MagicMock()
      blob.size = len(content)
      blob.content_type = "text/plain"
      blob.updated = None
      blob.content_encoding = None
      blob.storage_class = "STANDARD"
      blob.generation = 1
      blob.metageneration = 1
      blob.etag = None
      blob.md5_hash = None
      blob.crc32c = None
      blob.download_as_bytes.return_value = content
      return blob

    mock_bucket = MagicMock()
    blobs = {
        "file1.txt": make_blob("file1.txt", b"content1"),
        "file2.txt": make_blob("file2.txt", b"content2"),
        "file3.txt": make_blob("file3.txt", b"content3"),
    }

    def get_blob(name):
      blob_name = name.split("/")[-1] if "/" in name else name
      return blobs.get(blob_name, MagicMock())

    mock_bucket.blob.side_effect = get_blob
    gcs_source.client.bucket.return_value = mock_bucket

    paths = [
        "gs://test-bucket/file1.txt",
        "gs://test-bucket/file2.txt",
        "gs://test-bucket/file3.txt",
    ]
    results = list(gcs_source.load_many(paths))

    assert len(results) == 3
    contents = {r.content for r in results}
    assert contents == {b"content1", b"content2", b"content3"}


class TestExists:
  """Test existence checking."""

  def test_exists_true(self, gcs_source):
    """Test exists returns True for existing blob."""
    mock_bucket = MagicMock()
    mock_blob = MagicMock()
    mock_blob.exists.return_value = True
    mock_bucket.blob.return_value = mock_blob
    gcs_source.client.bucket.return_value = mock_bucket

    assert gcs_source.exists("gs://test-bucket/file.txt") is True

  def test_exists_false(self, gcs_source):
    """Test exists returns False for missing blob."""
    mock_bucket = MagicMock()
    mock_blob = MagicMock()
    mock_blob.exists.return_value = False
    mock_bucket.blob.return_value = mock_blob
    gcs_source.client.bucket.return_value = mock_bucket

    assert gcs_source.exists("gs://test-bucket/missing.txt") is False

  def test_exists_handles_errors(self, gcs_source):
    """Test exists returns False on errors."""
    mock_bucket = MagicMock()
    mock_blob = MagicMock()
    mock_blob.exists.side_effect = Exception("Network error")
    mock_bucket.blob.return_value = mock_blob
    gcs_source.client.bucket.return_value = mock_bucket

    assert gcs_source.exists("gs://test-bucket/file.txt") is False


class TestLazyFileIntegration:
  """Test GCSFileSource with LazyFile."""

  def test_lazy_file_level_0(self, mock_storage):
    """Test Level 0 access (name/extension) requires no I/O."""
    from adk_rlm.files.lazy import LazyFile
    from adk_rlm.files.sources.gcs import GCSFileSource

    _, mock_client = mock_storage
    source = GCSFileSource(bucket="test-bucket")
    source.client = mock_client

    lazy = LazyFile(path="gs://test-bucket/data/report.pdf", source=source)

    # Level 0 - no I/O
    assert lazy.name == "report.pdf"
    assert lazy.extension == ".pdf"
    assert lazy.level == 0

    # Verify no GCS calls were made
    mock_client.bucket.assert_not_called()

  def test_lazy_file_level_1(self, mock_storage):
    """Test Level 1 access (metadata) triggers reload."""
    from adk_rlm.files.lazy import LazyFile
    from adk_rlm.files.sources.gcs import GCSFileSource

    _, mock_client = mock_storage
    source = GCSFileSource(bucket="test-bucket")
    source.client = mock_client

    mock_bucket = MagicMock()
    mock_blob = MagicMock()
    mock_blob.size = 2048
    mock_blob.content_type = "application/pdf"
    mock_blob.updated = datetime.now()
    mock_blob.content_encoding = None
    mock_blob.storage_class = "STANDARD"
    mock_blob.generation = 1
    mock_blob.metageneration = 1
    mock_blob.etag = "abc"
    mock_blob.md5_hash = "xyz"
    mock_blob.crc32c = "crc"
    mock_bucket.blob.return_value = mock_blob
    mock_client.bucket.return_value = mock_bucket

    lazy = LazyFile(path="gs://test-bucket/report.pdf", source=source)

    # Level 1 - triggers metadata fetch
    size = lazy.size

    assert size == 2048
    assert lazy.level == 1
    mock_blob.reload.assert_called_once()


class TestFileLoaderIntegration:
  """Test GCS source integration with FileLoader."""

  def test_file_loader_gcs_detection(self, mock_storage):
    """Test FileLoader auto-detects gs:// paths."""
    from adk_rlm.files.loader import FileLoader
    from adk_rlm.files.sources.gcs import GCSFileSource

    _, mock_client = mock_storage
    gcs_source = GCSFileSource(bucket="test-bucket")
    gcs_source.client = mock_client

    # Mock resolve to return the path
    mock_bucket = MagicMock()
    mock_blob = MagicMock()
    mock_blob.exists.return_value = True
    mock_bucket.blob.return_value = mock_blob
    mock_client.bucket.return_value = mock_bucket

    loader = FileLoader(sources={"gcs": gcs_source})
    collection = loader.create_lazy_files(["gs://test-bucket/file.txt"])

    assert len(collection) == 1
    assert collection[0].name == "file.txt"

  def test_file_loader_gcs_not_configured_raises(self):
    """Test FileLoader raises clear error when GCS not configured."""
    from adk_rlm.files.loader import FileLoader

    loader = FileLoader()

    with pytest.raises(ValueError, match="GCS source not configured"):
      loader.create_lazy_files(["gs://some-bucket/file.txt"])
