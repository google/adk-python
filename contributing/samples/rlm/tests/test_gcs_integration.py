"""
Integration tests for GCSFileSource with real GCS access.

These tests require actual GCS access and are skipped unless
the RLM_GCS_TEST_BUCKET environment variable is set.

Required environment variables:
- RLM_GCS_TEST_BUCKET: GCS bucket name for testing
- RLM_GCS_TEST_FILE: (optional) Path to test file in bucket (default: test/sample.txt)
- RLM_GCS_TEST_PREFIX: (optional) Prefix for glob tests (default: test)
- RLM_GCS_TEST_PROJECT: (optional) GCP project ID

To run these tests:
    RLM_GCS_TEST_BUCKET=my-test-bucket pytest tests/test_gcs_integration.py -v
"""

import os

import pytest

# Skip all tests if GCS bucket not configured
pytestmark = pytest.mark.skipif(
    not os.environ.get("RLM_GCS_TEST_BUCKET"),
    reason="GCS integration tests require RLM_GCS_TEST_BUCKET env var",
)

# Also skip if google-cloud-storage is not installed
pytest.importorskip("google.cloud.storage")


@pytest.fixture
def gcs_source():
  """Create GCSFileSource for integration tests."""
  from adk_rlm.files.sources.gcs import GCSFileSource

  return GCSFileSource(
      bucket=os.environ["RLM_GCS_TEST_BUCKET"],
      project=os.environ.get("RLM_GCS_TEST_PROJECT"),
  )


@pytest.fixture
def test_bucket():
  """Get test bucket name."""
  return os.environ["RLM_GCS_TEST_BUCKET"]


@pytest.fixture
def test_file_path():
  """Path to test file in GCS bucket."""
  return os.environ.get("RLM_GCS_TEST_FILE", "test/sample.txt")


@pytest.fixture
def test_prefix():
  """Prefix for glob pattern tests."""
  return os.environ.get("RLM_GCS_TEST_PREFIX", "test")


class TestGCSIntegration:
  """Integration tests requiring real GCS access."""

  def test_source_type(self, gcs_source):
    """Test source type is 'gcs'."""
    assert gcs_source.source_type == "gcs"

  def test_exists_real_file(self, gcs_source, test_bucket, test_file_path):
    """Test checking existence of a real file."""
    result = gcs_source.exists(f"gs://{test_bucket}/{test_file_path}")
    assert result is True

  def test_exists_nonexistent_file(self, gcs_source, test_bucket):
    """Test exists returns False for missing file."""
    result = gcs_source.exists(f"gs://{test_bucket}/nonexistent-file-12345.txt")
    assert result is False

  def test_get_metadata(self, gcs_source, test_bucket, test_file_path):
    """Test fetching real file metadata."""
    path = f"gs://{test_bucket}/{test_file_path}"

    metadata = gcs_source.get_metadata(path)

    assert metadata.name == test_file_path.split("/")[-1]
    assert metadata.size_bytes > 0
    assert metadata.source_type == "gcs"
    assert metadata.path == path
    assert "bucket" in metadata.extra
    assert metadata.extra["bucket"] == test_bucket

  def test_load(self, gcs_source, test_bucket, test_file_path):
    """Test loading real file content."""
    path = f"gs://{test_bucket}/{test_file_path}"

    loaded = gcs_source.load(path)

    assert len(loaded.content) > 0
    assert loaded.metadata.path == path
    assert loaded.metadata.source_type == "gcs"

  def test_load_not_found(self, gcs_source, test_bucket):
    """Test proper error for missing file."""
    with pytest.raises(FileNotFoundError):
      gcs_source.load(f"gs://{test_bucket}/nonexistent-file-12345.txt")

  def test_resolve_single_file(self, gcs_source, test_bucket, test_file_path):
    """Test resolving a single file path."""
    path = f"gs://{test_bucket}/{test_file_path}"

    result = gcs_source.resolve(path)

    assert result == [path]

  def test_resolve_nonexistent_file(self, gcs_source, test_bucket):
    """Test resolving a nonexistent file returns empty list."""
    path = f"gs://{test_bucket}/nonexistent-file-12345.txt"

    result = gcs_source.resolve(path)

    assert result == []

  def test_resolve_glob_pattern(self, gcs_source, test_bucket, test_prefix):
    """Test glob pattern resolution."""
    pattern = f"gs://{test_bucket}/{test_prefix}/*"

    paths = gcs_source.resolve(pattern)

    # May return empty if no files, but should not error
    assert isinstance(paths, list)
    for path in paths:
      assert path.startswith(f"gs://{test_bucket}/")

  def test_load_many_single(self, gcs_source, test_bucket, test_file_path):
    """Test load_many with single file."""
    path = f"gs://{test_bucket}/{test_file_path}"

    results = list(gcs_source.load_many([path]))

    assert len(results) == 1
    assert len(results[0].content) > 0

  def test_load_many_multiple(self, gcs_source, test_bucket, test_file_path):
    """Test load_many with same file twice (tests parallelism)."""
    path = f"gs://{test_bucket}/{test_file_path}"
    paths = [path, path]

    results = list(gcs_source.load_many(paths))

    assert len(results) == 2


class TestGCSWithFileLoader:
  """Test GCS source integration with FileLoader."""

  def test_file_loader_with_gcs(self, gcs_source, test_bucket, test_file_path):
    """Test FileLoader works with GCS source."""
    from adk_rlm.files.loader import FileLoader

    path = f"gs://{test_bucket}/{test_file_path}"

    loader = FileLoader(sources={"gcs": gcs_source})
    collection = loader.create_lazy_files([path])

    assert len(collection) == 1
    assert collection[0].name == test_file_path.split("/")[-1]

  def test_lazy_loading_with_gcs(self, gcs_source, test_bucket, test_file_path):
    """Test lazy file loading from GCS."""
    from adk_rlm.files.loader import FileLoader

    path = f"gs://{test_bucket}/{test_file_path}"

    loader = FileLoader(sources={"gcs": gcs_source})
    collection = loader.create_lazy_files([path])

    lazy_file = collection[0]

    # Level 0 - no I/O yet
    assert lazy_file.level == 0
    _ = lazy_file.name  # Still no I/O

    # Level 1 - metadata fetch
    size = lazy_file.size
    assert lazy_file.level == 1
    assert size > 0

  def test_lazy_file_content_access(
      self, gcs_source, test_bucket, test_file_path
  ):
    """Test accessing lazy file content triggers download."""
    from adk_rlm.files.loader import FileLoader
    from adk_rlm.files.parsers.text import TextParser

    path = f"gs://{test_bucket}/{test_file_path}"

    loader = FileLoader(sources={"gcs": gcs_source}, parsers=[TextParser()])
    collection = loader.create_lazy_files([path])

    lazy_file = collection[0]

    # Access raw content (no parsing needed)
    raw_content = lazy_file.read()
    assert len(raw_content) > 0
    assert lazy_file.level >= 1  # At least metadata loaded


class TestGCSRetryBehavior:
  """Test retry behavior with real GCS (best-effort tests)."""

  def test_timeout_configuration(self, gcs_source):
    """Test that timeout is configurable."""
    from adk_rlm.files.sources.gcs import GCSFileSource
    from adk_rlm.files.sources.gcs import RetryConfig

    # Create source with custom timeout
    source = GCSFileSource(
        bucket=os.environ["RLM_GCS_TEST_BUCKET"],
        timeout=5.0,
        retry_config=RetryConfig(max_attempts=2, initial_delay=0.1),
    )

    assert source.timeout == 5.0
    assert source.retry_config.max_attempts == 2


class TestGCSEdgeCases:
  """Test edge cases with real GCS."""

  def test_path_without_gs_prefix(self, gcs_source, test_file_path):
    """Test loading with path without gs:// prefix uses default bucket."""
    # This should work because GCSFileSource has default bucket set
    loaded = gcs_source.load(test_file_path)

    assert len(loaded.content) > 0

  def test_metadata_extra_fields(self, gcs_source, test_bucket, test_file_path):
    """Test that extra metadata fields are populated."""
    path = f"gs://{test_bucket}/{test_file_path}"

    metadata = gcs_source.get_metadata(path)

    # Check that GCS-specific fields are present
    assert "blob_name" in metadata.extra
    assert "storage_class" in metadata.extra
    # generation and metageneration should be present
    assert "generation" in metadata.extra
