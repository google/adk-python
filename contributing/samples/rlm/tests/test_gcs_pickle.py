"""
Test that GCSFileSource can be pickled.

This is important because LazyFile objects store a reference to their source,
and if the source can't be pickled, serialization of file collections will fail.
"""

import pickle

import pytest


class TestGCSFileSourcePickle:
  """Test pickling of GCSFileSource."""

  @pytest.fixture
  def gcs_source(self):
    """Create a GCSFileSource instance."""
    pytest.importorskip("google.cloud.storage")
    from adk_rlm.files.sources.gcs import GCSFileSource

    return GCSFileSource(bucket="test-bucket")

  def test_gcs_source_can_be_pickled(self, gcs_source):
    """GCSFileSource should be pickleable."""
    # Pickle and unpickle
    pickled = pickle.dumps(gcs_source)
    unpickled = pickle.loads(pickled)

    # Check that the unpickled source has the same config
    assert unpickled.default_bucket == gcs_source.default_bucket
    assert unpickled.timeout == gcs_source.timeout
    assert unpickled.max_concurrent == gcs_source.max_concurrent

  def test_gcs_source_client_is_lazy(self, gcs_source):
    """GCSFileSource client should be lazily initialized."""
    # Before accessing client, _client should be None
    assert gcs_source._client is None

    # After pickling, _client should still be None
    pickled = pickle.dumps(gcs_source)
    unpickled = pickle.loads(pickled)
    assert unpickled._client is None

  def test_gcs_source_pickle_after_client_access(self, gcs_source):
    """GCSFileSource should be pickleable even after client is accessed."""
    # Access the client to initialize it
    # Note: This may fail if no credentials are available, which is fine for this test
    try:
      _ = gcs_source.client
    except Exception:
      pytest.skip("GCS credentials not available")

    # Should still be pickleable - client should be excluded
    pickled = pickle.dumps(gcs_source)
    unpickled = pickle.loads(pickled)

    # Client should be None after unpickling (will be re-created on demand)
    assert unpickled._client is None
    assert unpickled.default_bucket == gcs_source.default_bucket

  def test_lazy_file_with_gcs_source_can_be_pickled(self, gcs_source):
    """LazyFile with GCSFileSource should be pickleable."""
    from adk_rlm.files.lazy import LazyFile

    lazy_file = LazyFile(
        path="gs://test-bucket/test.txt",
        source=gcs_source,
        parser=None,
    )

    # Should be pickleable
    pickled = pickle.dumps(lazy_file)
    unpickled = pickle.loads(pickled)

    assert unpickled.path == lazy_file.path
    assert unpickled.source.default_bucket == gcs_source.default_bucket

  def test_lazy_file_collection_with_gcs_source_can_be_pickled(
      self, gcs_source
  ):
    """LazyFileCollection with GCS files should be pickleable."""
    from adk_rlm.files.lazy import LazyFile
    from adk_rlm.files.lazy import LazyFileCollection

    files = [
        LazyFile(
            path="gs://test-bucket/file1.txt", source=gcs_source, parser=None
        ),
        LazyFile(
            path="gs://test-bucket/file2.txt", source=gcs_source, parser=None
        ),
    ]
    collection = LazyFileCollection(files)

    # Should be pickleable
    pickled = pickle.dumps(collection)
    unpickled = pickle.loads(pickled)

    assert len(unpickled) == 2
    assert unpickled[0].path == "gs://test-bucket/file1.txt"
    assert unpickled[1].path == "gs://test-bucket/file2.txt"
