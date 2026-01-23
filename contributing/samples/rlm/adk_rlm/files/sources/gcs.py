"""
Google Cloud Storage file source implementation.

Provides file loading from GCS buckets with glob pattern support,
retry logic, and efficient metadata access.
"""

from concurrent.futures import as_completed
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import fnmatch
import mimetypes
from pathlib import Path
import tempfile
import time
from typing import Callable
from typing import Iterator
from typing import TypeVar

from adk_rlm.files.base import FileMetadata
from adk_rlm.files.base import LoadedFile
from adk_rlm.files.sources.base import FileSource

try:
  from google.cloud import storage
  from google.cloud.exceptions import Forbidden
  from google.cloud.exceptions import NotFound

  HAS_GCS = True
except ImportError:
  HAS_GCS = False
  storage = None  # type: ignore
  NotFound = Exception  # type: ignore
  Forbidden = Exception  # type: ignore

T = TypeVar("T")


@dataclass
class RetryConfig:
  """Configuration for retry behavior on transient errors."""

  max_attempts: int = 3
  initial_delay: float = 0.5
  max_delay: float = 30.0
  exponential_base: float = 2.0


class GCSFileSource(FileSource):
  """
  Load files from Google Cloud Storage.

  Supports gs:// URIs with glob patterns for batch file resolution.
  Uses Application Default Credentials by default.

  Example:
      ```python
      source = GCSFileSource(bucket="my-bucket")

      # Single file
      file = source.load("gs://my-bucket/data/report.pdf")

      # Glob pattern
      paths = source.resolve("gs://my-bucket/data/**/*.pdf")
      for path in paths:
          file = source.load(path)

      # With explicit credentials
      source = GCSFileSource(
          bucket="my-bucket",
          credentials_path="/path/to/service-account.json"
      )
      ```
  """

  def __init__(
      self,
      bucket: str | None = None,
      project: str | None = None,
      credentials: "storage.Client | None" = None,
      credentials_path: str | None = None,
      timeout: float = 60.0,
      retry_config: RetryConfig | None = None,
      max_concurrent: int = 10,
      large_file_threshold: int = 100_000_000,  # 100 MB
  ):
    """
    Initialize GCSFileSource.

    Args:
        bucket: Default bucket name (can be overridden in paths)
        project: GCP project ID (optional, inferred from credentials)
        credentials: Explicit google.auth.credentials.Credentials object
        credentials_path: Path to service account JSON file
        timeout: Request timeout in seconds
        retry_config: Retry configuration for transient errors
        max_concurrent: Max parallel downloads in load_many()
        large_file_threshold: Files larger than this stream to temp file
    """
    if not HAS_GCS:
      raise ImportError(
          "GCS support requires 'google-cloud-storage'. "
          "Install with: pip install google-cloud-storage"
      )

    self.default_bucket = bucket
    self._project = project
    self._credentials = credentials
    self._credentials_path = credentials_path
    self.timeout = timeout
    self.retry_config = retry_config or RetryConfig()
    self.max_concurrent = max_concurrent
    self.large_file_threshold = large_file_threshold

    # Client is lazily initialized to avoid pickle issues
    self._client: "storage.Client | None" = None

  @property
  def client(self) -> "storage.Client":
    """Lazily initialize and return the GCS client."""
    if self._client is None:
      if self._credentials:
        self._client = storage.Client(
            credentials=self._credentials, project=self._project
        )
      elif self._credentials_path:
        self._client = storage.Client.from_service_account_json(
            self._credentials_path, project=self._project
        )
      else:
        # Use Application Default Credentials
        self._client = storage.Client(project=self._project)
    return self._client

  def __getstate__(self):
    """Return state for pickling, excluding the client."""
    state = self.__dict__.copy()
    # Don't pickle the client - it will be recreated on demand
    state["_client"] = None
    return state

  def __setstate__(self, state):
    """Restore state from pickle."""
    self.__dict__.update(state)

  @property
  def source_type(self) -> str:
    """Return 'gcs' as the source type."""
    return "gcs"

  def _parse_path(self, path: str) -> tuple[str, str]:
    """
    Parse a GCS path into bucket and blob name.

    Args:
        path: GCS path (gs://bucket/key or just key for default bucket)

    Returns:
        Tuple of (bucket_name, blob_name)
    """
    if path.startswith("gs://"):
      path = path[5:]
      parts = path.split("/", 1)
      bucket_name = parts[0]
      blob_name = parts[1] if len(parts) > 1 else ""
    else:
      bucket_name = self.default_bucket
      blob_name = path

    if not bucket_name:
      raise ValueError("No bucket specified and no default bucket set")

    return bucket_name, blob_name

  def _is_retryable(self, error: Exception) -> bool:
    """Check if an error is retryable."""
    error_name = type(error).__name__
    error_str = str(error).lower()
    retryable_names = (
        "ServiceUnavailable",
        "TooManyRequests",
        "InternalError",
        "Timeout",
        "ConnectionError",
    )
    retryable_messages = (
        "serviceunavailable",
        "toomanyrequests",
        "internalerror",
        "timeout",
        "connectionerror",
        "connection reset",
        "connection refused",
    )
    return error_name in retryable_names or any(
        msg in error_str for msg in retryable_messages
    )

  def _with_retry(self, operation: Callable[[], T], context: str = "") -> T:
    """Execute operation with retry logic for transient errors."""
    last_error: Exception | None = None
    delay = self.retry_config.initial_delay

    for attempt in range(self.retry_config.max_attempts):
      try:
        return operation()
      except NotFound:
        raise FileNotFoundError(f"GCS object not found: {context}")
      except Forbidden as e:
        raise PermissionError(
            f"Access denied to GCS object: {context}. "
            f"Check bucket permissions and credentials. Error: {e}"
        )
      except Exception as e:
        if self._is_retryable(e):
          last_error = e
          if attempt < self.retry_config.max_attempts - 1:
            time.sleep(delay)
            delay = min(
                delay * self.retry_config.exponential_base,
                self.retry_config.max_delay,
            )
        else:
          raise

    raise RuntimeError(
        f"GCS operation failed after {self.retry_config.max_attempts} attempts:"
        f" {context}. Last error: {last_error}"
    )

  def resolve(self, path: str) -> list[str]:
    """
    Resolve GCS path, supporting glob patterns.

    For glob patterns, lists blobs with prefix matching and filters
    with fnmatch. Note: listing large buckets can be slow.

    Args:
        path: GCS path or glob pattern (e.g., "gs://bucket/data/**/*.pdf")

    Returns:
        List of resolved gs:// URIs
    """
    bucket_name, pattern = self._parse_path(path)

    # Check for glob patterns
    if not any(c in pattern for c in ["*", "?", "["]):
      # Not a glob - check if single object exists
      bucket = self.client.bucket(bucket_name)
      blob = bucket.blob(pattern)

      def check_exists():
        return blob.exists(timeout=self.timeout)

      if self._with_retry(check_exists, f"gs://{bucket_name}/{pattern}"):
        return [f"gs://{bucket_name}/{pattern}"]
      return []

    # Extract prefix (everything before first glob char)
    prefix_end = len(pattern)
    for char in ["*", "?", "["]:
      idx = pattern.find(char)
      if idx != -1:
        prefix_end = min(prefix_end, idx)

    prefix = pattern[:prefix_end]
    # Also trim to last / to get directory prefix
    if "/" in prefix:
      prefix = prefix.rsplit("/", 1)[0] + "/"
    else:
      prefix = ""

    # List blobs with prefix and filter
    bucket = self.client.bucket(bucket_name)
    results: list[str] = []

    def list_blobs():
      return list(bucket.list_blobs(prefix=prefix, timeout=self.timeout))

    blobs = self._with_retry(list_blobs, f"listing gs://{bucket_name}/{prefix}")

    for blob in blobs:
      if fnmatch.fnmatch(blob.name, pattern):
        results.append(f"gs://{bucket_name}/{blob.name}")

    return sorted(results)

  def get_metadata(self, path: str) -> FileMetadata:
    """
    Get blob metadata without downloading content.

    This is efficient for Level 1 lazy loading - only fetches
    metadata, not the blob content.

    Args:
        path: GCS path

    Returns:
        FileMetadata for the blob
    """
    bucket_name, blob_name = self._parse_path(path)
    bucket = self.client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    def reload_metadata():
      blob.reload(timeout=self.timeout)
      return blob

    blob = self._with_retry(reload_metadata, f"gs://{bucket_name}/{blob_name}")

    # Determine MIME type
    mime_type = blob.content_type
    if not mime_type:
      mime_type, _ = mimetypes.guess_type(blob_name)

    # Parse last modified
    last_modified = None
    if blob.updated:
      last_modified = blob.updated

    return FileMetadata(
        name=blob_name.split("/")[-1],
        path=f"gs://{bucket_name}/{blob_name}",
        source_type=self.source_type,
        size_bytes=blob.size or 0,
        mime_type=mime_type,
        last_modified=last_modified,
        extra={
            "bucket": bucket_name,
            "blob_name": blob_name,
            "content_encoding": blob.content_encoding,
            "storage_class": blob.storage_class,
            "generation": blob.generation,
            "metageneration": blob.metageneration,
            "etag": blob.etag,
            "md5_hash": blob.md5_hash,
            "crc32c": blob.crc32c,
        },
    )

  def _load_direct(self, path: str) -> LoadedFile:
    """Load file directly into memory."""
    bucket_name, blob_name = self._parse_path(path)
    bucket = self.client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    def download():
      return blob.download_as_bytes(timeout=self.timeout)

    content = self._with_retry(download, f"gs://{bucket_name}/{blob_name}")

    # Reload blob to get metadata after download
    try:
      blob.reload(timeout=self.timeout)
    except Exception:
      pass  # Metadata fetch is best-effort after download

    # Build metadata
    mime_type = blob.content_type
    if not mime_type:
      mime_type, _ = mimetypes.guess_type(blob_name)

    return LoadedFile(
        metadata=FileMetadata(
            name=blob_name.split("/")[-1],
            path=f"gs://{bucket_name}/{blob_name}",
            source_type=self.source_type,
            size_bytes=len(content),
            mime_type=mime_type,
            last_modified=blob.updated,
            extra={
                "bucket": bucket_name,
                "blob_name": blob_name,
                "etag": blob.etag,
            },
        ),
        content=content,
    )

  def _load_chunked(self, path: str, metadata: FileMetadata) -> LoadedFile:
    """Load large file via temp file to manage memory."""
    bucket_name, blob_name = self._parse_path(path)
    bucket = self.client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    def download_to_file():
      with tempfile.NamedTemporaryFile(delete=False) as tmp:
        blob.download_to_file(tmp, timeout=self.timeout)
        tmp.flush()
        return tmp.name

    tmp_path = self._with_retry(
        download_to_file, f"gs://{bucket_name}/{blob_name}"
    )

    try:
      content = Path(tmp_path).read_bytes()
    finally:
      Path(tmp_path).unlink(missing_ok=True)

    return LoadedFile(metadata=metadata, content=content)

  def load(self, path: str) -> LoadedFile:
    """
    Load file from GCS.

    For files larger than large_file_threshold, streams to a
    temp file first to manage memory.

    Args:
        path: GCS path (gs://bucket/key or key for default bucket)

    Returns:
        LoadedFile with content and metadata
    """
    # Check size first to decide loading strategy
    try:
      metadata = self.get_metadata(path)
    except FileNotFoundError:
      raise

    if metadata.size_bytes > self.large_file_threshold:
      return self._load_chunked(path, metadata)
    else:
      return self._load_direct(path)

  def load_many(self, paths: list[str]) -> Iterator[LoadedFile]:
    """
    Load multiple files in parallel.

    Uses ThreadPoolExecutor for concurrent downloads.

    Args:
        paths: List of GCS paths to load

    Yields:
        LoadedFile for each path (order not guaranteed)
    """
    if len(paths) == 0:
      return

    if len(paths) == 1:
      yield self.load(paths[0])
      return

    with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
      futures = {executor.submit(self.load, path): path for path in paths}
      for future in as_completed(futures):
        try:
          yield future.result()
        except Exception as e:
          # Re-raise with path context
          path = futures[future]
          raise RuntimeError(f"Failed to load {path}: {e}") from e

  def exists(self, path: str) -> bool:
    """
    Check if a blob exists.

    Args:
        path: GCS path to check

    Returns:
        True if blob exists, False otherwise
    """
    try:
      bucket_name, blob_name = self._parse_path(path)
      bucket = self.client.bucket(bucket_name)
      blob = bucket.blob(blob_name)
      return blob.exists(timeout=self.timeout)
    except Exception:
      return False
