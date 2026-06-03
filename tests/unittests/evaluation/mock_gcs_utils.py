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

from typing import Optional
from typing import Union


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
