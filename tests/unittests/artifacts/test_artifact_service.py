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

"""Tests for the artifact service."""

import enum
import io
from typing import Optional, Union
from unittest.mock import MagicMock

from google.adk.artifacts import GcsArtifactService, InMemoryArtifactService,S3ArtifactService
from google.genai import types
import pytest

Enum = enum.Enum


class ArtifactServiceType(Enum):
    IN_MEMORY = "IN_MEMORY"
    GCS = "GCS"
    S3 = "S3"


class MockBlob:
    def __init__(self, name: str) -> None:
        self.name = name
        self.content: Optional[bytes] = None
        self.content_type: Optional[str] = None

    def upload_from_string(
        self, data: Union[str, bytes], content_type: Optional[str] = None
    ) -> None:
        if isinstance(data, str):
            self.content = data.encode("utf-8")
        elif isinstance(data, bytes):
            self.content = data
        else:
            raise TypeError("data must be str or bytes")

        if content_type:
            self.content_type = content_type

    def download_as_bytes(self) -> bytes:
        if self.content is None:
            return b""
        return self.content

    def delete(self) -> None:
        self.content = None
        self.content_type = None


class MockBucket:
    def __init__(self, name: str) -> None:
        self.name = name
        self.blobs: dict[str, MockBlob] = {}

    def blob(self, blob_name: str) -> MockBlob:
        if blob_name not in self.blobs:
            self.blobs[blob_name] = MockBlob(blob_name)
        return self.blobs[blob_name]


class MockClient:
    def __init__(self) -> None:
        self.buckets: dict[str, MockBucket] = {}

    def bucket(self, bucket_name: str) -> MockBucket:
        if bucket_name not in self.buckets:
            self.buckets[bucket_name] = MockBucket(bucket_name)
        return self.buckets[bucket_name]

    def list_blobs(self, bucket: MockBucket, prefix: Optional[str] = None):
        if prefix:
            return [
                blob for name, blob in bucket.blobs.items() if name.startswith(prefix)
            ]
        return list(bucket.blobs.values())


def mock_gcs_artifact_service():
    storage_client = MockClient()
    service = GcsArtifactService(bucket_name="test_bucket", storage_client=storage_client)
    service.bucket = service.storage_client.bucket("test_bucket")
    return service


def mock_s3_artifact_service():
    storage = {}

    def put_object(Bucket, Key, Body, ContentType=None):
        storage[Key] = {
            "Body": Body,
            "ContentType": ContentType,
        }

    def get_object(Bucket, Key):
        if Key not in storage:
            from botocore.exceptions import ClientError
            raise ClientError(
                {"Error": {"Code": "NoSuchKey"}}, "GetObject"
            )
        return {
            "Body": io.BytesIO(storage[Key]["Body"]),
            "ContentType": storage[Key].get("ContentType", "application/octet-stream"),
        }

    def delete_object(Bucket, Key):
        if Key in storage:
            del storage[Key]

    def list_objects_v2(Bucket, Prefix):
        contents = []
        for key in storage:
            if key.startswith(Prefix):
                contents.append({"Key": key})
        return {"Contents": contents}

    def get_paginator(op):
        if op != "list_objects_v2":
            raise NotImplementedError

        class Paginator:
            def paginate(self, Bucket, Prefix):
                yield list_objects_v2(Bucket=Bucket, Prefix=Prefix)

        return Paginator()

    mock_s3 = MagicMock()
    mock_s3.put_object.side_effect = put_object
    mock_s3.get_object.side_effect = get_object
    mock_s3.delete_object.side_effect = delete_object
    mock_s3.get_paginator.side_effect = get_paginator

    return S3ArtifactService(bucket_name="test_bucket", s3_client=mock_s3)


def get_artifact_service(service_type: ArtifactServiceType = ArtifactServiceType.IN_MEMORY):
    if service_type == ArtifactServiceType.GCS:
        return mock_gcs_artifact_service()
    if service_type == ArtifactServiceType.S3:
        return mock_s3_artifact_service()
    return InMemoryArtifactService()


@pytest.mark.parametrize(
    "service_type", [ArtifactServiceType.IN_MEMORY, ArtifactServiceType.GCS, ArtifactServiceType.S3]
)
def test_load_empty(service_type):
    artifact_service = get_artifact_service(service_type)
    assert not artifact_service.load_artifact(
        app_name="test_app",
        user_id="test_user",
        session_id="session_id",
        filename="filename",
    )


@pytest.mark.parametrize(
    "service_type", [ArtifactServiceType.IN_MEMORY, ArtifactServiceType.GCS, ArtifactServiceType.S3]
)
def test_save_load_delete(service_type):
    artifact_service = get_artifact_service(service_type)
    artifact = types.Part.from_bytes(data=b"test_data", mime_type="text/plain")
    app_name = "app0"
    user_id = "user0"
    session_id = "123"
    filename = "file456"

    artifact_service.save_artifact(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        filename=filename,
        artifact=artifact,
    )
    assert (
        artifact_service.load_artifact(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
            filename=filename,
        )
        == artifact
    )

    artifact_service.delete_artifact(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        filename=filename,
    )
    assert not artifact_service.load_artifact(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        filename=filename,
    )


@pytest.mark.parametrize(
    "service_type", [ArtifactServiceType.IN_MEMORY, ArtifactServiceType.GCS, ArtifactServiceType.S3]
)
def test_list_keys(service_type):
    artifact_service = get_artifact_service(service_type)
    artifact = types.Part.from_bytes(data=b"test_data", mime_type="text/plain")
    app_name = "app0"
    user_id = "user0"
    session_id = "123"
    filenames = [f"filename{i}" for i in range(5)]

    for f in filenames:
        artifact_service.save_artifact(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
            filename=f,
            artifact=artifact,
        )

    assert (
        artifact_service.list_artifact_keys(
            app_name=app_name, user_id=user_id, session_id=session_id
        )
        == filenames
    )


@pytest.mark.parametrize(
    "service_type", [ArtifactServiceType.IN_MEMORY, ArtifactServiceType.GCS, ArtifactServiceType.S3]
)
def test_list_versions(service_type):
    artifact_service = get_artifact_service(service_type)

    app_name = "app0"
    user_id = "user0"
    session_id = "123"
    filename = "filename"
    versions = [
        types.Part.from_bytes(
            data=i.to_bytes(2, byteorder="big"), mime_type="text/plain"
        )
        for i in range(3)
    ]

    for part in versions:
        artifact_service.save_artifact(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
            filename=filename,
            artifact=part,
        )

    response_versions = artifact_service.list_versions(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        filename=filename,
    )

    assert response_versions == list(range(3))
