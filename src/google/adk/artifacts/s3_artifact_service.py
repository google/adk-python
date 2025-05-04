"""An artifact service implementation using Amazon S3."""

import logging
from typing import Optional

import boto3
from botocore.exceptions import ClientError
from google.genai import types
from typing_extensions import override

from .base_artifact_service import BaseArtifactService
from botocore.client import BaseClient
logger = logging.getLogger(__name__)


class S3ArtifactService(BaseArtifactService):
    """An artifact service implementation using Amazon S3."""

    def __init__(self, bucket_name: str,s3_client: Optional[BaseClient], **kwargs):
        """Initializes the S3ArtifactService.

        Args:
            bucket_name: The name of the S3 bucket to use.
            **kwargs: Optional parameters for boto3 client configuration.
        """
        self.bucket_name = bucket_name
        if s3_client is None:
            self.s3 = boto3.client("s3", **kwargs)
        else:
            self.s3 = s3_client

    def _file_has_user_namespace(self, filename: str) -> bool:
        """Checks if the filename has a user namespace."""
        return filename.startswith("user:")

    def _get_object_key(
        self,
        app_name: str,
        user_id: str,
        session_id: str,
        filename: str,
        version: int | str,
    ) -> str:
        """Constructs the S3 object key."""
        if self._file_has_user_namespace(filename):
            return f"{app_name}/{user_id}/user/{filename}/{version}"
        return f"{app_name}/{user_id}/{session_id}/{filename}/{version}"

    @override
    def save_artifact(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
        filename: str,
        artifact: types.Part,
    ) -> int:
        versions = self.list_versions(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
            filename=filename,
        )
        version = 0 if not versions else max(versions) + 1

        object_key = self._get_object_key(app_name, user_id, session_id, filename, version)

        self.s3.put_object(
            Bucket=self.bucket_name,
            Key=object_key,
            Body=artifact.inline_data.data,
            ContentType=artifact.inline_data.mime_type,
        )

        return version

    @override
    def load_artifact(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
        filename: str,
        version: Optional[int] = None,
    ) -> Optional[types.Part]:
        if version is None:
            versions = self.list_versions(
                app_name=app_name,
                user_id=user_id,
                session_id=session_id,
                filename=filename,
            )
            if not versions:
                return None
            version = max(versions)

        object_key = self._get_object_key(app_name, user_id, session_id, filename, version)

        try:
            response = self.s3.get_object(Bucket=self.bucket_name, Key=object_key)
            data = response["Body"].read()
            mime_type = response.get("ContentType", "application/octet-stream")
            return types.Part.from_bytes(data=data, mime_type=mime_type)
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                return None
            raise

    @override
    def list_artifact_keys(
        self, *, app_name: str, user_id: str, session_id: str
    ) -> list[str]:
        keys = set()

        prefixes = [
            f"{app_name}/{user_id}/{session_id}/",
            f"{app_name}/{user_id}/user/",
        ]

        for prefix in prefixes:
            paginator = self.s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
                for obj in page.get("Contents", []):
                    parts = obj["Key"].split("/")
                    if len(parts) >= 5:
                        filename = parts[3]
                        keys.add(filename)

        return sorted(keys)

    @override
    def delete_artifact(
        self, *, app_name: str, user_id: str, session_id: str, filename: str
    ) -> None:
        versions = self.list_versions(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
            filename=filename,
        )
        for version in versions:
            object_key = self._get_object_key(app_name, user_id, session_id, filename, version)
            try:
                self.s3.delete_object(Bucket=self.bucket_name, Key=object_key)
            except ClientError as e:
                logger.warning(f"Failed to delete {object_key}: {e}")

    @override
    def list_versions(
        self, *, app_name: str, user_id: str, session_id: str, filename: str
    ) -> list[int]:
        prefix = self._get_object_key(app_name, user_id, session_id, filename, "")
        versions = []

        paginator = self.s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
            for obj in page.get("Contents", []):
                parts = obj["Key"].split("/")
                if len(parts) >= 5:
                    try:
                        version = int(parts[4])
                        versions.append(version)
                    except ValueError:
                        continue  # Skip non-integer versions

        return sorted(versions)
