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

from __future__ import annotations

from google.auth.credentials import Credentials

from . import client


def list_buckets(
    *,
    project_id: str,
    credentials: Credentials,
    page_size: int | None = None,
    page_token: str | None = None,
) -> dict:
  """List GCS bucket names in a Google Cloud project.

  Args:
      project_id (str): The Google Cloud project id.
      credentials (Credentials): The credentials to use for the request.
      page_size (int, optional): The maximum number of buckets to return in a
        single page.
      page_token (str, optional): A page token, received from a previous
        list_buckets call.

  Returns:
      dict: Dictionary with a list of the GCS bucket names present in the project,
        and optionally next_page_token.
  """
  try:
    gcs_client = client.get_gcs_client(
        project=project_id, credentials=credentials
    )
    list_kwargs = {}
    if page_size is not None:
      list_kwargs["max_results"] = page_size
    if page_token is not None:
      list_kwargs["page_token"] = page_token
    buckets = gcs_client.list_buckets(**list_kwargs)

    if page_size is not None:
      page = next(buckets.pages, None)
      bucket_names = [bucket.name for bucket in page] if page else []
      next_page_token = buckets.next_page_token
    else:
      bucket_names = [bucket.name for bucket in buckets]
      next_page_token = None

    response = {
        "status": "SUCCESS",
        "results": bucket_names,
    }
    if next_page_token:
      response["next_page_token"] = next_page_token

    return response
  except Exception as ex:
    return {
        "status": "ERROR",
        "error_details": str(ex),
    }


def create_bucket(
    *,
    project_id: str,
    bucket_name: str,
    credentials: Credentials,
    location: str | None = None,
) -> dict:
  """Create a new GCS bucket.

  Args:
      project_id (str): The Google Cloud project id.
      bucket_name (str): The name of the GCS bucket to create.
      credentials (Credentials): The credentials to use for the request.
      location (str, optional): The location of the bucket.

  Returns:
      dict: Dictionary indicating success or error.
  """
  try:
    gcs_client = client.get_gcs_client(
        project=project_id, credentials=credentials
    )
    bucket = gcs_client.bucket(bucket_name)
    new_bucket = gcs_client.create_bucket(bucket, location=location)
    return {
        "status": "SUCCESS",
        "results": f"Bucket {new_bucket.name} created successfully.",
    }
  except Exception as ex:
    return {
        "status": "ERROR",
        "error_details": str(ex),
    }


def update_bucket(
    *,
    bucket_name: str,
    credentials: Credentials,
    versioning_enabled: bool | None = None,
    uniform_bucket_level_access_enabled: bool | None = None,
) -> dict:
  """Update properties of a GCS bucket.

  Args:
      bucket_name (str): The name of the GCS bucket to update.
      credentials (Credentials): The credentials to use for the request.
      versioning_enabled (bool, optional): Whether to enable versioning for the
        bucket.
      uniform_bucket_level_access_enabled (bool, optional): Whether to enable
        uniform bucket-level access.

  Returns:
      dict: Dictionary indicating success or error.
  """
  try:
    gcs_client = client.get_gcs_client(credentials=credentials)
    bucket = gcs_client.get_bucket(bucket_name)
    if versioning_enabled is not None:
      bucket.versioning_enabled = versioning_enabled
    if uniform_bucket_level_access_enabled is not None:
      bucket.iam_configuration.uniform_bucket_level_access_enabled = (
          uniform_bucket_level_access_enabled
      )

    if (
        versioning_enabled is not None
        or uniform_bucket_level_access_enabled is not None
    ):
      bucket.patch()

    return {
        "status": "SUCCESS",
        "results": f"Bucket {bucket.name} updated successfully.",
    }
  except Exception as ex:
    return {
        "status": "ERROR",
        "error_details": str(ex),
    }


def delete_bucket(*, bucket_name: str, credentials: Credentials) -> dict:
  """Delete a GCS bucket.

  Args:
      bucket_name (str): The name of the GCS bucket to delete.
      credentials (Credentials): The credentials to use for the request.

  Returns:
      dict: Dictionary indicating success or error.
  """
  try:
    gcs_client = client.get_gcs_client(credentials=credentials)
    bucket = gcs_client.get_bucket(bucket_name)
    bucket.delete()
    return {
        "status": "SUCCESS",
        "results": f"Bucket {bucket_name} deleted successfully.",
    }
  except Exception as ex:
    return {
        "status": "ERROR",
        "error_details": str(ex),
    }
