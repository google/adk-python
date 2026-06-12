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

import base64

from google.auth.credentials import Credentials

from . import client


def get_bucket(*, bucket_name: str, credentials: Credentials) -> dict:
  """Get metadata information about a GCS bucket.

  Args:
      bucket_name (str): The name of the GCS bucket.
      credentials (Credentials): The credentials to use for the request.

  Returns:
      dict: Dictionary representing the properties of the bucket.
  """
  try:
    gcs_client = client.get_gcs_client(credentials=credentials)
    bucket = gcs_client.get_bucket(bucket_name)
    results = getattr(bucket, "_properties", {}).copy()
    return {
        "status": "SUCCESS",
        "results": results,
    }
  except Exception as ex:
    return {
        "status": "ERROR",
        "error_details": str(ex),
    }


def list_objects(
    *,
    bucket_name: str,
    credentials: Credentials,
    prefix: str | None = None,
    page_size: int | None = None,
    page_token: str | None = None,
) -> dict:
  """List object names in a GCS bucket.

  Args:
      bucket_name (str): The name of the GCS bucket.
      credentials (Credentials): The credentials to use for the request.
      prefix (str, optional): Filter results to objects whose names begin with
        this prefix.
      page_size (int, optional): The maximum number of objects to return in a
        single page.
      page_token (str, optional): A page token, received from a previous
        list_objects call.

  Returns:
      dict: Dictionary with a list of the object names present in the bucket,
        and optionally next_page_token.
  """
  try:
    gcs_client = client.get_gcs_client(credentials=credentials)
    bucket = gcs_client.get_bucket(bucket_name)
    list_kwargs = {}
    if page_size is not None:
      list_kwargs["max_results"] = page_size
    if page_token is not None:
      list_kwargs["page_token"] = page_token
    if prefix is not None:
      list_kwargs["prefix"] = prefix
    blobs = bucket.list_blobs(**list_kwargs)
    if page_size is not None:
      page = next(blobs.pages, None)
      blob_names = [blob.name for blob in page] if page else []
      next_page_token = blobs.next_page_token
    else:
      blob_names = [blob.name for blob in blobs]
      next_page_token = None

    response = {
        "status": "SUCCESS",
        "results": blob_names,
    }
    if next_page_token:
      response["next_page_token"] = next_page_token

    return response
  except Exception as ex:
    return {
        "status": "ERROR",
        "error_details": str(ex),
    }


def get_object_metadata(
    *,
    bucket_name: str,
    object_name: str,
    credentials: Credentials,
    generation: int | None = None,
) -> dict:
  """Get metadata information about a GCS object (blob).

  Args:
      bucket_name (str): The name of the GCS bucket containing the object.
      object_name (str): The name of the GCS object.
      credentials (Credentials): The credentials to use for the request.
      generation (int, optional): If present, selects a specific generation of
        this object.

  Returns:
      dict: Dictionary representing the properties of the object.
  """
  try:
    gcs_client = client.get_gcs_client(credentials=credentials)
    bucket = gcs_client.get_bucket(bucket_name)
    get_blob_kwargs = {}
    if generation is not None:
      get_blob_kwargs["generation"] = generation
    blob = bucket.get_blob(object_name, **get_blob_kwargs)
    if blob is None:
      return {
          "status": "ERROR",
          "error_details": (
              f"Object {object_name} not found in bucket {bucket_name}"
          ),
      }
    results = getattr(blob, "_properties", {}).copy()
    return {
        "status": "SUCCESS",
        "results": results,
    }
  except Exception as ex:
    return {
        "status": "ERROR",
        "error_details": str(ex),
    }


def create_object(
    *,
    bucket_name: str,
    object_name: str,
    credentials: Credentials,
    data: str | None = None,
    source_file_path: str | None = None,
) -> dict:
  """Create a new object (blob) in a GCS bucket from provided data or a local file.

  Args:
      bucket_name (str): The name of the GCS bucket.
      object_name (str): The name of the GCS object to create.
      credentials (Credentials): The credentials to use for the request.
      data (str, optional): The content to write to the object.
      source_file_path (str, optional): The local filesystem path of the file to
        upload.

  Returns:
      dict: Dictionary indicating success or error.
  """
  try:
    gcs_client = client.get_gcs_client(credentials=credentials)
    bucket = gcs_client.get_bucket(bucket_name)
    blob = bucket.blob(object_name)
    if source_file_path is not None:
      blob.upload_from_filename(source_file_path)
    elif data is not None:
      blob.upload_from_string(data)
    else:
      return {
          "status": "ERROR",
          "error_details": (
              "Either 'data' or 'source_file_path' must be provided."
          ),
      }

    return {
        "status": "SUCCESS",
        "results": (
            f"Object {object_name} created successfully in bucket"
            f" {bucket_name}."
        ),
    }
  except Exception as ex:
    return {
        "status": "ERROR",
        "error_details": str(ex),
    }


def get_object_data(
    *,
    bucket_name: str,
    object_name: str,
    credentials: Credentials,
    generation: int | None = None,
    destination_file_path: str | None = None,
) -> dict:
  """Get the content/data of a GCS object (blob).

  Args:
      bucket_name (str): The name of the GCS bucket.
      object_name (str): The name of the GCS object.
      credentials (Credentials): The credentials to use for the request.
      generation (int, optional): If present, selects a specific generation of
        this object.
      destination_file_path (str, optional): The local filesystem path to save
        the downloaded file.

  Returns:
      dict: Dictionary containing the object data as a string or confirming file
        download.
  """
  try:
    gcs_client = client.get_gcs_client(credentials=credentials)
    bucket = gcs_client.get_bucket(bucket_name)
    get_blob_kwargs = {}
    if generation is not None:
      get_blob_kwargs["generation"] = generation
    blob = bucket.get_blob(object_name, **get_blob_kwargs)
    if blob is None:
      return {
          "status": "ERROR",
          "error_details": (
              f"Object {object_name} not found in bucket {bucket_name}"
          ),
      }

    if destination_file_path is not None:
      blob.download_to_filename(destination_file_path)
      return {
          "status": "SUCCESS",
          "results": (
              f"Object {object_name} downloaded successfully to"
              f" {destination_file_path}."
          ),
      }

    raw_bytes = blob.download_as_bytes()
    try:
      content = raw_bytes.decode("utf-8")
      encoding = "text"
    except UnicodeDecodeError:
      # Encode binary to base64 and decode bytes to str for JSON serializability
      content = base64.b64encode(raw_bytes).decode("utf-8")
      encoding = "base64"

    return {
        "status": "SUCCESS",
        "results": content,
        "encoding": encoding,
    }
  except Exception as ex:
    return {
        "status": "ERROR",
        "error_details": str(ex),
    }


def delete_objects(
    *,
    bucket_name: str,
    object_names: list[str],
    credentials: Credentials,
) -> dict:
  """Delete multiple objects (blobs) from a GCS bucket.

  Note: A GCS bucket must be empty before it can be deleted. Use this tool to
  delete all objects if you intend to delete the bucket.

  Args:
      bucket_name (str): The name of the GCS bucket.
      object_names (list[str]): List of object names to delete.
      credentials (Credentials): The credentials to use for the request.

  Returns:
      dict: Dictionary indicating success or error.
  """
  try:
    gcs_client = client.get_gcs_client(credentials=credentials)
    bucket = gcs_client.get_bucket(bucket_name)
    bucket.delete_blobs(blobs=object_names)
    return {
        "status": "SUCCESS",
        "results": (
            f"Objects {object_names} deleted successfully from bucket"
            f" {bucket_name}."
        ),
    }
  except Exception as ex:
    return {
        "status": "ERROR",
        "error_details": str(ex),
    }
