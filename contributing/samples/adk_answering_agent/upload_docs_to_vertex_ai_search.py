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

import os
import sys

from adk_answering_agent.settings import ADK_DOCS_ROOT_PATH
from adk_answering_agent.settings import ADK_PYTHON_ROOT_PATH
from adk_answering_agent.settings import GCS_BUCKET_NAME
from adk_answering_agent.settings import GOOGLE_CLOUD_PROJECT
from adk_answering_agent.settings import VERTEXAI_DATASTORE_ID
from google.api_core.exceptions import GoogleAPICallError
from google.cloud import discoveryengine_v1beta as discoveryengine
from google.cloud import storage
import markdown

GCS_PREFIX_TO_ROOT_PATH = {
    "adk-docs": ADK_DOCS_ROOT_PATH,
    "adk-python": ADK_PYTHON_ROOT_PATH,
}


def cleanup_gcs_prefix(project_id: str, bucket_name: str, prefix: str) -> bool:
  """Delete all the objects with the given prefix in the bucket."""
  print(f"Start cleaning up GCS: gs://{bucket_name}/{prefix}...")
  try:
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))

    if not blobs:
      print("GCS target location is already empty, no need to clean up.")
      return True

    bucket.delete_blobs(blobs)
    print(f"Successfully deleted {len(blobs)} objects.")
    return True
  except GoogleAPICallError as e:
    print(f"[ERROR] Failed to clean up GCS: {e}", file=sys.stderr)
    return False


def upload_directory_to_gcs(
    source_directory: str, project_id: str, bucket_name: str, prefix: str
) -> bool:
  """Upload the whole directory into GCS."""
  print(
      f"Start uploading directory {source_directory} to GCS:"
      f" gs://{bucket_name}/{prefix}..."
  )

  if not os.path.isdir(source_directory):
    print(f"[Error] {source_directory} is not a directory or does not exist.")
    return False

  storage_client = storage.Client(project=project_id)
  bucket = storage_client.bucket(bucket_name)
  file_count = 0
  for root, dirs, files in os.walk(source_directory):
    # Modify the 'dirs' list in-place to prevent os.walk from descending
    # into hidden directories.
    dirs[:] = [d for d in dirs if not d.startswith(".")]

    # Keep only .md and .py files.
    files = [f for f in files if f.endswith(".md") or f.endswith(".py")]

    for filename in files:
      local_path = os.path.join(root, filename)

      relative_path = os.path.relpath(local_path, source_directory)
      gcs_path = os.path.join(prefix, relative_path)

      try:
        content_type = None
        if filename.lower().endswith(".md"):
          # Vertex AI search doesn't recognize text/markdown,
          # convert it to html and use text/html instead
          content_type = "text/html"
          with open(local_path, "r", encoding="utf-8") as f:
            md_content = f.read()
          html_content = markdown.markdown(
              md_content, output_format="html5", encoding="utf-8"
          )
          if not html_content:
            print("  - Skipped empty file: " + local_path)
            continue
          gcs_path = gcs_path.removesuffix(".md") + ".html"
          bucket.blob(gcs_path).upload_from_string(
              html_content, content_type=content_type
          )
        else:  # Python files
          bucket.blob(gcs_path).upload_from_filename(
              local_path, content_type=content_type
          )
        type_msg = (
            f"(type {content_type})" if content_type else "(type auto-detect)"
        )
        print(
            f"  - Uploaded {type_msg}: {local_path} ->"
            f" gs://{bucket_name}/{gcs_path}"
        )
        file_count += 1
      except GoogleAPICallError as e:
        print(
            f"[ERROR] Error uploading file {local_path}: {e}", file=sys.stderr
        )
        return False

  print(f"Sucessfully uploaded {file_count} files to GCS.")
  return True


def import_from_gcs_to_vertex_ai(
    full_datastore_id: str,
    gcs_bucket: str,
) -> bool:
  """Triggers a bulk import task from a GCS folder to Vertex AI Search."""
  print(f"Triggering FULL SYNC import from gs://{gcs_bucket}/**...")

  try:
    client = discoveryengine.DocumentServiceClient()
    gcs_uri = f"gs://{gcs_bucket}/**"
    request = discoveryengine.ImportDocumentsRequest(
        # parent has the format of
        # "projects/{project_number}/locations/{location}/collections/{collection}/dataStores/{datastore_id}/branches/default_branch"
        parent=full_datastore_id + "/branches/default_branch",
        # Specify the GCS source and use "content" for unstructed data.
        gcs_source=discoveryengine.GcsSource(
            input_uris=[gcs_uri], data_schema="content"
        ),
        reconciliation_mode=discoveryengine.ImportDocumentsRequest.ReconciliationMode.FULL,
    )
    operation = client.import_documents(request=request)
    print(
        "Successfully started full sync import operation."
        f"Operation Name: {operation.operation.name}"
    )
    return True

  except GoogleAPICallError as e:
    print(f"[ERROR] Error triggering import: {e}", file=sys.stderr)
    return False


def main():
  # Check required environment variables.
  if not GOOGLE_CLOUD_PROJECT:
    print(
        "[ERROR] GOOGLE_CLOUD_PROJECT environment variable not set. Exiting...",
        file=sys.stderr,
    )
    return 1
  if not GCS_BUCKET_NAME:
    print(
        "[ERROR] GCS_BUCKET_NAME environment variable not set. Exiting...",
        file=sys.stderr,
    )
    return 1
  if not VERTEXAI_DATASTORE_ID:
    print(
        "[ERROR] VERTEXAI_DATASTORE_ID environment variable not set."
        " Exiting...",
        file=sys.stderr,
    )
    return 1
  if not ADK_DOCS_ROOT_PATH:
    print(
        "[ERROR] ADK_DOCS_ROOT_PATH environment variable not set. Exiting...",
        file=sys.stderr,
    )
    return 1
  if not ADK_PYTHON_ROOT_PATH:
    print(
        "[ERROR] ADK_PYTHON_ROOT_PATH environment variable not set. Exiting...",
        file=sys.stderr,
    )
    return 1

  for gcs_prefix in GCS_PREFIX_TO_ROOT_PATH:
    # 1. Cleanup the GSC for a clean start.
    if not cleanup_gcs_prefix(
        GOOGLE_CLOUD_PROJECT, GCS_BUCKET_NAME, gcs_prefix
    ):
      print("[ERROR] Failed to clean up GCS. Exiting...", file=sys.stderr)
      return 1

    # 2. Upload the docs to GCS.
    if not upload_directory_to_gcs(
        GCS_PREFIX_TO_ROOT_PATH[gcs_prefix],
        GOOGLE_CLOUD_PROJECT,
        GCS_BUCKET_NAME,
        gcs_prefix,
    ):
      print("[ERROR] Failed to upload docs to GCS. Exiting...", file=sys.stderr)
      return 1

  # 3. Import the docs from GCS to Vertex AI Search.
  if not import_from_gcs_to_vertex_ai(VERTEXAI_DATASTORE_ID, GCS_BUCKET_NAME):
    print(
        "[ERROR] Failed to import docs from GCS to Vertex AI Search."
        " Exiting...",
        file=sys.stderr,
    )
    return 1

  print("--- Sync task has been successfully initiated ---")
  return 0


if __name__ == "__main__":
  sys.exit(main())
