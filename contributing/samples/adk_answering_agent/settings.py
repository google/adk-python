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

from dotenv import load_dotenv

load_dotenv(override=True)

GITHUB_BASE_URL = "https://api.github.com"
GITHUB_GRAPHQL_URL = GITHUB_BASE_URL + "/graphql"

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
  raise ValueError("GITHUB_TOKEN environment variable not set")

VERTEXAI_DATASTORE_ID = os.getenv("VERTEXAI_DATASTORE_ID")
if not VERTEXAI_DATASTORE_ID:
  raise ValueError("VERTEXAI_DATASTORE_ID environment variable not set")

GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
GEMINI_API_DATASTORE_ID = os.getenv("GEMINI_API_DATASTORE_ID")
ADK_GCP_SA_KEY = os.getenv("ADK_GCP_SA_KEY")

ADK_DOCS_ROOT_PATH = os.getenv("ADK_DOCS_ROOT_PATH")
ADK_PYTHON_ROOT_PATH = os.getenv("ADK_PYTHON_ROOT_PATH")

OWNER = os.getenv("OWNER", "google")
REPO = os.getenv("REPO", "adk-python")
BOT_RESPONSE_LABEL = os.getenv("BOT_RESPONSE_LABEL", "bot responded")
DISCUSSION_NUMBER = os.getenv("DISCUSSION_NUMBER")

IS_INTERACTIVE = os.getenv("INTERACTIVE", "1").lower() in ["true", "1"]
