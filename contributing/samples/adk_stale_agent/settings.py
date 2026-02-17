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

import os

from dotenv import load_dotenv

# Load environment variables from a .env file for local testing
load_dotenv(override=True)

# --- GitHub API Configuration ---
GITHUB_BASE_URL = "https://api.github.com"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
  raise ValueError("GITHUB_TOKEN environment variable not set")

OWNER = os.getenv("OWNER", "google")
REPO = os.getenv("REPO", "adk-python")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemini-2.5-flash")

STALE_LABEL_NAME = "stale"
REQUEST_CLARIFICATION_LABEL = "request clarification"

# --- THRESHOLDS IN HOURS ---
# Default: 168 hours (7 days)
# The number of hours of inactivity after a maintainer comment before an issue is marked as stale.
STALE_HOURS_THRESHOLD = float(os.getenv("STALE_HOURS_THRESHOLD", 168))

# Default: 168 hours (7 days)
# The number of hours of inactivity after an issue is marked 'stale' before it is closed.
CLOSE_HOURS_AFTER_STALE_THRESHOLD = float(
    os.getenv("CLOSE_HOURS_AFTER_STALE_THRESHOLD", 168)
)

# --- Performance Configuration ---
# The number of issues to process concurrently.
# Higher values are faster but increase the immediate rate of API calls
CONCURRENCY_LIMIT = int(os.getenv("CONCURRENCY_LIMIT", 3))

# --- GraphQL Query Limits ---
# The number of most recent comments to fetch for context analysis.
GRAPHQL_COMMENT_LIMIT = int(os.getenv("GRAPHQL_COMMENT_LIMIT", 30))

# The number of most recent description edits to fetch.
GRAPHQL_EDIT_LIMIT = int(os.getenv("GRAPHQL_EDIT_LIMIT", 10))

# The number of most recent timeline events (labels, renames, reopens) to fetch.
GRAPHQL_TIMELINE_LIMIT = int(os.getenv("GRAPHQL_TIMELINE_LIMIT", 20))

# --- Rate Limiting ---
# Time in seconds to wait between processing chunks.
SLEEP_BETWEEN_CHUNKS = float(os.getenv("SLEEP_BETWEEN_CHUNKS", 1.5))
