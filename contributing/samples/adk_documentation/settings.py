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

load_dotenv(override=True)

GITHUB_BASE_URL = "https://api.github.com"

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
  raise ValueError("GITHUB_TOKEN environment variable not set")

DOC_OWNER = os.getenv("DOC_OWNER", "google")
CODE_OWNER = os.getenv("CODE_OWNER", "google")
DOC_REPO = os.getenv("DOC_REPO", "adk-docs")
CODE_REPO = os.getenv("CODE_REPO", "adk-python")
LOCAL_REPOS_DIR_PATH = os.getenv("LOCAL_REPOS_DIR_PATH", "/tmp")

IS_INTERACTIVE = os.getenv("INTERACTIVE", "1").lower() in ["true", "1"]
