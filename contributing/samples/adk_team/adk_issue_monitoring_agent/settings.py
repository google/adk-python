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
from pathlib import Path

from dotenv import load_dotenv

CURRENT_DIR = Path(__file__).resolve().parent
ENV_PATH = CURRENT_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

GITHUB_BASE_URL = "https://api.github.com"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
  raise ValueError("GITHUB_TOKEN environment variable not set")

OWNER = os.getenv("OWNER", "google")
REPO = os.getenv("REPO", "adk-python")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemini-2.5-flash")

SPAM_LABEL_NAME = os.getenv("SPAM_LABEL_NAME", "spam")
CONCURRENCY_LIMIT = int(os.getenv("CONCURRENCY_LIMIT", 3))
BOT_NAME = os.getenv("BOT_NAME", "adk-bot")
BOT_ALERT_SIGNATURE = os.getenv(
    "BOT_ALERT_SIGNATURE", "🚨 **Automated Spam Detection Alert** 🚨"
)
SLEEP_BETWEEN_CHUNKS = float(os.getenv("SLEEP_BETWEEN_CHUNKS", 1.5))


# Toggle for the initial run
INITIAL_FULL_SCAN = os.getenv("INITIAL_FULL_SCAN", "false").lower() == "true"
