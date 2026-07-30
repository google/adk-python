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

"""Shared constants for ADK CLI telemetry collecting and reporting."""

from __future__ import annotations

import os

# Local file where the client rate-limiting backoff lock timestamp is stored.
LOCK_FILE = os.path.expanduser("~/.adk/clearcut_lock")
# Local JSONL file where command metric logs are queued before flushing.
QUEUE_FILE = os.path.expanduser("~/.adk/telemetry_queue.jsonl")
# Local directory mapping terminal parent PIDs to their active sessions.
TELEMETRY_SESSIONS_DIR = os.path.expanduser("~/.adk/telemetry_sessions")
