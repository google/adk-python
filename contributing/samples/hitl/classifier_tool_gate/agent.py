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

import sqlite3
from typing import Any

from classifier_tool_gate.tool_gate import create_tool_gate_callback
from google.adk import Agent
from google.adk.labs.classifiers import LlmClassifier

_db = sqlite3.connect(":memory:", check_same_thread=False)
_db.executescript("""
    CREATE TABLE customers (
        id INTEGER PRIMARY KEY, name TEXT, email TEXT, plan TEXT
    );
    INSERT INTO customers VALUES
        (1, 'Ada Lovelace', 'ada@example.com', 'pro'),
        (2, 'Alan Turing', 'alan@example.com', 'free'),
        (3, 'Grace Hopper', 'grace@example.com', 'pro');
""")


def run_sql(query: str) -> dict[str, Any]:
  """Runs one SQLite statement against the `customers` table.

  The table has columns id, name, email and plan.
  """
  try:
    cursor = _db.execute(query)
    rows = cursor.fetchall()
    _db.commit()
  except sqlite3.Error as e:
    return {"error": str(e)}
  return {"rows": [list(row) for row in rows], "rows_changed": cursor.rowcount}


def send_email(to: str, subject: str, body: str) -> dict[str, str]:
  """Sends an email to a customer."""
  return {"result": f"Sent {subject!r} to {to}."}


# Any BaseClassifier works here, such as a dedicated decision model.
classifier = LlmClassifier(model="gemini-3.5-flash")

tool_gate = create_tool_gate_callback(
    classifier=classifier,
    instructions=(
        "Would this tool call delete or change data, or contact a customer?"
    ),
    on_threshold_exceeded="confirm",
)

root_agent = Agent(
    name="customer_db_assistant",
    instruction=(
        "You help support staff look up and manage customers. Use `run_sql` to"
        " query or change the `customers` table, and `send_email` to email a"
        " customer. Report tool results to the user."
    ),
    tools=[run_sql, send_email],
    before_tool_callback=tool_gate,
)
