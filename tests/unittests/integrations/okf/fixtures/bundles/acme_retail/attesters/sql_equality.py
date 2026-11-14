# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Deterministic attester for Attested Computations with runtime: bigquery.

Verifies two things about a receipt produced by skills/run-on-bq.md:

  1. Provenance: the SQL that actually ran (`receipt.executed_sql`) equals the
     computation body of the concept (or the file at `computation:`) after
     normalizing whitespace, casing of keywords, and comment stripping.
     Named bind variables (@name) are compared symbolically; their values are
     not inspected here (the executor is trusted to bind).

  2. Fidelity: the value the caller displayed to the user (`claimed_value`)
     equals the first cell of `receipt.result`.

Returns a verdict dict:
  { "ok": bool, "reason": str | None, "details": { ... } }

Never uses an LLM. Never makes network calls. Safe to run consumer-side.
"""

from __future__ import annotations

import re
from typing import Any

# ---- normalization helpers ---------------------------------------------------

_COMMENT_LINE = re.compile(r"--[^\n]*")
_COMMENT_BLOCK = re.compile(r"/\*.*?\*/", flags=re.DOTALL)
_WHITESPACE = re.compile(r"\s+")
_KEYWORDS = frozenset({
    "SELECT",
    "FROM",
    "WHERE",
    "GROUP",
    "BY",
    "ORDER",
    "HAVING",
    "JOIN",
    "LEFT",
    "RIGHT",
    "INNER",
    "OUTER",
    "ON",
    "AS",
    "AND",
    "OR",
    "SUM",
    "COUNT",
    "AVG",
    "MIN",
    "MAX",
    "CASE",
    "WHEN",
    "THEN",
    "ELSE",
    "END",
    "WITH",
    "UNION",
    "ALL",
    "DISTINCT",
    "LIMIT",
    "OFFSET",
})


def _canonicalize(sql: str) -> str:
  """Strip comments, collapse whitespace, uppercase keywords."""
  s = _COMMENT_BLOCK.sub(" ", sql)
  s = _COMMENT_LINE.sub(" ", s)
  s = _WHITESPACE.sub(" ", s).strip()

  # Uppercase only known SQL keywords; leave identifiers alone.
  def _upper_kw(match: re.Match[str]) -> str:
    w = match.group(0)
    return w.upper() if w.upper() in _KEYWORDS else w

  s = re.sub(r"[A-Za-z_][A-Za-z_0-9]*", _upper_kw, s)
  return s


# ---- entrypoint --------------------------------------------------------------


def attest(
    *,
    sanctioned_sql: str,
    receipt: dict[str, Any],
    claimed_value: Any,
) -> dict[str, Any]:
  """Verify a BigQuery receipt against a sanctioned computation.

  Args:
    sanctioned_sql: the SQL from the concept's `# Computation` fence or its
                    `computation:` file.
    receipt:        the dict returned by skills/run-on-bq.md,
                    expected to carry `executed_sql` and `result`.
    claimed_value:  the value the caller is about to display.

  Returns:
    A verdict dict. Callers MUST refuse to display `claimed_value` when
    verdict["ok"] is False.
  """
  executed = receipt.get("executed_sql")
  if not executed:
    return {
        "ok": False,
        "reason": "receipt is missing executed_sql",
        "details": {"receipt_keys": sorted(receipt)},
    }

  can_sanctioned = _canonicalize(sanctioned_sql)
  can_executed = _canonicalize(executed)

  if can_sanctioned != can_executed:
    return {
        "ok": False,
        "reason": "executed SQL does not match sanctioned computation",
        "details": {
            "sanctioned_canonical": can_sanctioned,
            "executed_canonical": can_executed,
        },
    }

  result = receipt.get("result")
  if result is None:
    return {
        "ok": False,
        "reason": "receipt is missing result",
        "details": {},
    }

  # `result` may be a scalar (single cell) or a row-shaped list.
  first_cell = result[0] if isinstance(result, list) else result
  if first_cell != claimed_value:
    return {
        "ok": False,
        "reason": "claimed value does not match receipt result",
        "details": {"claimed": claimed_value, "receipt_first_cell": first_cell},
    }

  return {
      "ok": True,
      "reason": None,
      "details": {"job_id": receipt.get("job_id")},
  }
