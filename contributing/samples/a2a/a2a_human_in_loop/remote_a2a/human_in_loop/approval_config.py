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

"""Server-side, config-derived approval threshold for this sample.

The threshold that decides whether a reimbursement needs manager confirmation
must be resolved from server-side configuration only, and must never be
something a tool caller -- the model, or anyone crafting tool-call arguments --
can pass in and change. This is the general shape of a real, previously-seen
production bug: a caller-controlled value (a threshold, a confidence score, a
spend limit) silently overrode the value an approval gate was supposed to
enforce, defeating the gate. The fix is not to trust a prompt instruction to
keep the model from calling the tool for large amounts (that is what this
sample did before this change) -- it is to make the tool itself refuse,
in code, using a value the caller has no argument to influence.

Nothing in this module accepts a threshold from a caller: the only input is
an environment variable read at call time, set by whoever deploys the agent,
never by a tool-call argument.
"""

from __future__ import annotations

import os

# Name of the environment variable a deployer can set to change the
# threshold. This is intentionally NOT a parameter of `reimburse()` or any
# other tool function in this sample.
APPROVAL_THRESHOLD_ENV_VAR = 'REIMBURSEMENT_APPROVAL_THRESHOLD_USD'
DEFAULT_APPROVAL_THRESHOLD_USD = 100.0


def get_approval_threshold_usd() -> float:
  """Returns the current auto-approval threshold, in USD.

  Reads `REIMBURSEMENT_APPROVAL_THRESHOLD_USD` from the environment if set
  (server-side config), otherwise falls back to
  `DEFAULT_APPROVAL_THRESHOLD_USD`. There is deliberately no code path here
  that lets a tool-call argument, or any caller-supplied dict, influence this
  value.
  """
  raw = os.environ.get(APPROVAL_THRESHOLD_ENV_VAR)
  if raw is None:
    return DEFAULT_APPROVAL_THRESHOLD_USD
  try:
    return float(raw)
  except ValueError:
    return DEFAULT_APPROVAL_THRESHOLD_USD


def requires_manager_approval(amount: float) -> bool:
  """Whether `amount` requires manager confirmation before it is reimbursed."""
  return amount >= get_approval_threshold_usd()
