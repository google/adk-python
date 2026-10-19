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

"""The caller identity that the serving layer established for an invocation."""

from __future__ import annotations

import logging
from typing import Optional

from pydantic import BaseModel
from pydantic import ConfigDict

from ..features import FeatureName
from ..features import is_feature_enabled

logger = logging.getLogger("google_adk." + __name__)


class CallerPrincipal(BaseModel):
  """Who sent this invocation, and whether a serving layer vouched for them.

  A serving edge (for example the A2A executor) sets this from the
  authentication it actually performed on the inbound request. It is never
  derived from message content, event authorship, or transport metadata,
  because the caller controls all of those.

  The confirmation flow uses it to decide whether a human-in-the-loop approval
  can be honored. Three states are meaningful:

  - ``None`` on the invocation context: no remote trust boundary was crossed
    (for example an in-process ``Runner.run_async`` call). The caller is the
    operator by construction.
  - ``authenticated=True``: the edge verified the caller's identity.
  - ``authenticated=False``: the edge saw the request but could not vouch for
    who sent it.
  """

  model_config = ConfigDict(extra="forbid", frozen=True)
  """The pydantic model config."""

  authenticated: bool
  """True only if a serving-layer authenticator verified the caller."""

  user_name: Optional[str] = None
  """The verified identity when ``authenticated`` is True, else None."""

  source: Optional[str] = None
  """Short label for the edge that set this, for example ``"a2a"``.

  Informational only. Trust decisions must key off ``authenticated``, never
  off the source, so a transport can not be used as a proxy for identity.
  """


def caller_may_confirm(principal: Optional[CallerPrincipal]) -> bool:
  """Decides whether a tool confirmation from this caller may be honored.

  Every consumer of a human-in-the-loop answer asks this one question, so the
  agent pipeline and the workflow tool node refuse the same callers. Three
  states, and only the middle one is a refusal:

  - No principal: nothing vouched for the caller because nothing had to. The
    invocation was started in process, so the caller is the operator.
  - Principal present, not authenticated: a serving layer handled this request
    and could not say who sent it, so the approval is not known to be the
    operator's.
  - Principal present and authenticated: the serving layer verified the caller.

  The question asked here is deliberately about authentication and not about
  transport. A transport is a proxy for identity, and a proxy for identity
  fails in both directions: it refuses authenticated peers that happen to
  arrive over the wire, and it admits anyone who reaches an ungated path.

  Strict mode is the STRICT_CALLER_PRINCIPAL feature, off by default for now
  so that upgrading cannot break a working deployment that runs its A2A server
  without an authenticator; those get a warning instead. The intent is to flip
  the registry default at the next major version.

  Callers that get ``False`` should store the answer as ``confirmed=False``
  rather than drop it. Dropping it leaves the confirmation request pending with
  nothing left to resolve it, which is what made an earlier attempt at this
  guard stall every human-in-the-loop tool. ``confirmed=False`` is a state the
  framework already has a contract for -- it is what a human decline produces
  -- so the tool returns its rejection response and the turn ends with a
  reason the caller can see.

  Args:
    principal: The ``caller_principal`` of the current invocation context.

  Returns:
    True if the answer may be honored as given, False if it must be treated as
    a rejection.
  """
  if principal is None or principal.authenticated:
    return True

  if not is_feature_enabled(FeatureName.STRICT_CALLER_PRINCIPAL):
    logger.warning(
        "Honoring a tool confirmation from an unauthenticated caller"
        " (principal source %r). The serving layer could not say who sent this"
        " approval, so it is not known to be the operator's. Enable the"
        " STRICT_CALLER_PRINCIPAL feature to refuse these instead; that is"
        " intended to become the default in a future major version.",
        principal.source,
    )
    return True

  logger.error(
      "Refusing a tool confirmation from an unauthenticated caller (principal"
      " source %r). Enable authentication on the serving layer, or disable the"
      " STRICT_CALLER_PRINCIPAL feature to downgrade this to a warning.",
      principal.source,
  )
  return False
