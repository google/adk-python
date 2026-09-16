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

from typing import Optional

from pydantic import BaseModel
from pydantic import ConfigDict


class CallerPrincipal(BaseModel):
  """Who submitted this invocation, and whether the serving layer vouched for them.

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
