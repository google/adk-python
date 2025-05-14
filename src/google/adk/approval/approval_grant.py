from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel


class ApprovalActor(BaseModel):
  id: str
  type: str = Literal["user", "agent", "tool"]
  on_behalf_of: ApprovalActor | None = None


class ApprovalEffect(str, Enum):
  allow = "allow"
  deny = "deny"
  challenge = "challenge"


ApprovalAction = str
ApprovalResource = str


class ApprovalGrant(BaseModel):
  """Effect the actions on the resources to the grantee by the grantor until the expiration."""

  effect: Literal[ApprovalEffect.allow, ApprovalEffect.deny]
  """Whether to grant an allow or deny."""
  actions: list[ApprovalAction]
  """The actions to which the grant will effect."""
  resources: list[ApprovalResource]
  """The resources that this grant affects."""
  grantee: ApprovalActor
  """Who the grant applies to."""
  grantor: ApprovalActor
  """The permission holder that granted toe access (e.g. user, or delegated by a parent agent)."""
  expiration_time: Optional[datetime] = None  # Optional expiration time
  """The time after which the grant ceases to be valid."""

  comment: Optional[str] = None
  """Comment from the grantor (typically the end user) from the point of granting. This is used when communicating a grant update to a model, for example a deny, to explain the reason."""
